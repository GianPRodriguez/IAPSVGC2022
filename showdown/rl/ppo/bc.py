"""Behavioral cloning warm-start for Masked PPO (docs/ppo_spec.md §8)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from showdown.rl.constants import OBS_DIM, SLOT_ACTION_N


@dataclass
class BCConfig:
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 4
    device: str = 'cpu'
    seed: int = 0


def masked_ce_loss(logits, actions, mask, *, F):
    """Masked cross-entropy: CE only over legal logits (IAM)."""
    from showdown.rl.ppo.network import apply_mask_logits

    z = apply_mask_logits(logits, mask)
    return F.cross_entropy(z, actions)


class BCTrainer:
    """Supervised fine-tune of ActorCritic on expert (obs, a1, a2, m1, m2)."""

    def __init__(self, cfg: Optional[BCConfig] = None):
        self.cfg = cfg or BCConfig()
        self._torch = None
        self.net = None
        self.optim = None
        self._rng = np.random.default_rng(int(self.cfg.seed))
        self._ensure()

    def _ensure(self) -> None:
        if self.net is not None:
            return
        import torch
        import torch.nn.functional as F
        from showdown.rl.ppo.network import ActorCritic

        self._torch = torch
        self._F = F
        self.device = torch.device(self.cfg.device)
        self.net = ActorCritic().to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

    def train_epoch(
        self,
        obs: np.ndarray,
        a1: np.ndarray,
        a2: np.ndarray,
        m1: np.ndarray,
        m2: np.ndarray,
    ) -> Dict[str, float]:
        self._ensure()
        torch = self._torch
        F = self._F
        n = int(obs.shape[0])
        if n == 0:
            return {'bc_loss': float('nan'), 'bc_acc_legal': float('nan')}
        idx = self._rng.permutation(n)
        losses = []
        correct = 0
        total = 0
        self.net.train()
        bs = max(1, int(self.cfg.batch_size))
        for start in range(0, n, bs):
            sl = idx[start:start + bs]
            o = torch.as_tensor(obs[sl], dtype=torch.float32, device=self.device)
            A1 = torch.as_tensor(a1[sl], dtype=torch.long, device=self.device)
            A2 = torch.as_tensor(a2[sl], dtype=torch.long, device=self.device)
            M1 = torch.as_tensor(m1[sl], dtype=torch.float32, device=self.device)
            M2 = torch.as_tensor(m2[sl], dtype=torch.float32, device=self.device)
            z1, z2, _v = self.net(o, a1=A1)
            assert z2 is not None
            loss = masked_ce_loss(z1, A1, M1, F=F) + masked_ce_loss(z2, A2, M2, F=F)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()
            losses.append(float(loss.item()))
            with torch.no_grad():
                from showdown.rl.ppo.network import apply_mask_logits
                p1 = apply_mask_logits(z1, M1).argmax(-1)
                p2 = apply_mask_logits(z2, M2).argmax(-1)
                correct += int(((p1 == A1) & (p2 == A2)).sum().item())
                total += int(A1.shape[0])
        return {
            'bc_loss': float(np.mean(losses)) if losses else float('nan'),
            'bc_acc_legal': float(correct / max(1, total)),
        }

    def fit(
        self,
        dataset: Sequence[Tuple[np.ndarray, int, int, np.ndarray, np.ndarray]],
        *,
        epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """dataset entries: (obs, a1, a2, m1, m2)."""
        if not dataset:
            return {'bc_loss': float('nan'), 'bc_acc_legal': float('nan')}
        obs = np.stack([np.asarray(d[0], dtype=np.float32).reshape(OBS_DIM) for d in dataset])
        a1 = np.asarray([int(d[1]) for d in dataset], dtype=np.int64)
        a2 = np.asarray([int(d[2]) for d in dataset], dtype=np.int64)
        m1 = np.stack([np.asarray(d[3], dtype=np.float32).reshape(SLOT_ACTION_N) for d in dataset])
        m2 = np.stack([np.asarray(d[4], dtype=np.float32).reshape(SLOT_ACTION_N) for d in dataset])
        last: Dict[str, float] = {}
        for _ in range(int(epochs if epochs is not None else self.cfg.epochs)):
            last = self.train_epoch(obs, a1, a2, m1, m2)
        return last

    def save(self, path: Path | str) -> Path:
        self._ensure()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save({'net': self.net.state_dict(), 'bc': True}, path)
        return path

    def export_to_agent(self, agent) -> None:
        """Copy weights into a PPOAgent and set post-BC LR."""
        self._ensure()
        agent._ensure_torch()
        agent.net.load_state_dict(self.net.state_dict())
        agent.use_post_bc_lr()
