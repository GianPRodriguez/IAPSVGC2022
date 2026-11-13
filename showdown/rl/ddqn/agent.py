"""DDQN agent: masked ε-greedy + Double update (docs/ddqn_spec.md)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from showdown.rl.action_space import (
    apply_q_mask,
    decode_joint_action,
    encode_joint_action,
)
from showdown.rl.constants import JOINT_ACTION_N, OBS_DIM, PASS_ID


@dataclass
class DDQNConfig:
    gamma: float = 0.95
    lr: float = 3e-4
    tau: float = 0.005
    batch_size: int = 256
    replay_capacity: int = 200_000
    warmup_steps: int = 5_000
    train_freq: int = 4
    huber_delta: float = 1.0
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 500_000
    device: str = 'cpu'
    seed: int = 0


class DDQNAgent:
    def __init__(self, cfg: Optional[DDQNConfig] = None):
        self.cfg = cfg or DDQNConfig()
        self._torch = None
        self.online = None
        self.target = None
        self.optim = None
        self.replay = None
        self._rng = np.random.default_rng(int(self.cfg.seed))
        self.env_steps = 0
        self.grad_steps = 0
        self._ensure_torch()

    def _ensure_torch(self) -> None:
        if self.online is not None:
            return
        try:
            import torch
            import torch.nn.functional as F
            from showdown.rl.ddqn.network import QNetwork, copy_weights, soft_update
            from showdown.rl.ddqn.replay import ReplayBuffer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'DDQN requires PyTorch. Install: pip install -r requirements-rl.txt'
            ) from exc
        self._torch = torch
        self._F = F
        self._soft_update = soft_update
        device = torch.device(self.cfg.device)
        self.device = device
        self.online = QNetwork().to(device)
        self.target = QNetwork().to(device)
        copy_weights(self.online, self.target)
        self.target.eval()
        self.optim = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.replay = ReplayBuffer(capacity=self.cfg.replay_capacity, obs_dim=OBS_DIM)

    @property
    def epsilon(self) -> float:
        t = min(1.0, float(self.env_steps) / max(1, self.cfg.eps_decay_steps))
        return float(self.cfg.eps_start + t * (self.cfg.eps_end - self.cfg.eps_start))

    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        *,
        epsilon: Optional[float] = None,
    ) -> int:
        """Return joint action id. epsilon=None → schedule; 0 → greedy."""
        self._ensure_torch()
        m = np.asarray(mask, dtype=np.float32).reshape(JOINT_ACTION_N)
        legal = np.flatnonzero(m > 0.5)
        if legal.size == 0:
            return encode_joint_action(PASS_ID, PASS_ID)
        eps = self.epsilon if epsilon is None else float(epsilon)
        if self.env_steps < self.cfg.warmup_steps or self._rng.random() < eps:
            return int(self._rng.choice(legal))
        return self.greedy(obs, m)

    def act_batch(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        *,
        epsilon: Optional[float] = None,
    ) -> np.ndarray:
        """ε-greedy en lote para N entornos (una sola pasada de la red).

        Misma política que :meth:`act` aplicada fila a fila: warmup/ε → acción
        legal aleatoria; si no, argmax de Q enmascarado.
        """
        self._ensure_torch()
        torch = self._torch
        obs = np.asarray(obs, dtype=np.float32).reshape(-1, OBS_DIM)
        masks = np.asarray(masks, dtype=np.float32).reshape(-1, JOINT_ACTION_N)
        n = obs.shape[0]
        eps = self.epsilon if epsilon is None else float(epsilon)
        explore = (self.env_steps < self.cfg.warmup_steps) or (eps >= 1.0)
        rand = self._rng.random(n) < eps
        need_q = (not explore) and bool((~rand).any())
        q = None
        if need_q:
            self.online.eval()
            with torch.no_grad():
                x = torch.as_tensor(obs, device=self.device)
                q = self.online(x).float().cpu().numpy()
        out = np.zeros(n, dtype=np.int64)
        for i in range(n):
            legal = np.flatnonzero(masks[i] > 0.5)
            if legal.size == 0:
                out[i] = encode_joint_action(PASS_ID, PASS_ID)
            elif explore or rand[i] or q is None:
                out[i] = int(self._rng.choice(legal))
            else:
                out[i] = int(np.argmax(apply_q_mask(q[i], masks[i])))
        return out

    def greedy(self, obs: np.ndarray, mask: np.ndarray) -> int:
        self._ensure_torch()
        torch = self._torch
        self.online.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
            q = self.online(x).cpu().numpy().reshape(-1)
        q_m = apply_q_mask(q, mask)
        return int(np.argmax(q_m))

    def observe(
        self,
        *,
        s: np.ndarray,
        a: int,
        r: float,
        s_next: np.ndarray,
        terminated: bool,
        m_next: np.ndarray,
        skip_buffer: bool = False,
        truncated: bool = False,
        opponent_code: int = 0,
    ) -> Optional[Dict[str, float]]:
        """Store transition e_t=(s,a,r,s',done,m'); maybe learn."""
        self._ensure_torch()
        self.env_steps += 1
        self.replay.add(
            s=s, a=a, r=r, s_next=s_next, terminated=terminated, m_next=m_next,
            skip_buffer=skip_buffer, truncated=truncated,
            opponent_code=int(opponent_code),
        )
        if self.env_steps < self.cfg.warmup_steps:
            return None
        if self.env_steps % self.cfg.train_freq != 0:
            return None
        if len(self.replay) < self.cfg.batch_size:
            return None
        return self.learn()

    def learn(self) -> Dict[str, float]:
        self._ensure_torch()
        torch = self._torch
        F = self._F
        from showdown.rl.ddqn.replay import ReplayBuffer

        s, a, r, s2, term, m2, opp = self.replay.sample(self.cfg.batch_size, self._rng)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(a, dtype=torch.int64, device=self.device)
        r_t = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(term, dtype=torch.float32, device=self.device)
        m2_t = torch.as_tensor(m2, dtype=torch.float32, device=self.device)

        self.online.train()
        q_all = self.online(s_t)
        q_sa = q_all.gather(1, a_t.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            q_online_next = self.online(s2_t)
            # Mask illegal before argmax
            q_online_next = q_online_next + (1.0 - m2_t) * (-1.0e9)
            a_star = q_online_next.argmax(dim=1)
            q_target_next = self.target(s2_t).gather(1, a_star.view(-1, 1)).squeeze(1)
            y = r_t + (1.0 - done) * self.cfg.gamma * q_target_next

        loss = F.huber_loss(q_sa, y, delta=self.cfg.huber_delta)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optim.step()
        self._soft_update(self.online, self.target, self.cfg.tau)
        self.grad_steps += 1
        batch_frac = ReplayBuffer.batch_origin_fractions(opp)
        buf_frac = self.replay.origin_fractions()
        out: Dict[str, float] = {
            'loss': float(loss.item()),
            'mean_q': float(q_sa.mean().item()),
            'mean_target': float(y.mean().item()),
            'td_error': float((q_sa - y).abs().mean().item()),
            'eps': self.epsilon,
        }
        for name, frac in batch_frac.items():
            out[f'batch_frac_{name}'] = float(frac)
        for name, frac in buf_frac.items():
            out[f'buf_frac_{name}'] = float(frac)
        return out

    def save(self, path: Path | str) -> Path:
        self._ensure_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'config': asdict(self.cfg),
            'online': self.online.state_dict(),
            'target': self.target.state_dict(),
            'env_steps': self.env_steps,
            'grad_steps': self.grad_steps,
        }
        self._torch.save(payload, path)
        return path

    def load(self, path: Path | str, *, map_location: Optional[str] = None) -> None:
        self._ensure_torch()
        torch = self._torch
        loc = map_location or self.cfg.device
        try:
            payload = torch.load(path, map_location=loc, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=loc)
        if 'config' in payload and isinstance(payload['config'], dict):
            for k, v in payload['config'].items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)
        self.online.load_state_dict(payload['online'])
        self.target.load_state_dict(payload.get('target', payload['online']))
        self.env_steps = int(payload.get('env_steps', 0))
        self.grad_steps = int(payload.get('grad_steps', 0))

    @staticmethod
    def decode(joint_id: int) -> Tuple[int, int]:
        return decode_joint_action(joint_id)

    @staticmethod
    def encode(a1: int, a2: int) -> int:
        return encode_joint_action(a1, a2)
