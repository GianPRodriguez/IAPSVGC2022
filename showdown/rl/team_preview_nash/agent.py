"""Team Preview Nash agent: Value Net + Nash LP (spec v1.0)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from showdown.rl.team_preview_action import N_ACTIONS, action_to_team_string
from showdown.rl.team_preview_nash.nash_solver import (
    entropy,
    solve_mixed_strategy,
    support_size,
)
from showdown.rl.team_preview_obs import OBS_DIM


@dataclass
class PreviewNashConfig:
    lr: float = 3e-4
    rollout_pairs: int = 2048
    smoke_pairs: int = 64
    minibatch_size: int = 128
    epochs: int = 4
    grad_clip: float = 0.5
    solver_timeout_s: float = 0.2
    device: str = 'cpu'
    seed: int = 0
    loss: str = 'bce'  # 'bce' | 'mse'


class PreviewNashAgent:
    def __init__(self, cfg: Optional[PreviewNashConfig] = None):
        self.cfg = cfg or PreviewNashConfig()
        self._torch = None
        self.net = None
        self.optim = None
        self.env_pairs = 0
        self.update_steps = 0
        self._rng = np.random.default_rng(self.cfg.seed)
        self._ensure_torch()

    def _ensure_torch(self) -> None:
        if self.net is not None:
            return
        try:
            import torch
            from showdown.rl.team_preview_nash.network import LowRankPayoffNet
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'Team Preview Nash requires PyTorch. pip install -r requirements-rl.txt'
            ) from exc
        self._torch = torch
        device = torch.device(self.cfg.device)
        self.device = device
        self.net = LowRankPayoffNet(OBS_DIM, N_ACTIONS).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

    def payoff_and_nash(
        self, obs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
        """Returns (M_hat, p*, q*, v*, method)."""
        self._ensure_torch()
        torch = self._torch
        self.net.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            M = self.net.payoff_matrix(o).detach().cpu().numpy().astype(np.float64)
        A = 2.0 * M - 1.0
        p, q, v, method = solve_mixed_strategy(
            A, timeout_s=self.cfg.solver_timeout_s,
        )
        return M, p, q, v, method

    def act(
        self,
        obs: np.ndarray,
        *,
        greedy: bool = False,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, Dict]:
        """
        greedy=False → sample from p* (in-game).
        greedy=True → argmax p* (eval / thesis tables).
        """
        _M, p, q, v, method = self.payoff_and_nash(obs)
        if mask is not None:
            m = np.asarray(mask, dtype=np.float64)
            p = p * (m > 0.5)
            s = p.sum()
            if s <= 0:
                p = np.where(m > 0.5, 1.0, 0.0)
                p = p / p.sum()
            else:
                p = p / s
        if greedy:
            a = int(np.argmax(p))
        else:
            a = int(self._rng.choice(N_ACTIONS, p=p))
        info = {
            'v_star': float(v),
            'entropy_p': entropy(p),
            'support': support_size(p),
            'solver': method,
            'team': action_to_team_string(a),
            'preview_action': a,
            'q_star_max': float(np.max(q)),
        }
        return a, info

    def update_from_batch(
        self,
        obs: np.ndarray,
        i: np.ndarray,
        j: np.ndarray,
        r: np.ndarray,
    ) -> Dict[str, float]:
        """One or more epochs of BCE/MSE on (obs, i, j, r)."""
        self._ensure_torch()
        torch = self._torch
        self.net.train()
        n = int(obs.shape[0])
        B = min(int(self.cfg.minibatch_size), max(n, 1))
        idx_all = np.arange(n)
        total_loss = 0.0
        steps = 0
        for _ in range(int(self.cfg.epochs)):
            self._rng.shuffle(idx_all)
            for start in range(0, n, B):
                batch = idx_all[start: start + B]
                o = torch.as_tensor(obs[batch], dtype=torch.float32, device=self.device)
                ii = torch.as_tensor(i[batch], dtype=torch.long, device=self.device)
                jj = torch.as_tensor(j[batch], dtype=torch.long, device=self.device)
                rr = torch.as_tensor(r[batch], dtype=torch.float32, device=self.device)
                pred = self.net.pairwise_wr(o, ii, jj)
                if self.cfg.loss == 'mse':
                    loss = torch.mean((pred - rr) ** 2)
                else:
                    # soft BCE for r in {0, 0.5, 1}
                    eps = 1e-6
                    pred_c = pred.clamp(eps, 1 - eps)
                    loss = torch.mean(
                        -(rr * torch.log(pred_c) + (1 - rr) * torch.log(1 - pred_c))
                    )
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip,
                )
                self.optim.step()
                total_loss += float(loss.item())
                steps += 1
        self.update_steps += 1
        return {'loss': total_loss / max(steps, 1)}
