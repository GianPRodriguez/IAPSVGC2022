"""Shared critic V_phi — 1-step TD(0) Huber (docs/qd_pga_map_elites_spec.md)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from showdown.rl.constants import OBS_DIM
from showdown.rl.qd_pga.replay import ValueReplay


class SharedCritic:
    """Global value network; updated off-policy from ``ValueReplay``."""

    def __init__(
        self,
        *,
        lr: float = 3e-4,
        gamma: float = 0.95,
        vf_coef: float = 0.5,
        device: str = 'cpu',
        obs_dim: int = OBS_DIM,
    ):
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.vf_coef = float(vf_coef)
        self.device_name = str(device)
        self.obs_dim = int(obs_dim)
        self._torch = None
        self.net = None
        self.optim = None
        self.update_steps = 0
        self._ensure_torch()

    def _ensure_torch(self) -> None:
        if self.net is not None:
            return
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        self._torch = torch
        self._F = F
        device = torch.device(self.device_name)
        self.device = device

        class _Critic(nn.Module):
            def __init__(self, obs_dim: int):
                super().__init__()
                self.fc1 = nn.Linear(obs_dim, 512)
                self.fc2 = nn.Linear(512, 512)
                self.value = nn.Linear(512, 1)

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                h = F.relu(self.fc1(obs))
                h = F.relu(self.fc2(h))
                return self.value(h).squeeze(-1)

        self.net = _Critic(self.obs_dim).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def value(self, obs: np.ndarray) -> float:
        self._ensure_torch()
        torch = self._torch
        self.net.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
            return float(self.net(x).item())

    def values_batch(self, obs: np.ndarray) -> np.ndarray:
        self._ensure_torch()
        torch = self._torch
        self.net.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if x.ndim == 1:
                x = x.view(1, -1)
            return self.net(x).detach().cpu().numpy().astype(np.float32)

    def td_update(
        self,
        replay: ValueReplay,
        *,
        n_updates: int = 8,
        batch_size: int = 256,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        """Run ``n_updates`` Huber TD(0) minibatches. No-op if replay too small."""
        self._ensure_torch()
        torch = self._torch
        F = self._F
        if len(replay) < int(batch_size):
            return {}
        rng = rng or np.random.default_rng()
        losses = []
        self.net.train()
        for _ in range(int(n_updates)):
            s, r, s2, term = replay.sample(batch_size, rng)
            s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
            r_t = torch.as_tensor(r, dtype=torch.float32, device=self.device)
            s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
            done = torch.as_tensor(term, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                v_next = self.net(s2_t)
                y = r_t + self.gamma * v_next * (1.0 - done)
            v = self.net(s_t)
            loss = self.vf_coef * F.huber_loss(v, y)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optim.step()
            losses.append(float(loss.item()))
            self.update_steps += 1
        return {'critic_loss': float(np.mean(losses)) if losses else float('nan')}
