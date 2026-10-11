"""Episode buffer for 1-step Team Preview PPO."""
from __future__ import annotations

from typing import Dict

import numpy as np

from showdown.rl.team_preview_action import N_ACTIONS
from showdown.rl.team_preview_obs import OBS_DIM


class PreviewRolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int = OBS_DIM):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.ptr = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.logps = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.masks = np.zeros((capacity, N_ACTIONS), dtype=np.float32)
        self.advantages = np.zeros((capacity,), dtype=np.float32)
        self.returns = np.zeros((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self.ptr

    def clear(self) -> None:
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        logp: float,
        value: float,
        mask: np.ndarray,
    ) -> None:
        if self.ptr >= self.capacity:
            raise RuntimeError('buffer full')
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.logps[i] = float(logp)
        self.values[i] = float(value)
        self.masks[i] = mask
        self.ptr += 1

    def finish(self) -> None:
        """1-step: advantage = R - V; return = R (γ=1)."""
        n = self.ptr
        r = self.rewards[:n]
        v = self.values[:n]
        adv = r - v
        # normalize
        if n > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.advantages[:n] = adv
        self.returns[:n] = r

    def as_dict(self) -> Dict[str, np.ndarray]:
        n = self.ptr
        return {
            'obs': self.obs[:n],
            'actions': self.actions[:n],
            'logps': self.logps[:n],
            'advantages': self.advantages[:n],
            'returns': self.returns[:n],
            'masks': self.masks[:n],
        }
