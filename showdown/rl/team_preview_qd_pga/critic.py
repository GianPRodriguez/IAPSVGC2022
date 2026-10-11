"""Shared preview critic V_φ (512D) — not stored per elite."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from showdown.rl.team_preview_obs import OBS_DIM


class SharedPreviewCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.v = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        return self.v(h).squeeze(-1)
