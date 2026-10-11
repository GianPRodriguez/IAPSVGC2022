"""Flat-360 actor–critic for Team Preview (spec §5)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from showdown.rl.constants import ILLEGAL_LOGIT
from showdown.rl.team_preview_action import N_ACTIONS
from showdown.rl.team_preview_obs import OBS_DIM


class PreviewActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.n_actions = int(n_actions)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.policy = nn.Linear(256, n_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        return self.policy(h), self.value(h).squeeze(-1)


def apply_mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(dtype=logits.dtype)
    return logits + (1.0 - m) * float(ILLEGAL_LOGIT)


def masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    z = apply_mask_logits(logits, mask)
    log_p = F.log_softmax(z, dim=-1)
    p = log_p.exp()
    m = (mask > 0.5).to(dtype=p.dtype)
    return -(p * log_p * m).sum(dim=-1)
