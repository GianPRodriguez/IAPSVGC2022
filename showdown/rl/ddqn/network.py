"""MLP Q-network 1024→512→512→576 (DDQNSpec)."""
from __future__ import annotations

import torch
import torch.nn as nn

from showdown.rl.constants import JOINT_ACTION_N, OBS_DIM


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = JOINT_ACTION_N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


@torch.no_grad()
def soft_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    """θ⁻ ← τθ + (1-τ)θ⁻."""
    t = float(tau)
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - t).add_(op.data, alpha=t)


def copy_weights(online: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(online.state_dict())
