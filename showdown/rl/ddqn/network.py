"""Q-network: codificador estructurado → 512 → 512 → 1600 (DDQNSpec v1.10)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from showdown.rl.constants import JOINT_ACTION_N, OBS_DIM
from showdown.rl.encoders import combat_encoder


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = JOINT_ACTION_N,
                 encoder: Optional[str] = None):
        super().__init__()
        self.encoder = combat_encoder(obs_dim, encoder)
        self.net = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.encoder(obs))


@torch.no_grad()
def soft_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    """θ⁻ ← τθ + (1-τ)θ⁻."""
    t = float(tau)
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - t).add_(op.data, alpha=t)


def copy_weights(online: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(online.state_dict())
