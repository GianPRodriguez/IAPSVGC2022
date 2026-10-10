"""Actor–critic MLP — auto-regressive Slot1→Slot2 (docs/ppo_spec.md)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from showdown.rl.constants import ILLEGAL_LOGIT, OBS_DIM, SLOT_ACTION_N


class ActorCritic(nn.Module):
    """
    Shared torso 1024→512→512, Slot1 head 24, Slot2 head on [h; onehot(a1)], V(s).
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = SLOT_ACTION_N):
        super().__init__()
        self.n_actions = int(n_actions)
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.policy1 = nn.Linear(512, n_actions)
        self.policy2 = nn.Linear(512 + n_actions, n_actions)
        self.value = nn.Linear(512, 1)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(obs))
        return F.relu(self.fc2(h))

    def forward(
        self,
        obs: torch.Tensor,
        a1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        h = self.features(obs)
        z1 = self.policy1(h)
        v = self.value(h).squeeze(-1)
        z2 = None
        if a1 is not None:
            oh = F.one_hot(a1.long(), num_classes=self.n_actions).float()
            z2 = self.policy2(torch.cat([h, oh], dim=-1))
        return z1, z2, v

    def slot2_logits(self, h: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
        oh = F.one_hot(a1.long(), num_classes=self.n_actions).float()
        return self.policy2(torch.cat([h, oh], dim=-1))


def apply_mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """z + (1-M)*(-1e9); mask 1=legal."""
    m = mask.to(dtype=logits.dtype)
    return logits + (1.0 - m) * float(ILLEGAL_LOGIT)


def masked_categorical_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax over legal actions only (batch,)."""
    z = apply_mask_logits(logits, mask)
    log_p = F.log_softmax(z, dim=-1)
    p = log_p.exp()
    # Zero contribution from illegal (numerically ~0 after mask)
    m = (mask > 0.5).to(dtype=p.dtype)
    ent = -(p * log_p * m).sum(dim=-1)
    return ent
