"""Low-rank payoff Value Net for Team Preview Nash (spec §3)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from showdown.rl.team_preview_action import N_ACTIONS
from showdown.rl.team_preview_obs import OBS_DIM

D_ACTION = 32
D_LAT = 64
TRUNK_H = 256


class LowRankPayoffNet(nn.Module):
    """
    Shared MLP trunk on 512D obs → Φ_row(h, e_i), Φ_col(h, e_j) ∈ R^{64}.
    M̂_ij = σ(⟨Φ_row, Φ_col⟩).
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        d_a: int = D_ACTION,
        d_lat: int = D_LAT,
    ):
        super().__init__()
        self.n_actions = int(n_actions)
        self.d_a = int(d_a)
        self.d_lat = int(d_lat)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, TRUNK_H),
            nn.ReLU(),
            nn.Linear(TRUNK_H, TRUNK_H),
            nn.ReLU(),
        )
        self.embed_row = nn.Embedding(n_actions, d_a)
        self.embed_col = nn.Embedding(n_actions, d_a)
        self.row_head = nn.Linear(TRUNK_H + d_a, d_lat)
        self.col_head = nn.Linear(TRUNK_H + d_a, d_lat)

    def encode_h(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def phi_row(self, h: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        e = self.embed_row(actions)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if e.dim() == 1:
            e = e.unsqueeze(0)
        # broadcast h over action batch if needed
        if h.shape[0] == 1 and e.shape[0] > 1:
            h = h.expand(e.shape[0], -1)
        return self.row_head(torch.cat([h, e], dim=-1))

    def phi_col(self, h: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        e = self.embed_col(actions)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if e.dim() == 1:
            e = e.unsqueeze(0)
        if h.shape[0] == 1 and e.shape[0] > 1:
            h = h.expand(e.shape[0], -1)
        return self.col_head(torch.cat([h, e], dim=-1))

    def pairwise_logit(
        self, obs: torch.Tensor, i: torch.Tensor, j: torch.Tensor,
    ) -> torch.Tensor:
        """Logits before sigmoid for batch of (obs, i, j)."""
        h = self.encode_h(obs)
        pr = self.phi_row(h, i)
        pc = self.phi_col(h, j)
        return (pr * pc).sum(dim=-1)

    def pairwise_wr(
        self, obs: torch.Tensor, i: torch.Tensor, j: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.pairwise_logit(obs, i, j))

    def payoff_matrix(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Full M̂ ∈ [0,1]^{360×360} for a single obs (or first of batch).
        One matmul after 360+360 embeddings.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        h = self.encode_h(obs[:1])  # (1, H)
        idx = torch.arange(self.n_actions, device=obs.device)
        # row: (360, d_lat), col: (360, d_lat)
        pr = self.phi_row(h, idx)
        pc = self.phi_col(h, idx)
        logits = pr @ pc.T  # (360, 360)
        return torch.sigmoid(logits)
