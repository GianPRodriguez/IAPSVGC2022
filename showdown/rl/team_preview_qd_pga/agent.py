"""Deploy / act with Team Preview QD-PGA π*."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from showdown.rl.team_preview_action import action_to_team_string
from showdown.rl.team_preview_obs import OBS_DIM
from showdown.rl.team_preview_ppo.network import PreviewActorCritic, apply_mask_logits
from showdown.rl.team_preview_qd_pga.archive import load_actor_into


@dataclass
class PreviewQDPGAConfig:
    device: str = 'cpu'
    seed: int = 0


class PreviewQDPGAAgent:
    def __init__(self, cfg: Optional[PreviewQDPGAConfig] = None):
        self.cfg = cfg or PreviewQDPGAConfig()
        self._torch = None
        self.net = None
        self._rng = np.random.default_rng(self.cfg.seed)
        self._ensure()

    def _ensure(self) -> None:
        if self.net is not None:
            return
        import torch
        self._torch = torch
        self.device = torch.device(self.cfg.device)
        self.net = PreviewActorCritic(OBS_DIM).to(self.device)

    def load_actor(self, actor_sd) -> None:
        self._ensure()
        load_actor_into(self.net, actor_sd)

    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        *,
        greedy: bool = False,
    ) -> Tuple[int, dict]:
        self._ensure()
        torch = self._torch
        self.net.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            m = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.net(o)
            z = apply_mask_logits(logits, m)
            if greedy:
                a = int(torch.argmax(z, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=z)
                a = int(dist.sample().item())
        return a, {
            'team': action_to_team_string(a),
            'preview_action': a,
            'value': float(value.item()),
        }
