"""Masked PPO agent for Team Preview flat-360."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from showdown.rl.team_preview_action import N_ACTIONS
from showdown.rl.team_preview_obs import OBS_DIM


@dataclass
class PreviewPPOConfig:
    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip_eps: float = 0.2
    lr: float = 3e-4
    rollout_episodes: int = 2048
    smoke_episodes: int = 64
    minibatch_size: int = 128
    epochs: int = 4
    vf_coef: float = 0.5
    ent_coef: float = 0.02
    grad_clip: float = 0.5
    device: str = 'cpu'
    seed: int = 0


class PreviewPPOAgent:
    def __init__(self, cfg: Optional[PreviewPPOConfig] = None):
        self.cfg = cfg or PreviewPPOConfig()
        self._torch = None
        self.net = None
        self.optim = None
        self.buffer = None
        self.env_episodes = 0
        self.update_steps = 0
        self._ensure_torch()

    def _ensure_torch(self) -> None:
        if self.net is not None:
            return
        try:
            import torch
            import torch.nn.functional as F
            from showdown.rl.team_preview_ppo.buffer import PreviewRolloutBuffer
            from showdown.rl.team_preview_ppo.network import (
                PreviewActorCritic,
                apply_mask_logits,
                masked_entropy,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'Team Preview PPO requires PyTorch. pip install -r requirements-rl.txt'
            ) from exc
        self._torch = torch
        self._F = F
        self._apply_mask = apply_mask_logits
        self._masked_ent = masked_entropy
        device = torch.device(self.cfg.device)
        self.device = device
        self.net = PreviewActorCritic(OBS_DIM, N_ACTIONS).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.buffer = PreviewRolloutBuffer(self.cfg.rollout_episodes, OBS_DIM)

    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        *,
        greedy: bool = False,
    ) -> Tuple[int, float, float]:
        self._ensure_torch()
        torch = self._torch
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            m = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.net(o)
            z = self._apply_mask(logits, m)
            if greedy:
                a = int(torch.argmax(z, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=z)
                a = int(dist.sample().item())
            logp = float(torch.distributions.Categorical(logits=z).log_prob(
                torch.tensor([a], device=self.device)
            ).item())
            v = float(value.item())
        return a, logp, v

    def update(self) -> dict:
        self._ensure_torch()
        torch = self._torch
        F = self._F
        self.buffer.finish()
        data = self.buffer.as_dict()
        n = len(self.buffer)
        if n == 0:
            return {}
        obs = torch.as_tensor(data['obs'], device=self.device)
        actions = torch.as_tensor(data['actions'], device=self.device)
        old_logp = torch.as_tensor(data['logps'], device=self.device)
        adv = torch.as_tensor(data['advantages'], device=self.device)
        ret = torch.as_tensor(data['returns'], device=self.device)
        masks = torch.as_tensor(data['masks'], device=self.device)

        idx = np.arange(n)
        metrics = {'pg_loss': 0.0, 'v_loss': 0.0, 'entropy': 0.0, 'n': n}
        batches = 0
        for _ in range(self.cfg.epochs):
            self._torch.manual_seed(self.cfg.seed + self.update_steps + _)
            np.random.default_rng(self.cfg.seed + _).shuffle(idx)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = idx[start: start + self.cfg.minibatch_size]
                logits, values = self.net(obs[mb])
                z = self._apply_mask(logits, masks[mb])
                dist = torch.distributions.Categorical(logits=z)
                logp = dist.log_prob(actions[mb])
                ratio = torch.exp(logp - old_logp[mb])
                a = adv[mb]
                surr1 = ratio * a
                surr2 = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps,
                ) * a
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(values, ret[mb])
                ent = self._masked_ent(logits, masks[mb]).mean()
                loss = pg_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * ent
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                self.optim.step()
                metrics['pg_loss'] += float(pg_loss.item())
                metrics['v_loss'] += float(v_loss.item())
                metrics['entropy'] += float(ent.item())
                batches += 1
        if batches:
            metrics['pg_loss'] /= batches
            metrics['v_loss'] /= batches
            metrics['entropy'] /= batches
        self.update_steps += 1
        self.buffer.clear()
        return metrics
