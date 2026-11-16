"""Masked PPO agent for Team Preview flat-360."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

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
    # 90 formaciones canónicas en vez de 360 órdenes (team_preview_action.canonical_mask).
    canonical: bool = False


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
        self._apply_mask_raw = apply_mask_logits
        self._canon = None
        self._masked_ent = masked_entropy
        device = torch.device(self.cfg.device)
        self.device = device
        self.net = PreviewActorCritic(OBS_DIM, N_ACTIONS).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.buffer = PreviewRolloutBuffer(self.cfg.rollout_episodes, OBS_DIM)

    def _masked_logits(self, logits, m):
        """Máscara legal (y canónica si ``cfg.canonical``; se lee en cada llamada, así que un
        checkpoint cargado con ``canonical=True`` también la aplica en inferencia)."""
        if bool(self.cfg.canonical):
            if self._canon is None or self._canon.device != m.device:
                from showdown.rl.team_preview_action import canonical_mask
                self._canon = self._torch.as_tensor(canonical_mask(), dtype=m.dtype, device=m.device)
            m = m * self._canon
        return self._apply_mask_raw(logits, m)

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
            z = self._masked_logits(logits, m)
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
        metrics = {'pg_loss': 0.0, 'v_loss': 0.0, 'entropy': 0.0, 'n': n,
                   'approx_kl': 0.0, 'clip_frac': 0.0}
        # Diagnóstico pre-update (tesis): ¿V explica el resultado? ¿la política se concentra?
        rewards = np.asarray(self.buffer.rewards[:n], dtype=np.float64)
        values_old = np.asarray(self.buffer.values[:n], dtype=np.float64)
        var_r = float(rewards.var())
        metrics['explained_var'] = float(1.0 - (rewards - values_old).var() / var_r) if var_r > 1e-12 else 0.0
        metrics['mean_reward'] = float(rewards.mean())
        metrics['mean_value'] = float(values_old.mean())
        with torch.no_grad():
            lg, _v = self.net(obs)
            p = torch.softmax(self._masked_logits(lg, masks), dim=-1)
            metrics['max_prob'] = float(p.max(dim=-1).values.mean().item())
            h = self._masked_ent(lg, masks)
            metrics['effective_actions'] = float(torch.exp(h).mean().item())
        batches = 0
        # Permutación distinta en cada update y época (antes: misma secuencia en
        # todos los updates porque la semilla no incluía update_steps).
        shuffle_rng = np.random.default_rng([int(self.cfg.seed), int(self.update_steps)])
        for _ in range(self.cfg.epochs):
            self._torch.manual_seed(self.cfg.seed + self.update_steps + _)
            shuffle_rng.shuffle(idx)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = idx[start: start + self.cfg.minibatch_size]
                logits, values = self.net(obs[mb])
                z = self._masked_logits(logits, masks[mb])
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
                with torch.no_grad():
                    log_ratio = logp - old_logp[mb]
                    metrics['approx_kl'] += float(((ratio - 1.0) - log_ratio).mean().item())
                    metrics['clip_frac'] += float(
                        ((ratio - 1.0).abs() > self.cfg.clip_eps).float().mean().item()
                    )
                batches += 1
        if batches:
            for k in ('pg_loss', 'v_loss', 'entropy', 'approx_kl', 'clip_frac'):
                metrics[k] /= batches
        self.update_steps += 1
        self.buffer.clear()
        return metrics

    def save(self, path: Union[str, Path]) -> Path:
        self._ensure_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'kind': 'team_preview_ppo',
            'cfg': asdict(self.cfg),
            'state_dict': self.net.state_dict(),
            'env_episodes': int(self.env_episodes),
            'update_steps': int(self.update_steps),
        }
        self._torch.save(payload, path)
        return path

    def load(self, path: Union[str, Path]) -> None:
        self._ensure_torch()
        path = Path(path)
        try:
            payload = self._torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            payload = self._torch.load(path, map_location='cpu')
        if isinstance(payload, dict) and 'state_dict' in payload:
            sd = payload['state_dict']
            cfg = payload.get('cfg') or {}
            if isinstance(cfg, dict):
                for k, v in cfg.items():
                    if hasattr(self.cfg, k):
                        setattr(self.cfg, k, v)
            self.env_episodes = int(payload.get('env_episodes', 0))
            self.update_steps = int(payload.get('update_steps', 0))
        else:
            sd = payload
        self.net.load_state_dict(sd)
        self.net.to(self.device)
