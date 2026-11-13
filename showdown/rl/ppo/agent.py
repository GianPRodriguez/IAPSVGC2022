"""Masked PPO agent — AR policy + GAE (docs/ppo_spec.md)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from showdown.rl.constants import OBS_DIM, PASS_ID, SLOT_ACTION_N


@dataclass
class PPOConfig:
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    lr_post_bc: float = 1e-4
    rollout_steps: int = 2048
    num_envs: int = 8
    minibatch_size: int = 256
    epochs: int = 4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    grad_clip: float = 0.5
    device: str = 'cpu'
    seed: int = 0


class PPOAgent:
    def __init__(self, cfg: Optional[PPOConfig] = None):
        self.cfg = cfg or PPOConfig()
        self._torch = None
        self.net = None
        self.optim = None
        self.buffer = None
        self._rng = np.random.default_rng(int(self.cfg.seed))
        self.env_steps = 0
        self.update_steps = 0
        self._ensure_torch()

    def _ensure_torch(self) -> None:
        if self.net is not None:
            return
        try:
            import torch
            import torch.nn.functional as F
            from showdown.rl.ppo.buffer import RolloutBuffer
            from showdown.rl.ppo.network import (
                ActorCritic,
                apply_mask_logits,
                masked_categorical_entropy,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'PPO requires PyTorch. Install: pip install -r requirements-rl.txt'
            ) from exc
        self._torch = torch
        self._F = F
        self._apply_mask = apply_mask_logits
        self._masked_ent = masked_categorical_entropy
        device = torch.device(self.cfg.device)
        self.device = device
        self.net = ActorCritic().to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.buffer = RolloutBuffer(capacity=self.cfg.rollout_steps, obs_dim=OBS_DIM)

    def set_lr(self, lr: float) -> None:
        self._ensure_torch()
        for pg in self.optim.param_groups:
            pg['lr'] = float(lr)

    def use_post_bc_lr(self) -> None:
        self.set_lr(self.cfg.lr_post_bc)

    def _slot_masks(
        self,
        env,
        *,
        slot1_action_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from showdown.rl.action_space import build_autoregressive_masks

        live = env.backend.action_masks(slot1_action_id=slot1_action_id)
        if live is not None:
            if slot1_action_id is None:
                return (
                    np.asarray(live[0], dtype=np.float32),
                    np.asarray(live[1], dtype=np.float32),
                )
            return (
                np.asarray(live[0], dtype=np.float32),
                np.asarray(live[1], dtype=np.float32),
            )
        assert env._snap is not None
        return build_autoregressive_masks(env._snap, slot1_action_id=slot1_action_id)

    def act(
        self,
        obs: np.ndarray,
        m1: np.ndarray,
        m2_fn,
        *,
        greedy: bool = False,
    ) -> Tuple[int, int, float, float]:
        """
        Sample (or greedy) (a1, a2).

        ``m2_fn(a1) -> mask24`` for conditioned Slot2 mask.
        Returns a1, a2, logπ, V(s).
        """
        self._ensure_torch()
        torch = self._torch
        self.net.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
            m1_t = torch.as_tensor(m1, dtype=torch.float32, device=self.device).view(1, -1)
            z1, _, v = self.net(x)
            z1m = self._apply_mask(z1, m1_t)
            if greedy:
                a1 = int(z1m.argmax(dim=-1).item())
            else:
                dist1 = torch.distributions.Categorical(logits=z1m)
                a1 = int(dist1.sample().item())
            # Ensure legal
            legal1 = np.flatnonzero(np.asarray(m1).reshape(-1) > 0.5)
            if legal1.size == 0:
                a1 = PASS_ID
            elif np.asarray(m1).reshape(-1)[a1] <= 0.5:
                a1 = int(legal1[0])

            m2 = np.asarray(m2_fn(a1), dtype=np.float32).reshape(SLOT_ACTION_N)
            m2_t = torch.as_tensor(m2, dtype=torch.float32, device=self.device).view(1, -1)
            a1_t = torch.tensor([a1], dtype=torch.long, device=self.device)
            h = self.net.features(x)
            z2 = self.net.slot2_logits(h, a1_t)
            z2m = self._apply_mask(z2, m2_t)
            if greedy:
                a2 = int(z2m.argmax(dim=-1).item())
            else:
                dist2 = torch.distributions.Categorical(logits=z2m)
                a2 = int(dist2.sample().item())
            legal2 = np.flatnonzero(m2 > 0.5)
            if legal2.size == 0:
                a2 = PASS_ID
            elif m2[a2] <= 0.5:
                a2 = int(legal2[0])

            # log-prob under masked categoricals (recompute for chosen actions)
            logp1 = torch.distributions.Categorical(logits=z1m).log_prob(
                torch.tensor([a1], device=self.device)
            )
            a2_t = torch.tensor([a2], dtype=torch.long, device=self.device)
            z2 = self.net.slot2_logits(h, torch.tensor([a1], device=self.device))
            z2m = self._apply_mask(z2, m2_t)
            logp2 = torch.distributions.Categorical(logits=z2m).log_prob(a2_t)
            logp = float((logp1 + logp2).item())
            value = float(v.item())
        return a1, a2, logp, value

    def evaluate_actions(
        self,
        obs: 'torch.Tensor',
        a1: 'torch.Tensor',
        a2: 'torch.Tensor',
        m1: 'torch.Tensor',
        m2: 'torch.Tensor',
    ) -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor', 'torch.Tensor']:
        """Return logπ, entropy, values, (unused)."""
        self._ensure_torch()
        torch = self._torch
        z1, z2, v = self.net(obs, a1=a1)
        assert z2 is not None
        z1m = self._apply_mask(z1, m1)
        z2m = self._apply_mask(z2, m2)
        dist1 = torch.distributions.Categorical(logits=z1m)
        dist2 = torch.distributions.Categorical(logits=z2m)
        logp = dist1.log_prob(a1) + dist2.log_prob(a2)
        ent = self._masked_ent(z1, m1) + self._masked_ent(z2, m2)
        return logp, ent, v, z1m

    def observe(
        self,
        *,
        obs: np.ndarray,
        a1: int,
        a2: int,
        logp: float,
        reward: float,
        value: float,
        terminated: bool,
        m1: np.ndarray,
        m2: np.ndarray,
        skip_buffer: bool = False,
        trunc_reason: Optional[str] = None,
        bootstrap_value: float = 0.0,
        env_index: Optional[int] = None,
    ) -> bool:
        self._ensure_torch()
        self.env_steps += 1
        kwargs = dict(
            obs=obs, a1=a1, a2=a2, logp=logp, reward=reward, value=value,
            terminated=terminated, m1=m1, m2=m2,
            skip_buffer=skip_buffer, trunc_reason=trunc_reason,
            bootstrap_value=bootstrap_value,
        )
        if env_index is not None and hasattr(self.buffer, 'buffers'):
            return self.buffer.add(env_index=int(env_index), **kwargs)
        return self.buffer.add(**kwargs)

    def is_rollout_full(self) -> bool:
        if len(self.buffer) >= self.cfg.rollout_steps:
            return True
        any_full = getattr(self.buffer, 'any_full', None)
        return bool(any_full()) if callable(any_full) else False

    # --- entrenamiento vectorizado ----------------------------------------

    def use_vec_buffer(self, num_envs: int) -> None:
        """Un sub-buffer por entorno (GAE por entorno). ``rollout_steps`` es el total."""
        from showdown.rl.ppo.buffer import VecRolloutBuffer

        self._ensure_torch()
        n = max(1, int(num_envs))
        self.cfg.num_envs = n
        per_env = max(1, -(-int(self.cfg.rollout_steps) // n))  # ceil
        self.buffer = VecRolloutBuffer(n, capacity=per_env, obs_dim=OBS_DIM)

    def values_batch(self, obs: np.ndarray) -> np.ndarray:
        """V(s) para un lote ``[N, OBS_DIM]``."""
        self._ensure_torch()
        torch = self._torch
        obs = np.asarray(obs, dtype=np.float32).reshape(-1, OBS_DIM)
        if obs.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        self.net.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            _z1, _z2, v = self.net(x)
        return v.detach().float().cpu().numpy().reshape(-1)

    def act_batch(
        self,
        obs: np.ndarray,
        m1: np.ndarray,
        mask2_fn,
        *,
        greedy: bool = False,
    ):
        """Versión en lote de :meth:`act` para N entornos.

        ``mask2_fn(a1s) -> [N, 24]`` devuelve la máscara Slot2 condicionada a
        cada a1 (un viaje de ida y vuelta a los workers por paso).
        Devuelve ``(a1, a2, logp, value, m2)`` como arrays de longitud N.
        """
        self._ensure_torch()
        torch = self._torch
        obs = np.asarray(obs, dtype=np.float32).reshape(-1, OBS_DIM)
        m1 = np.asarray(m1, dtype=np.float32).reshape(-1, SLOT_ACTION_N)
        n = obs.shape[0]
        self.net.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, device=self.device)
            m1_t = torch.as_tensor(m1, device=self.device)
            h = self.net.features(x)
            z1, _, v = self.net(x)
            z1m = self._apply_mask(z1, m1_t)
            if greedy:
                a1_t = z1m.argmax(dim=-1)
            else:
                a1_t = torch.distributions.Categorical(logits=z1m).sample()
            a1 = a1_t.cpu().numpy().astype(np.int64)
            # Legalidad (filas sin acciones legales -> PASS), igual que act().
            for i in range(n):
                legal = np.flatnonzero(m1[i] > 0.5)
                if legal.size == 0:
                    a1[i] = PASS_ID
                elif m1[i, a1[i]] <= 0.5:
                    a1[i] = int(legal[0])
            m2 = np.asarray(mask2_fn(a1), dtype=np.float32).reshape(n, SLOT_ACTION_N)
            a1_t = torch.as_tensor(a1, dtype=torch.long, device=self.device)
            m2_t = torch.as_tensor(m2, device=self.device)
            z2 = self.net.slot2_logits(h, a1_t)
            z2m = self._apply_mask(z2, m2_t)
            if greedy:
                a2_t = z2m.argmax(dim=-1)
            else:
                a2_t = torch.distributions.Categorical(logits=z2m).sample()
            a2 = a2_t.cpu().numpy().astype(np.int64)
            for i in range(n):
                legal = np.flatnonzero(m2[i] > 0.5)
                if legal.size == 0:
                    a2[i] = PASS_ID
                elif m2[i, a2[i]] <= 0.5:
                    a2[i] = int(legal[0])
            a2_t = torch.as_tensor(a2, dtype=torch.long, device=self.device)
            logp1 = torch.distributions.Categorical(logits=z1m).log_prob(a1_t)
            logp2 = torch.distributions.Categorical(logits=z2m).log_prob(a2_t)
            logp = (logp1 + logp2).float().cpu().numpy().reshape(-1)
            value = v.float().cpu().numpy().reshape(-1)
        return a1, a2, logp, value, m2

    def update(self, *, last_value: float = 0.0) -> Dict[str, float]:
        """GAE + K epochs of clipped PPO. Clears buffer afterward."""
        self._ensure_torch()
        torch = self._torch
        F = self._F
        if len(self.buffer) == 0:
            return {}
        self.buffer.compute_gae(
            last_value=last_value,
            gamma=self.cfg.gamma,
            lam=self.cfg.gae_lambda,
            normalize_adv=True,
        )
        clip_fracs = []
        approx_kls = []
        entropies = []
        losses = []
        self.net.train()
        for _ in range(int(self.cfg.epochs)):
            for batch in self.buffer.minibatches(self.cfg.minibatch_size, self._rng):
                obs = torch.as_tensor(batch['obs'], dtype=torch.float32, device=self.device)
                a1 = torch.as_tensor(batch['a1'], dtype=torch.long, device=self.device)
                a2 = torch.as_tensor(batch['a2'], dtype=torch.long, device=self.device)
                old_logp = torch.as_tensor(batch['logp'], dtype=torch.float32, device=self.device)
                adv = torch.as_tensor(batch['adv'], dtype=torch.float32, device=self.device)
                ret = torch.as_tensor(batch['ret'], dtype=torch.float32, device=self.device)
                m1 = torch.as_tensor(batch['m1'], dtype=torch.float32, device=self.device)
                m2 = torch.as_tensor(batch['m2'], dtype=torch.float32, device=self.device)

                logp, ent, v, _ = self.evaluate_actions(obs, a1, a2, m1, m2)
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps,
                ) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(v, ret)
                entropy = ent.mean()
                loss = (
                    policy_loss
                    + self.cfg.vf_coef * value_loss
                    - self.cfg.ent_coef * entropy
                )
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                self.optim.step()

                with torch.no_grad():
                    clip_fracs.append(
                        float(((ratio - 1.0).abs() > self.cfg.clip_eps).float().mean().item())
                    )
                    approx_kls.append(float((old_logp - logp).mean().item()))
                    entropies.append(float(entropy.item()))
                    losses.append(float(loss.item()))

        self.update_steps += 1
        # Explained variance
        data = self.buffer.get()
        y_true = data['ret']
        y_pred = data['val']
        var_y = float(np.var(y_true))
        expl = (
            float('nan') if var_y < 1e-8
            else float(1.0 - np.var(y_true - y_pred) / var_y)
        )
        self.buffer.clear()
        return {
            'loss': float(np.mean(losses)) if losses else float('nan'),
            'clip_fraction': float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            'approx_kl': float(np.mean(approx_kls)) if approx_kls else 0.0,
            'entropy': float(np.mean(entropies)) if entropies else 0.0,
            'explained_variance': expl,
        }

    def save(self, path: Path | str) -> Path:
        self._ensure_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'config': asdict(self.cfg),
            'net': self.net.state_dict(),
            'env_steps': self.env_steps,
            'update_steps': self.update_steps,
        }
        self._torch.save(payload, path)
        return path

    def load(self, path: Path | str, *, map_location: Optional[str] = None) -> None:
        self._ensure_torch()
        torch = self._torch
        loc = map_location or self.cfg.device
        try:
            payload = torch.load(path, map_location=loc, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=loc)
        if 'config' in payload and isinstance(payload['config'], dict):
            for k, v in payload['config'].items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)
        key = 'net' if 'net' in payload else 'online'
        self.net.load_state_dict(payload[key])
        self.env_steps = int(payload.get('env_steps', 0))
        self.update_steps = int(payload.get('update_steps', 0))

    def load_bc_weights(self, path: Path | str) -> None:
        """Load actor–critic from BC checkpoint; switch to post-BC LR."""
        self.load(path)
        self.use_post_bc_lr()
