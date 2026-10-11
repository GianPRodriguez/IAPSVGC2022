"""Mutation burn + PPO-clip improve for preview bandit."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from showdown.rl.team_preview_env import TeamPreviewEnv
from showdown.rl.team_preview_ppo.network import apply_mask_logits, masked_entropy


def burn_rollout(
    env: TeamPreviewEnv,
    net,
    device,
    *,
    n_episodes: int,
    rng: np.random.Generator,
) -> int:
    """Sample IAM episodes; no learning / no replay. Returns count."""
    import torch
    net.eval()
    n = 0
    with torch.no_grad():
        for _ in range(int(n_episodes)):
            obs, info = env.reset()
            mask = info.get('action_mask')
            if mask is None:
                mask = env.action_masks()
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _v = net(o)
            z = apply_mask_logits(logits, m)
            dist = torch.distributions.Categorical(logits=z)
            a = int(dist.sample().item())
            env.step(a)
            n += 1
    return n


def pg_improve(
    env: TeamPreviewEnv,
    net,
    critic,
    actor_optim,
    critic_optim,
    device,
    *,
    n_episodes: int,
    epochs: int = 2,
    clip_eps: float = 0.2,
    ent_coef: float = 0.02,
    grad_clip: float = 0.5,
    minibatch: int = 128,
) -> dict:
    """Collect bandit episodes; PPO on actor vs shared critic; update critic."""
    import torch
    import torch.nn.functional as F

    obs_l: List[np.ndarray] = []
    act_l: List[int] = []
    logp_l: List[float] = []
    rew_l: List[float] = []
    mask_l: List[np.ndarray] = []

    net.train()
    critic.train()
    for _ in range(int(n_episodes)):
        obs, info = env.reset()
        mask = info.get('action_mask')
        if mask is None:
            mask = env.action_masks()
        o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = net(o)
            z = apply_mask_logits(logits, m)
            dist = torch.distributions.Categorical(logits=z)
            a_t = dist.sample()
            logp = float(dist.log_prob(a_t).item())
            a = int(a_t.item())
        _o2, r, term, trunc, _ = env.step(a)
        assert term or trunc
        obs_l.append(obs.astype(np.float32))
        act_l.append(a)
        logp_l.append(logp)
        rew_l.append(float(r))
        mask_l.append(np.asarray(mask, dtype=np.float32))

    if not obs_l:
        return {}

    obs_t = torch.as_tensor(np.stack(obs_l), dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act_l, dtype=torch.long, device=device)
    old_logp = torch.as_tensor(logp_l, dtype=torch.float32, device=device)
    ret = torch.as_tensor(rew_l, dtype=torch.float32, device=device)
    masks = torch.as_tensor(np.stack(mask_l), dtype=torch.float32, device=device)

    with torch.no_grad():
        v = critic(obs_t)
        adv = ret - v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = obs_t.shape[0]
    idx = np.arange(n)
    metrics = {'pg_loss': 0.0, 'v_loss': 0.0, 'entropy': 0.0}
    batches = 0
    B = min(int(minibatch), n)
    for ep in range(int(epochs)):
        np.random.default_rng(ep).shuffle(idx)
        for start in range(0, n, B):
            mb = idx[start: start + B]
            logits, _ = net(obs_t[mb])
            z = apply_mask_logits(logits, masks[mb])
            dist = torch.distributions.Categorical(logits=z)
            logp = dist.log_prob(act_t[mb])
            ratio = torch.exp(logp - old_logp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv[mb]
            pg_loss = -torch.min(surr1, surr2).mean()
            ent = masked_entropy(logits, masks[mb]).mean()
            loss_pi = pg_loss - ent_coef * ent
            actor_optim.zero_grad(set_to_none=True)
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            actor_optim.step()

            v_pred = critic(obs_t[mb])
            v_loss = F.mse_loss(v_pred, ret[mb])
            critic_optim.zero_grad(set_to_none=True)
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
            critic_optim.step()

            metrics['pg_loss'] += float(pg_loss.item())
            metrics['v_loss'] += float(v_loss.item())
            metrics['entropy'] += float(ent.item())
            batches += 1
    if batches:
        for k in metrics:
            metrics[k] /= batches
    return metrics
