"""PG emit + mutation burn-rollout (docs/qd_pga_map_elites_spec.md)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from showdown.rl.env import VGCEnv
from showdown.rl.observation import build_observation_ppo
from showdown.rl.ppo.agent import PPOAgent
from showdown.rl.ppo.buffer import RolloutBuffer
from showdown.rl.qd_pga.critic import SharedCritic
from showdown.rl.qd_pga.mosaic import sample_mosaic_foe
from showdown.rl.qd_pga.replay import ValueReplay
from showdown.rl.reward import RewardCandidate

K_PG_DEFAULT = 2
G_C_DEFAULT = 8
CRITIC_BATCH = 256


def _slot_masks(env: VGCEnv, *, slot1_action_id: Optional[int] = None):
    from showdown.rl.action_space import build_autoregressive_masks, decode_slot_action, build_slot_mask

    live = env.backend.action_masks(slot1_action_id=slot1_action_id)
    if live is not None:
        return (
            np.asarray(live[0], dtype=np.float32),
            np.asarray(live[1], dtype=np.float32),
        )
    assert env._snap is not None
    if slot1_action_id is None:
        return build_autoregressive_masks(env._snap, slot1_action_id=None)
    m1, _ = build_autoregressive_masks(env._snap, slot1_action_id=None)
    m2 = build_slot_mask(env._snap, 1, other_slot_action=decode_slot_action(slot1_action_id))
    return m1, m2


def _m2_fn(env: VGCEnv):
    def _fn(a1: int) -> np.ndarray:
        _m1, m2 = _slot_masks(env, slot1_action_id=a1)
        return np.asarray(m2, dtype=np.float32)

    return _fn


def _ensure_foe(env: VGCEnv, foe: str) -> VGCEnv:
    backend = env.backend
    env.reward_candidate = RewardCandidate.DENSE
    if type(backend).__name__ == 'MockBattleBackend':
        return env
    if hasattr(backend, 'set_opponent') and getattr(backend, 'opponent', None) != foe:
        backend.set_opponent(foe)
    return env


def _collect(
    agent: PPOAgent,
    env: VGCEnv,
    critic: SharedCritic,
    *,
    steps: int,
    rng: np.random.Generator,
    global_steps: int,
    t_total: int,
    seed: int,
    use_mosaic: bool,
    replay: Optional[ValueReplay],
    write_replay: bool,
) -> Tuple[VGCEnv, int, RolloutBuffer, dict]:
    """Collect up to ``steps`` env transitions; values from shared critic."""
    env.reward_candidate = RewardCandidate.DENSE
    budget = min(int(steps), max(0, int(t_total) - int(global_steps)))
    buf = RolloutBuffer(capacity=max(1, budget))
    meta = {
        'last_obs': None,
        'terminated': False,
        'truncated': False,
        'skip': False,
        'trunc_reason': None,
        'boot_v': 0.0,
    }
    if budget <= 0:
        return env, 0, buf, meta

    if use_mosaic:
        env = _ensure_foe(env, sample_mosaic_foe(rng))

    obs, info = env.reset(seed=int(seed) + int(global_steps))
    assert env._snap is not None
    obs = build_observation_ppo(env._snap)
    m1, _ = _slot_masks(env)
    local = 0
    skip = False
    trunc_reason = None
    terminated = truncated = False
    boot_v = 0.0

    while local < budget:
        # Sample from actor; value from shared V_φ
        a1, a2, logp, _v_actor = agent.act(obs, m1, _m2_fn(env), greedy=False)
        value = critic.value(obs)
        m2 = _m2_fn(env)(a1)
        next_obs, reward_t, terminated, truncated, info = env.step((a1, a2))
        assert env._snap is not None
        next_obs = build_observation_ppo(env._snap)
        skip = bool(info.get('skip_buffer'))
        trunc_reason = info.get('trunc_reason')
        eng = info.get('engine') or {}
        if isinstance(eng, dict) and eng.get('used_fallback'):
            skip = True
        boot_v = 0.0
        if trunc_reason == 'turn_cap' and not skip:
            boot_v = critic.value(next_obs)

        buf.add(
            obs=obs, a1=a1, a2=a2, logp=logp, reward=float(reward_t), value=value,
            terminated=bool(terminated), m1=m1, m2=m2,
            skip_buffer=skip, trunc_reason=trunc_reason,
            bootstrap_value=boot_v,
        )
        if write_replay and replay is not None:
            replay.add(
                s=obs, r=float(reward_t), s_next=next_obs,
                terminated=bool(terminated),
                skip_buffer=skip, trunc_reason=trunc_reason,
            )
        local += 1
        obs, m1 = next_obs, _slot_masks(env)[0]
        meta = {
            'last_obs': obs,
            'terminated': bool(terminated),
            'truncated': bool(truncated),
            'skip': bool(skip),
            'trunc_reason': trunc_reason,
            'boot_v': float(boot_v),
        }

        if terminated or truncated:
            if use_mosaic:
                env = _ensure_foe(env, sample_mosaic_foe(rng))
            obs, info = env.reset()
            assert env._snap is not None
            obs = build_observation_ppo(env._snap)
            m1, _ = _slot_masks(env)

        if int(global_steps) + local >= int(t_total):
            break

    return env, int(local), buf, meta


def _ppo_actor_update(
    agent: PPOAgent,
    buf: RolloutBuffer,
    critic: SharedCritic,
    *,
    meta: dict,
    k_pg: int = K_PG_DEFAULT,
) -> dict:
    """GAE vs shared V_φ; PPO-clip on actor only (no local value loss)."""
    if len(buf) == 0:
        return {}
    last_obs = meta.get('last_obs')
    terminated = bool(meta.get('terminated'))
    truncated = bool(meta.get('truncated'))
    skip = bool(meta.get('skip'))
    trunc_reason = meta.get('trunc_reason')
    boot_v = float(meta.get('boot_v') or 0.0)
    last_v = 0.0
    if not (terminated or truncated) and not skip and last_obs is not None:
        last_v = critic.value(last_obs)
    elif trunc_reason == 'turn_cap':
        last_v = boot_v

    buf.compute_gae(
        last_value=last_v,
        gamma=agent.cfg.gamma,
        lam=agent.cfg.gae_lambda,
        normalize_adv=True,
    )

    torch_mod = agent._torch
    clip_eps = agent.cfg.clip_eps
    ent_coef = agent.cfg.ent_coef
    clip_fracs = []
    losses = []
    # Actor params only
    actor_params = [p for n, p in agent.net.named_parameters() if not n.startswith('value.')]
    agent.net.train()
    for _ in range(int(k_pg)):
        for batch in buf.minibatches(agent.cfg.minibatch_size, agent._rng):
            obs = torch_mod.as_tensor(batch['obs'], dtype=torch_mod.float32, device=agent.device)
            a1 = torch_mod.as_tensor(batch['a1'], dtype=torch_mod.long, device=agent.device)
            a2 = torch_mod.as_tensor(batch['a2'], dtype=torch_mod.long, device=agent.device)
            old_logp = torch_mod.as_tensor(batch['logp'], dtype=torch_mod.float32, device=agent.device)
            adv = torch_mod.as_tensor(batch['adv'], dtype=torch_mod.float32, device=agent.device)
            m1 = torch_mod.as_tensor(batch['m1'], dtype=torch_mod.float32, device=agent.device)
            m2 = torch_mod.as_tensor(batch['m2'], dtype=torch_mod.float32, device=agent.device)

            logp, ent, _v, _ = agent.evaluate_actions(obs, a1, a2, m1, m2)
            ratio = torch_mod.exp(logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch_mod.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch_mod.min(surr1, surr2).mean()
            entropy = ent.mean()
            loss = policy_loss - ent_coef * entropy
            agent.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch_mod.nn.utils.clip_grad_norm_(actor_params, agent.cfg.grad_clip)
            # Zero value-head grads so Adam step does not move V
            if agent.net.value.weight.grad is not None:
                agent.net.value.weight.grad.zero_()
            if agent.net.value.bias is not None and agent.net.value.bias.grad is not None:
                agent.net.value.bias.grad.zero_()
            agent.optim.step()
            with torch_mod.no_grad():
                clip_fracs.append(
                    float(((ratio - 1.0).abs() > clip_eps).float().mean().item())
                )
                losses.append(float(loss.item()))
    agent.update_steps += 1
    buf.clear()
    return {
        'policy_loss': float(np.mean(losses)) if losses else float('nan'),
        'clip_fraction': float(np.mean(clip_fracs)) if clip_fracs else 0.0,
    }


def pg_improve(
    agent: PPOAgent,
    env: VGCEnv,
    critic: SharedCritic,
    replay: ValueReplay,
    *,
    steps: int,
    rng: np.random.Generator,
    global_steps: int,
    t_total: int,
    seed: int = 0,
    use_mosaic: bool = True,
    k_pg: int = K_PG_DEFAULT,
    g_c: int = G_C_DEFAULT,
) -> Tuple[VGCEnv, int, dict]:
    """Collect, write replay, critic TD, then actor PPO-clip K epochs."""
    env, taken, buf, meta = _collect(
        agent, env, critic,
        steps=steps, rng=rng, global_steps=global_steps, t_total=t_total,
        seed=seed, use_mosaic=use_mosaic, replay=replay, write_replay=True,
    )
    metrics: dict = {}
    if taken > 0 and len(buf) > 0:
        metrics.update(_ppo_actor_update(agent, buf, critic, meta=meta, k_pg=k_pg))
    metrics.update(
        critic.td_update(replay, n_updates=g_c, batch_size=CRITIC_BATCH, rng=rng)
    )
    return env, taken, metrics


def mut_burn_rollout(
    agent: PPOAgent,
    env: VGCEnv,
    critic: SharedCritic,
    *,
    steps: int,
    rng: np.random.Generator,
    global_steps: int,
    t_total: int,
    seed: int = 0,
    use_mosaic: bool = True,
) -> Tuple[VGCEnv, int, dict]:
    """Burn T_emit Cand. 2 steps; no replay write; no PG."""
    env, taken, _buf, _meta = _collect(
        agent, env, critic,
        steps=steps, rng=rng, global_steps=global_steps, t_total=t_total,
        seed=seed, use_mosaic=use_mosaic, replay=None, write_replay=False,
    )
    return env, taken, {}
