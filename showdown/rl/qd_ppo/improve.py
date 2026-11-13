"""PPO improvement under Cand. 2 + mosaic (docs/qd_ppo_spec.md)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from showdown.rl.env import VGCEnv
from showdown.rl.observation import build_observation_ppo
from showdown.rl.ppo.agent import PPOAgent
from showdown.rl.qd_ppo.mosaic import sample_mosaic_foe
from showdown.rl.reward import RewardCandidate
from showdown.rl.train_log import TrainHeartbeat


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


def _bootstrap_value(agent: PPOAgent, obs: np.ndarray) -> float:
    import torch

    agent.net.eval()
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).view(1, -1)
        _z1, _z2, v = agent.net(x)
        return float(v.item())


def _ensure_foe(env: VGCEnv, foe: str) -> VGCEnv:
    backend = env.backend
    env.reward_candidate = RewardCandidate.DENSE
    if type(backend).__name__ == 'MockBattleBackend':
        return env
    if hasattr(backend, 'set_opponent') and getattr(backend, 'opponent', None) != foe:
        backend.set_opponent(foe)
    return env


def ppo_improve(
    agent: PPOAgent,
    env: VGCEnv,
    *,
    steps: int,
    rng: np.random.Generator,
    global_steps: int,
    t_total: int,
    seed: int = 0,
    use_mosaic: bool = True,
    heartbeat: Optional[TrainHeartbeat] = None,
    phase: str = '',
) -> Tuple[VGCEnv, int, dict]:
    """
    Collect up to ``steps`` env transitions under Cand. 2; hard-stop at t_total.

    Returns (env, steps_taken, metrics).
    """
    from showdown.rl.vec_rollout import is_vec_env

    budget = min(int(steps), max(0, int(t_total) - int(global_steps)))
    if is_vec_env(env):
        return _ppo_improve_vec(
            agent, env, budget=budget, rng=rng, global_steps=global_steps,
            use_mosaic=use_mosaic, heartbeat=heartbeat, phase=phase,
        )
    env.reward_candidate = RewardCandidate.DENSE
    if budget <= 0:
        return env, 0, {}

    if use_mosaic:
        env = _ensure_foe(env, sample_mosaic_foe(rng))

    obs, info = env.reset(seed=int(seed) + int(global_steps))
    assert env._snap is not None
    obs = build_observation_ppo(env._snap)
    m1, _ = _slot_masks(env)
    local = 0
    metrics: dict = {}
    trunc_reason = None
    skip = False
    boot_v = 0.0
    terminated = truncated = False

    while local < budget:
        a1, a2, logp, value = agent.act(obs, m1, _m2_fn(env), greedy=False)
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
            boot_v = _bootstrap_value(agent, next_obs)

        agent.observe(
            obs=obs, a1=a1, a2=a2, logp=logp, reward=float(reward_t), value=value,
            terminated=bool(terminated), m1=m1, m2=m2,
            skip_buffer=skip, trunc_reason=trunc_reason,
            bootstrap_value=boot_v,
        )
        local += 1
        obs, m1 = next_obs, _slot_masks(env)[0]

        if terminated or truncated:
            if use_mosaic:
                env = _ensure_foe(env, sample_mosaic_foe(rng))
            obs, info = env.reset()
            assert env._snap is not None
            obs = build_observation_ppo(env._snap)
            m1, _ = _slot_masks(env)

        if agent.is_rollout_full() or local >= budget:
            last_v = 0.0
            if not (terminated or truncated) and not skip:
                last_v = _bootstrap_value(agent, obs)
            elif trunc_reason == 'turn_cap':
                last_v = boot_v
            metrics = agent.update(last_value=last_v)

        if heartbeat is not None:
            heartbeat.log(
                int(global_steps) + local,
                phase=phase or 'improve',
                extras={
                    'loss': metrics.get('loss') if metrics else None,
                    'kl': metrics.get('approx_kl') if metrics else None,
                },
            )

        if int(global_steps) + local >= int(t_total):
            break

    return env, int(local), metrics


def _ppo_improve_vec(agent, venv, *, budget, rng, global_steps, use_mosaic, heartbeat, phase):
    """Versión vectorizada: mismas reglas (Cand. 2, mosaico por episodio, update al llenar)."""
    from showdown.rl.vec_rollout import run_vec_rollout

    if budget <= 0:
        return venv, 0, {}
    n = int(venv.num_envs)
    buf = getattr(agent, 'buffer', None)
    if getattr(buf, 'num_envs', None) != n:
        agent.use_vec_buffer(n)
    venv.set_reward(RewardCandidate.DENSE.value)
    metrics: dict = {}

    def act_fn(obs, m1, idx):
        a1, a2, logp, value, m2 = agent.act_batch(
            obs, m1, lambda a1s: venv.mask2_subset(idx, a1s), greedy=False,
        )
        return {'a1': a1, 'a2': a2, 'logp': logp, 'value': value, 'm2': m2}

    def on_step(i, rec):
        r = rec['result']
        agent.observe(
            obs=rec['obs'], a1=rec['a1'], a2=rec['a2'], logp=rec['logp'],
            reward=float(r['reward']), value=rec['value'],
            terminated=bool(r['terminated']), m1=rec['m1'], m2=rec['m2'],
            skip_buffer=rec['skip'], trunc_reason=r['info']['trunc_reason'],
            bootstrap_value=rec['boot_v'], env_index=i,
        )

    def flush(obs, last_skip):
        nonlocal metrics
        if len(agent.buffer) == 0:
            return
        last_v = agent.values_batch(obs)
        for i, sk in enumerate(last_skip):
            if sk:
                last_v[i] = 0.0
        metrics = agent.update(last_value=last_v)

    out = run_vec_rollout(
        venv, steps=budget, act_fn=act_fn, on_step=on_step,
        boot_fn=agent.values_batch,
        foe_sampler=(lambda: sample_mosaic_foe(rng)) if use_mosaic else None,
        should_flush=agent.is_rollout_full, flush_fn=flush, final_flush=True,
        heartbeat=heartbeat, hb_offset=int(global_steps), phase=phase or 'improve',
        hb_extras=lambda: {'loss': metrics.get('loss'), 'kl': metrics.get('approx_kl')},
    )
    return venv, int(out['taken']), metrics
