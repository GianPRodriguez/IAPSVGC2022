"""Cand. 1 fitness + BD aggregation (docs/qd_ppo_spec.md)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from showdown.rl.env import VGCEnv
from showdown.rl.observation import build_observation_ppo
from showdown.rl.qd_ppo.mosaic import sample_mosaic_foe
from showdown.rl.reward import RewardCandidate

G_DEFAULT = 20
G_MIN = 10


@dataclass
class EvalResult:
    fitness: float
    bd: Optional[Tuple[float, float]]
    g_eligible: int
    n_games: int
    accepted: bool  # g >= g_min


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


def _ensure_foe(env: VGCEnv, foe: str, *, seed: int) -> VGCEnv:
    """Switch live opponent when backend supports it; mock is a no-op."""
    backend = env.backend
    if foe == 'mock' or type(backend).__name__ == 'MockBattleBackend':
        env.reward_candidate = RewardCandidate.SPARSE
        return env
    if hasattr(backend, 'set_opponent'):
        if getattr(backend, 'opponent', None) != foe:
            backend.set_opponent(foe)
        env.reward_candidate = RewardCandidate.SPARSE
        return env
    return env


def evaluate_policy(
    agent,
    env: VGCEnv,
    *,
    rng: np.random.Generator,
    n_games: int = G_DEFAULT,
    g_min: int = G_MIN,
    seed: int = 0,
    use_mosaic: bool = True,
) -> EvalResult:
    """
    Run G BO1 under Cand. 1 (greedy masked).

    f = mean over all G; BD = mean over eligible only; accept iff g >= g_min.
    """
    returns: list[float] = []
    bds: list[Tuple[float, float]] = []
    prev_cand = env.reward_candidate
    env.reward_candidate = RewardCandidate.SPARSE

    try:
        for gi in range(int(n_games)):
            if use_mosaic:
                foe = sample_mosaic_foe(rng)
                env = _ensure_foe(env, foe, seed=seed)
            obs, info = env.reset(seed=int(seed) + 10_000 + gi)
            assert env._snap is not None
            obs = build_observation_ppo(env._snap)
            m1, _ = _slot_masks(env)
            steps = 0
            while steps < 80:
                a1, a2, _lp, _v = agent.act(obs, m1, _m2_fn(env), greedy=True)
                obs, _r, term, trunc, info = env.step((a1, a2))
                assert env._snap is not None
                obs = build_observation_ppo(env._snap)
                m1, _ = _slot_masks(env)
                steps += 1
                if term or trunc:
                    break
            # Cand. 1 episodic return ≈ terminal sparse
            ret = float(info.get('episode_return', 0.0))
            returns.append(ret)
            tel = info.get('telemetry') or {}
            if tel.get('archive_eligible') or (
                env._bd is not None and env._bd.archive_eligible()
            ):
                bd_raw = info.get('bd_raw') or info.get('bd')
                if bd_raw is None and env._bd is not None:
                    bd_raw = env._bd.bd_vector()
                if bd_raw is not None:
                    bds.append((float(bd_raw[0]), float(bd_raw[1])))
    finally:
        env.reward_candidate = prev_cand

    g = len(bds)
    fitness = float(np.mean(returns)) if returns else 0.0
    if g < int(g_min):
        return EvalResult(
            fitness=fitness, bd=None, g_eligible=g, n_games=len(returns), accepted=False,
        )
    bd1 = float(np.mean([b[0] for b in bds]))
    bd2 = float(np.mean([b[1] for b in bds]))
    return EvalResult(
        fitness=fitness,
        bd=(bd1, bd2),
        g_eligible=g,
        n_games=len(returns),
        accepted=True,
    )
