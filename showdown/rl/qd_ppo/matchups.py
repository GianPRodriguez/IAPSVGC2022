"""Enfrentamientos fijos y evaluación pareada (QD-PPO 2.0, docs/qd_ppo_spec.md §I 2.0).

Un *enfrentamiento* es ``{'opponent', 'seed', 'split'}``: la semilla fija los equipos de los dos
lados (``pokeenv_backend._default_team_metas``) y ``split`` el grupo de equipos (``None`` = el de
entrenamiento, ``'val'`` = validación reservada). Dos políticas que juegan la misma lista de
enfrentamientos se comparan con números aleatorios comunes: la varianza por equipos y rival
desaparece de la diferencia (el azar del simulador sigue siendo propio de cada partida).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from showdown.rl.reward import RewardCandidate

Match = Dict[str, object]


def make_matchups(
    rng: np.random.Generator,
    n: int,
    foe_sampler: Callable[[np.random.Generator], str],
    *,
    split: Optional[str] = None,
) -> List[Match]:
    """``n`` enfrentamientos: rival del mosaico y semilla de equipos al azar."""
    return [
        {'opponent': foe_sampler(rng), 'seed': int(rng.integers(0, 2**31 - 1)), 'split': split}
        for _ in range(int(n))
    ]


@dataclass
class MatchResults:
    """Por partida, en el orden de los enfrentamientos; ``None`` = partida caída (no cuenta)."""

    returns: List[Optional[float]]
    bds: List[Optional[Tuple[float, float]]] = field(default_factory=list)

    def valid_returns(self) -> List[float]:
        return [float(r) for r in self.returns if r is not None]

    @property
    def fitness(self) -> float:
        v = self.valid_returns()
        return float(np.mean(v)) if v else 0.0

    def eligible_bds(self) -> List[Tuple[float, float]]:
        return [b for b in self.bds if b is not None]

    def bd(self) -> Optional[Tuple[float, float]]:
        e = self.eligible_bds()
        if not e:
            return None
        return float(np.mean([b[0] for b in e])), float(np.mean([b[1] for b in e]))


def paired_diff(a: MatchResults, b: MatchResults) -> float:
    """Diferencia media de retorno, partida a partida (mismos enfrentamientos; sin las caídas)."""
    d = [float(x) - float(y) for x, y in zip(a.returns, b.returns) if x is not None and y is not None]
    return float(np.mean(d)) if d else 0.0


def play_matchups(agent, env, matches: Sequence[Match], *, max_steps: int = 80) -> MatchResults:
    """Juega ``matches`` (greedy, Cand. 1) y devuelve retorno y descriptor por partida, en orden."""
    from showdown.rl.vec_rollout import is_vec_env

    if is_vec_env(env):
        return _play_vec(agent, env, list(matches))
    return _play_single(agent, env, list(matches), max_steps=max_steps)


def _play_vec(agent, venv, matches: List[Match]) -> MatchResults:
    n_games = len(matches)
    returns: List[Optional[float]] = [None] * n_games
    bds: List[Optional[Tuple[float, float]]] = [None] * n_games
    if n_games == 0:
        return MatchResults([], [])
    n = int(venv.num_envs)
    venv.set_reward(RewardCandidate.SPARSE.value)
    try:
        first = list(range(min(n, n_games)))
        venv.start_matches(first, [matches[k] for k in first])
        o, m = venv.last_obs_mask(first)
        obs: Dict[int, np.ndarray] = {i: o[k] for k, i in enumerate(first)}
        mask: Dict[int, np.ndarray] = {i: m[k] for k, i in enumerate(first)}
        playing: Dict[int, int] = {i: i for i in first}  # entorno -> índice de partida
        reserved: Dict[int, int] = {}                     # entorno -> partida siguiente
        nxt = len(first)
        while playing:
            idx = sorted(playing)
            # Cada entorno reserva su siguiente enfrentamiento: si el episodio termina en este
            # paso, el worker empieza esa partida (start_match) en lugar de un reinicio cualquiera.
            for i in idx:
                if i not in reserved and nxt < n_games:
                    reserved[i] = nxt
                    nxt += 1
            on_done = [dict(matches[reserved[i]]) if i in reserved else None for i in idx]
            a1, a2, _lp, _v, _m2 = agent.act_batch(
                np.stack([obs[i] for i in idx]), np.stack([mask[i] for i in idx]),
                lambda a1s, _idx=idx: venv.mask2_subset(_idx, a1s), greedy=True,
            )
            venv.step_async(idx, list(zip(a1.tolist(), a2.tolist())), on_done)
            for i, r in venv.drain():
                k = playing[i]
                if r['terminated'] or r['truncated']:
                    info = r['info']
                    if info.get('trunc_reason') != 'crash':   # una caída no es un empate
                        returns[k] = float(info.get('episode_return', 0.0))
                        if info.get('archive_eligible') and info.get('bd_raw') is not None:
                            b = info['bd_raw']
                            bds[k] = (float(b[0]), float(b[1]))
                    if i not in reserved:
                        playing.pop(i)
                        continue
                    k2 = reserved.pop(i)
                    if r.get('worker_restarted'):   # el worker se reinició: se empieza ahora
                        venv.start_matches([i], [matches[k2]])
                        oo, mm = venv.last_obs_mask([i])
                        r = {'obs': oo[0], 'mask': mm[0]}
                    playing[i] = k2
                obs[i], mask[i] = r['obs'], r['mask']
    finally:
        venv.set_reward(RewardCandidate.DENSE.value)
    return MatchResults(returns, bds)


def _play_single(agent, env, matches: List[Match], *, max_steps: int) -> MatchResults:
    from showdown.rl.observation import build_observation_ppo
    from showdown.rl.qd_ppo.evaluate import _ensure_foe, _m2_fn, _slot_masks

    returns: List[float] = []
    bds: List[Optional[Tuple[float, float]]] = []
    prev = env.reward_candidate
    try:
        for mt in matches:
            env = _ensure_foe(env, str(mt.get('opponent') or 'mock'), seed=0)
            env.reward_candidate = RewardCandidate.SPARSE
            backend = getattr(env, 'backend', None)
            if backend is not None and hasattr(backend, '_team_split'):
                backend._team_split = mt.get('split')
            env.reset(seed=int(mt.get('seed') or 0))
            obs = build_observation_ppo(env._snap)
            m1, _ = _slot_masks(env)
            info: dict = {}
            for _ in range(int(max_steps)):
                a1, a2, _lp, _v = agent.act(obs, m1, _m2_fn(env), greedy=True)
                _o, _r, term, trunc, info = env.step((a1, a2))
                obs = build_observation_ppo(env._snap)
                m1, _ = _slot_masks(env)
                if term or trunc:
                    break
            returns.append(float(info.get('episode_return', 0.0)))
            b = None
            if env._bd is not None and env._bd.archive_eligible():
                v = env._bd.bd_vector()
                b = (float(v[0]), float(v[1]))
            bds.append(b)
    finally:
        env.reward_candidate = prev
    return MatchResults(returns, bds)
