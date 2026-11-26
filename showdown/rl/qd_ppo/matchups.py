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


Transition = Dict[str, object]


def _ret(info) -> float:
    """Resultado Cand. 1 de la partida (``outcome``); si falta, el retorno acumulado."""
    o = info.get('outcome')
    return float(o) if o is not None else float(info.get('episode_return', 0.0))


def _slot_from_joint(jm: np.ndarray) -> np.ndarray:
    """Máscara del hueco 1 a partir de la conjunta (1600 = 40 × 40)."""
    n = int(round(np.sqrt(jm.shape[-1])))
    return (np.asarray(jm, dtype=np.float32).reshape(-1, n, n).max(axis=-1) > 0).astype(np.float32)


def play_matchups(agent, env, matches: Sequence[Match], *, max_steps: int = 80,
                  reward: str = 'sparse', on_transition: Optional[Callable[[Transition], None]] = None,
                  greedy: bool = True) -> MatchResults:
    """Juega ``matches`` y devuelve el resultado Cand. 1 y el descriptor de cada partida, en orden.

    ``reward``: recompensa activa durante las partidas (``'dense'`` = Cand. 2 para el replay del
    crítico de J 2.0; el resultado devuelto es siempre Cand. 1). ``on_transition``: recibe cada paso
    ``{obs, action (conjunta), reward, next_obs, terminated, truncated, mask, next_mask}``; las máscaras
    son las conjuntas si el entorno las da (``mask_kind='joint'``).
    """
    from showdown.rl.vec_rollout import is_vec_env

    if is_vec_env(env):
        return _play_vec(agent, env, list(matches), reward=reward, on_transition=on_transition,
                         greedy=greedy)
    return _play_single(agent, env, list(matches), max_steps=max_steps, reward=reward,
                        on_transition=on_transition, greedy=greedy)


def _act(agent, venv, idx, obs, mask, greedy):
    o = np.stack([obs[i] for i in idx])
    m = np.stack([mask[i] for i in idx])
    if m.shape[-1] > 40:  # máscara conjunta: hueco 1 = alguna fila legal; hueco 2 = la fila de a1
        m1 = _slot_from_joint(m)
        n = m1.shape[-1]

        def m2(a1s, _m=m):
            return np.stack([_m[k].reshape(n, n)[int(a)] for k, a in enumerate(a1s)]).astype(np.float32)
        return agent.act_batch(o, m1, m2, greedy=greedy)
    return agent.act_batch(o, m, lambda a1s, _idx=idx: venv.mask2_subset(_idx, a1s), greedy=greedy)


def _play_vec(agent, venv, matches: List[Match], *, reward: str = 'sparse',
              on_transition=None, greedy: bool = True) -> MatchResults:
    n_games = len(matches)
    returns: List[Optional[float]] = [None] * n_games
    bds: List[Optional[Tuple[float, float]]] = [None] * n_games
    if n_games == 0:
        return MatchResults([], [])
    n = int(venv.num_envs)
    venv.set_reward((RewardCandidate.DENSE if reward == 'dense' else RewardCandidate.SPARSE).value)
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
            a1, a2, _lp, _v, _m2 = _act(agent, venv, idx, obs, mask, greedy)
            act = {i: (int(a1[k]), int(a2[k])) for k, i in enumerate(idx)}
            venv.step_async(idx, [act[i] for i in idx], on_done)
            for i, r in venv.drain():
                k = playing[i]
                info = r['info']
                done = bool(r['terminated'] or r['truncated'])
                crash = info.get('trunc_reason') == 'crash'
                if on_transition is not None and not crash and not info.get('skip_buffer'):
                    nobs = r.get('final_obs') if done else r['obs']
                    on_transition({
                        'obs': obs[i], 'action': act[i][0] * 40 + act[i][1], 'reward': float(r['reward']),
                        'next_obs': nobs, 'terminated': bool(r['terminated']),
                        'truncated': bool(r['truncated']), 'mask': mask[i],
                        'next_mask': None if done else r['mask'], 'bd_step': info.get('bd_step'),
                    })
                if done:
                    if not crash:   # una caída no es un empate
                        returns[k] = _ret(info)
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


def _play_single(agent, env, matches: List[Match], *, max_steps: int, reward: str = 'sparse',
                 on_transition=None, greedy: bool = True) -> MatchResults:
    from showdown.rl.observation import build_observation_ppo
    from showdown.rl.qd_ppo.evaluate import _ensure_foe, _m2_fn, _slot_masks

    joint = on_transition is not None
    if joint:
        from showdown.rl.ddqn.train import _joint_masks
    returns: List[float] = []
    bds: List[Optional[Tuple[float, float]]] = []
    prev = env.reward_candidate
    try:
        for mt in matches:
            env = _ensure_foe(env, str(mt.get('opponent') or 'mock'), seed=0)
            env.reward_candidate = RewardCandidate.DENSE if reward == 'dense' else RewardCandidate.SPARSE
            backend = getattr(env, 'backend', None)
            if backend is not None and hasattr(backend, '_team_split'):
                backend._team_split = mt.get('split')
            env.reset(seed=int(mt.get('seed') or 0))
            obs = build_observation_ppo(env._snap)
            m1, _ = _slot_masks(env)
            jm = np.asarray(_joint_masks(env), dtype=np.float32) if joint else None
            info: dict = {}
            for _ in range(int(max_steps)):
                a1, a2, _lp, _v = agent.act(obs, m1, _m2_fn(env), greedy=greedy)
                bd0 = env._bd
                c0 = (bd0.offensive, bd0.status_setup, bd0.paired_attacks, bd0.focus_turns)
                _o, rew, term, trunc, info = env.step((a1, a2))
                nobs = build_observation_ppo(env._snap)
                done = bool(term or trunc)
                if joint and not info.get('skip_buffer'):
                    from showdown.rl.vec_env import bd_step_proxies
                    c1 = (bd0.offensive, bd0.status_setup, bd0.paired_attacks, bd0.focus_turns)
                    njm = None if done else np.asarray(_joint_masks(env), dtype=np.float32)
                    on_transition({'obs': obs, 'action': int(a1) * 40 + int(a2), 'reward': float(rew),
                                   'next_obs': nobs, 'terminated': bool(term), 'truncated': bool(trunc),
                                   'mask': jm, 'next_mask': njm, 'bd_step': bd_step_proxies(c0, c1)})
                    jm = njm
                obs = nobs
                if done:
                    break
                m1, _ = _slot_masks(env)
            returns.append(_ret(info))
            b = None
            if env._bd is not None and env._bd.archive_eligible():
                v = env._bd.bd_vector()
                b = (float(v[0]), float(v[1]))
            bds.append(b)
    finally:
        env.reward_candidate = prev
    return MatchResults(returns, bds)


def play_matchups_multi(agents, env, groups: Sequence[Sequence[Match]], *, reward: str = 'sparse',
                        on_transition: Optional[Callable[[Transition], None]] = None,
                        greedy: bool = True) -> List[MatchResults]:
    """Varias políticas a la vez: ``groups[g]`` son los enfrentamientos de ``agents[g]``.

    En el entorno vectorizado todas las partidas comparten los entornos (cada entorno juega con la
    política de su partida) y avanzan de forma asíncrona: ningún entorno espera a la partida más
    lenta. Fuera del vectorizado, se juegan en orden.
    """
    from showdown.rl.vec_rollout import is_vec_env

    if not is_vec_env(env):
        return [play_matchups(a, env, list(g), reward=reward, on_transition=on_transition, greedy=greedy)
                for a, g in zip(agents, groups)]
    flat = [(gi, m) for gi, g in enumerate(groups) for m in g]
    out_r: List[List[Optional[float]]] = [[None] * len(g) for g in groups]
    out_b: List[List[Optional[Tuple[float, float]]]] = [[None] * len(g) for g in groups]
    pos = []
    count = [0] * len(groups)
    for gi, _m in flat:
        pos.append(count[gi])
        count[gi] += 1
    if not flat:
        return [MatchResults([], []) for _ in groups]
    venv = env
    n = int(venv.num_envs)
    venv.set_reward((RewardCandidate.DENSE if reward == 'dense' else RewardCandidate.SPARSE).value)
    try:
        first = list(range(min(n, len(flat))))
        venv.start_matches(first, [flat[k][1] for k in first])
        o, mm = venv.last_obs_mask(first)
        obs = {i: o[k] for k, i in enumerate(first)}
        mask = {i: mm[k] for k, i in enumerate(first)}
        playing = {i: i for i in first}          # entorno -> índice en ``flat``
        reserved: Dict[int, int] = {}
        nxt = len(first)
        busy = set()
        last_act: Dict[int, Tuple[int, int]] = {}
        while playing:
            ready = [i for i in sorted(playing) if i not in busy]
            for i in ready:
                if i not in reserved and nxt < len(flat):
                    reserved[i] = nxt
                    nxt += 1
            by_agent: Dict[int, List[int]] = {}
            for i in ready:
                by_agent.setdefault(flat[playing[i]][0], []).append(i)
            for gi, idx in by_agent.items():
                a1, a2, _lp, _v, _m2 = _act(agents[gi], venv, idx, obs, mask, greedy)
                acts = [(int(a1[k]), int(a2[k])) for k in range(len(idx))]
                for i, a in zip(idx, acts):
                    last_act[i] = a
                venv.step_async(idx, acts, [dict(flat[reserved[i]][1]) if i in reserved else None
                                            for i in idx])
                busy.update(idx)
            for i, r in venv.step_wait_any():
                busy.discard(i)
                k = playing[i]
                gi, pk = flat[k][0], pos[k]
                info = r['info']
                done = bool(r['terminated'] or r['truncated'])
                crash = info.get('trunc_reason') == 'crash'
                if on_transition is not None and not crash and not info.get('skip_buffer'):
                    a = last_act[i]
                    on_transition({
                        'obs': obs[i], 'action': a[0] * 40 + a[1], 'reward': float(r['reward']),
                        'next_obs': r.get('final_obs') if done else r['obs'],
                        'terminated': bool(r['terminated']), 'truncated': bool(r['truncated']),
                        'mask': mask[i], 'next_mask': None if done else r['mask'],
                        'bd_step': info.get('bd_step'), 'group': gi,
                    })
                if done:
                    if not crash:
                        out_r[gi][pk] = _ret(info)
                        if info.get('archive_eligible') and info.get('bd_raw') is not None:
                            b = info['bd_raw']
                            out_b[gi][pk] = (float(b[0]), float(b[1]))
                    if i not in reserved:
                        playing.pop(i)
                        continue
                    k2 = reserved.pop(i)
                    if r.get('worker_restarted'):
                        venv.start_matches([i], [flat[k2][1]])
                        oo, m2 = venv.last_obs_mask([i])
                        r = {'obs': oo[0], 'mask': m2[0]}
                    playing[i] = k2
                obs[i], mask[i] = r['obs'], r['mask']
        venv.drain()
    finally:
        venv.set_reward(RewardCandidate.DENSE.value)
    return [MatchResults(out_r[g], out_b[g]) for g in range(len(groups))]

