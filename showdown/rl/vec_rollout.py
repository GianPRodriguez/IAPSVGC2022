"""Motor de recolección asíncrona sobre entornos vectorizados (familia QD).

QD-PPO, QD-PGA y QD-PPGA comparten la misma forma de recolectar: un rival del
mosaico por episodio, pasos de PPO con máscaras autoregresivas y evaluación de
G partidas por candidato. Estas funciones hacen lo mismo que las versiones de
un entorno, pero con N entornos en procesos y decisiones en lote:

- ``run_vec_rollout``: recolecta ``steps`` transiciones repartidas entre los N
  entornos (pasos asíncronos: la red decide para los que ya respondieron).
- ``evaluate_policy_vec``: juega ``n_games`` partidas greedy en paralelo y
  devuelve fitness (Cand. 1) y BD medio, con la misma regla de elegibilidad.

Las funciones de cada método (``ppo_improve``, ``_collect``,
``evaluate_policy``) delegan aquí cuando reciben un entorno vectorizado.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def is_vec_env(env) -> bool:
    return hasattr(env, 'num_envs') and hasattr(env, 'step_async')


def is_mock_vec(venv) -> bool:
    """Todos los entornos con backend mock: el rival no se puede cambiar.

    Igual que la versión de un entorno, donde ``_ensure_foe`` es un no-op con
    MockBattleBackend: el mosaico no debe convertir un entorno mock en live.
    """
    return all(getattr(s, 'opponent', 'mock') == 'mock' for s in getattr(venv, 'specs', []))


def run_vec_rollout(
    venv,
    *,
    steps: int,
    act_fn: Callable[[np.ndarray, np.ndarray, List[int]], Dict[str, np.ndarray]],
    on_step: Callable[[int, Dict[str, Any]], None],
    boot_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    foe_sampler: Optional[Callable[[], str]] = None,
    should_flush: Optional[Callable[[], bool]] = None,
    flush_fn: Optional[Callable[[np.ndarray, List[bool]], None]] = None,
    final_flush: bool = False,
    heartbeat=None,
    hb_offset: int = 0,
    phase: str = '',
    hb_extras: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Recolecta ``steps`` transiciones; empieza episodios nuevos en todos los entornos.

    - ``act_fn(obs[k], m1[k], idx) -> {a1, a2, logp, value, m2}`` (arrays de k).
    - ``on_step(i, rec)``: ``rec`` con obs, m1, a1, a2, logp, value, m2, el
      resultado del worker (``result``) y ``boot_v`` (V(s') si turn_cap).
    - ``foe_sampler()``: rival del siguiente episodio de cada entorno (mosaico).
    - ``should_flush`` / ``flush_fn(obs_actual, last_skip)``: actualización
      on-policy; antes se esperan los entornos en vuelo.
    """
    n = int(venv.num_envs)
    budget = max(0, int(steps))
    if is_mock_vec(venv):
        foe_sampler = None
    out = {'taken': 0, 'last_obs': None, 'last_skip': [False] * n, 'restarts': 0,
           'ep_counts': {}}
    if budget <= 0:
        return out
    first = [foe_sampler() if foe_sampler else venv.specs[i].opponent for i in range(n)]
    obs, m1 = venv.set_opponents(first)
    current = list(first)
    ep_counts: Dict[str, int] = {}
    for f in current:
        ep_counts[f] = ep_counts.get(f, 0) + 1
    pending = [foe_sampler() if foe_sampler else None for _ in range(n)]
    flight: Dict[int, tuple] = {}
    taken = 0
    issued = 0
    last_skip = [False] * n
    restarts = 0
    ready = list(range(n))

    def _issue(idx: List[int]) -> None:
        nonlocal issued
        if not idx:
            return
        act = act_fn(obs[idx], m1[idx], idx)
        venv.step_async(idx, list(zip(np.asarray(act['a1']).tolist(),
                                      np.asarray(act['a2']).tolist())),
                        [pending[i] for i in idx])
        for k, i in enumerate(idx):
            flight[i] = (obs[i].copy(), m1[i].copy(), int(act['a1'][k]), int(act['a2'][k]),
                         float(act['logp'][k]), float(act['value'][k]),
                         np.asarray(act['m2'][k], dtype=np.float32))
        issued += len(idx)

    def _absorb(results) -> List[int]:
        nonlocal taken, restarts
        boot = np.zeros(len(results), dtype=np.float32)
        if boot_fn is not None:
            ks = [k for k, (_i, r) in enumerate(results)
                  if r['info']['trunc_reason'] == 'turn_cap'
                  and not (r['info']['skip_buffer'] or r['info']['used_fallback'])]
            if ks:
                boot[ks] = boot_fn(np.stack([results[k][1]['final_obs'] for k in ks]))
        done_idx = []
        for k, (i, r) in enumerate(results):
            o_i, m1_i, a1, a2, logp, v, m2 = flight.pop(i)
            info = r['info']
            skip = bool(info['skip_buffer'] or info['used_fallback'])
            last_skip[i] = skip
            on_step(i, {
                'obs': o_i, 'm1': m1_i, 'a1': a1, 'a2': a2, 'logp': logp, 'value': v,
                'm2': m2, 'result': r, 'skip': skip, 'boot_v': float(boot[k]),
            })
            if r.get('worker_restarted'):
                restarts += 1
            if r['terminated'] or r['truncated']:
                current[i] = r.get('next_opponent') or current[i]
                ep_counts[current[i]] = ep_counts.get(current[i], 0) + 1
                pending[i] = foe_sampler() if foe_sampler else None
            obs[i] = r['obs']
            m1[i] = r['mask']
            taken += 1
            done_idx.append(i)
        return done_idx

    while taken < budget:
        _issue(ready[:max(0, budget - issued)])
        if not flight:
            break
        ready = _absorb(venv.step_wait_any())
        if should_flush is not None and flush_fn is not None and should_flush():
            if flight:
                ready = ready + _absorb(venv.drain())
            flush_fn(obs, last_skip)
        if heartbeat is not None:
            extras = {'envs': n, 'restarts': restarts}
            if hb_extras is not None:
                extras.update(hb_extras())
            heartbeat.log(int(hb_offset) + taken, phase=phase or 'collect', extras=extras)
    if flight:
        _absorb(venv.drain())
    if final_flush and flush_fn is not None:
        flush_fn(obs, last_skip)
    out.update(taken=int(taken), last_obs=obs, last_skip=last_skip, restarts=restarts,
               ep_counts=ep_counts)
    return out


def evaluate_policy_vec(
    agent,
    venv,
    *,
    rng: np.random.Generator,
    n_games: int,
    g_min: int,
    use_mosaic: bool = True,
    foe_sampler: Optional[Callable[[np.random.Generator], str]] = None,
):
    """``n_games`` partidas greedy (Cand. 1) en paralelo; misma regla que la versión simple.

    fitness = media del retorno de todas las partidas; BD = media solo de las
    elegibles; se acepta si hay al menos ``g_min`` elegibles.
    """
    from showdown.rl.qd_ppo.evaluate import EvalResult
    from showdown.rl.reward import RewardCandidate

    if foe_sampler is None:
        from showdown.rl.qd_ppo.mosaic import sample_mosaic_foe as foe_sampler
    n = int(venv.num_envs)
    games = int(n_games)
    if is_mock_vec(venv):
        use_mosaic = False
    returns: List[float] = []
    bds: List[Tuple[float, float]] = []
    if games <= 0:
        return EvalResult(fitness=0.0, bd=None, g_eligible=0, n_games=0, accepted=False)
    active = list(range(min(n, games)))
    opps = []
    for i in range(n):
        if use_mosaic and i in active:
            opps.append(foe_sampler(rng))
        else:
            opps.append(venv.specs[i].opponent)
    venv.set_reward(RewardCandidate.SPARSE.value)
    try:
        obs, m1 = venv.set_opponents(opps)
        started = len(active)
        pending = [foe_sampler(rng) if (use_mosaic and started + k < games) else None
                   for k in range(n)]
        ready = list(active)
        busy = set()
        while len(returns) < games:
            idx = [i for i in ready if i not in busy]
            if idx:
                a1, a2, _lp, _v, _m2 = agent.act_batch(
                    obs[idx], m1[idx],
                    lambda a1s, _idx=idx: venv.mask2_subset(_idx, a1s),
                    greedy=True,
                )
                venv.step_async(idx, list(zip(a1.tolist(), a2.tolist())),
                                [pending[i] for i in idx])
                busy.update(idx)
            ready = []
            for i, r in venv.step_wait_any():
                busy.discard(i)
                obs[i] = r['obs']
                m1[i] = r['mask']
                if r['terminated'] or r['truncated']:
                    info = r['info']
                    returns.append(float(info.get('episode_return', 0.0)))
                    if info.get('archive_eligible') and info.get('bd_raw') is not None:
                        b = info['bd_raw']
                        bds.append((float(b[0]), float(b[1])))
                    if started < games:
                        started += 1
                        pending[i] = foe_sampler(rng) if (use_mosaic and started < games) else None
                        ready.append(i)
                    # si ya se lanzaron todas las partidas, el entorno queda ocioso
                else:
                    ready.append(i)
            if not busy and not ready and len(returns) < games:
                break
        venv.drain()
    finally:
        venv.set_reward(RewardCandidate.DENSE.value)
    returns = returns[:games]
    g = len(bds)
    fitness = float(np.mean(returns)) if returns else 0.0
    if g < int(g_min):
        return EvalResult(fitness=fitness, bd=None, g_eligible=g, n_games=len(returns),
                          accepted=False)
    return EvalResult(
        fitness=fitness,
        bd=(float(np.mean([b[0] for b in bds])), float(np.mean([b[1] for b in bds]))),
        g_eligible=g,
        n_games=len(returns),
        accepted=True,
    )


def make_qd_env(
    *,
    make_single: Callable[[], Any],
    mock: bool,
    num_envs: int,
    seed: int,
    start_ps: bool,
    account_prefix: str,
    combat_family: str,
    envs_per_server: int = 4,
    base_port: Optional[int] = None,
):
    """Entorno de un entrenador QD: simple (``num_envs<=1``) o vectorizado."""
    if int(num_envs) <= 1:
        return make_single()
    from showdown.rl.reward import RewardCandidate
    from showdown.rl.vec_env import EnvSpec, make_vec_env

    return make_vec_env(
        EnvSpec(
            opponent='mock' if mock else 'random',
            account_prefix=account_prefix,
            combat_family=combat_family,
            reward=RewardCandidate.DENSE.value,
            obs_kind='ppo',
            mask_kind='slot',
        ),
        num_envs=int(num_envs),
        seed=int(seed),
        envs_per_server=int(envs_per_server),
        base_port=base_port,
        start_ps=bool(start_ps) and not mock,
    )
