"""Simetrías de la ObsSpec v2 para aumentación de datos (rl_env_spec v1.20 §3.d).

En dobles, la posición izquierda/derecha no cambia la mecánica (todos son
adyacentes), y el orden de las 2 reservas propias solo fija el índice de la
acción de cambio. Por eso un mismo estado aparece con varias codificaciones
equivalentes. El grupo es Z2³ (8 elementos, todos involuciones que conmutan):

  bit 0  σ_o  intercambia nuestros activos a↔b  (y slot 1 ↔ slot 2 de la acción)
  bit 1  σ_f  intercambia los activos rivales a↔b (objetivo 0 ↔ 1 en movimientos y mega)
  bit 2  σ_b  intercambia nuestras reservas 1↔2 (cambio 32 ↔ 33); solo si ambas
              están presentes y vivas (si no, la codificación canónica pone la viva
              primero y el estado permutado no existiría)

Para cada elemento ``g``: ``obs' = obs[OBS_PERM[g]]``, ids por slot
``a' = SLOT_PERM[g][a]`` y máscaras ``m' = m[SLOT_PERM[g]]`` (involuciones).
σ_o además intercambia los slots. Las permutaciones se verifican contra el
constructor de la obs en ``tests/test_rl_symmetry.py``.

Uso: DDQN (replay, exacto), críticos TD (exacto) y PPO/QD (término de
consistencia, ver ``ppo_symmetry_batch``). ``RL_SYM_AUG=0`` lo apaga (ablación).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

from showdown.rl.constants import (
    FOE_MOVE_DIM,
    JOINT_ACTION_N,
    MEGA_BASE,
    MOVE_SLOT_DIM,
    N_FOE_MONS,
    N_MOVE_ACTIONS,
    N_OUR_MONS,
    OBS_DIM,
    OTS_MON_DIM,
    SLOT_ACTION_N,
    SWITCH_BASE,
)

SYM_ENV = 'RL_SYM_AUG'
N_ELEMENTS = 8
BIT_OUR, BIT_FOE, BIT_BENCH = 1, 2, 4


def sym_aug_enabled(flag: bool = True) -> bool:
    """Config del agente AND variable de entorno (``RL_SYM_AUG=0`` apaga)."""
    env = str(os.environ.get(SYM_ENV, '1')).strip().lower()
    return bool(flag) and env not in ('0', 'false', 'no', 'off')


def _swap(p: np.ndarray, a: int, b: int, n: int = 1) -> None:
    ia = p[a:a + n].copy()
    p[a:a + n] = p[b:b + n]
    p[b:b + n] = ia


@lru_cache(maxsize=1)
def _base_perms() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from showdown.rl.observation import obs_layout

    lay = obs_layout()
    ident = np.arange(OBS_DIM, dtype=np.int64)
    pm, pf, pb = ident.copy(), ident.copy(), ident.copy()
    mb = lay['our_1'][0] - lay['our_0'][0]
    om = lay['our_moves'][0]
    fm = lay['foe_moves'][0]
    pr = lay['pairs'][0]
    fd = lay['field'][0]
    n_cell = N_OUR_MONS * N_FOE_MONS

    # σ_o: nuestros activos a ↔ b
    _swap(pm, lay['our_0'][0], lay['our_1'][0], mb)
    _swap(pm, om, om + 4 * MOVE_SLOT_DIM, 4 * MOVE_SLOT_DIM)
    for k in range(8):  # en cada movimiento rival: daño/KO sobre NUESTRO activo 0 ↔ 1
        o = fm + k * FOE_MOVE_DIM
        for i in (35, 37, 41):
            _swap(pm, o + i, o + i + 1)
    for m in range(3):  # filas 0 ↔ 1 de las 3 matrices nuestros × rival
        base = pr + m * n_cell
        _swap(pm, base, base + N_FOE_MONS, N_FOE_MONS)
    _swap(pm, fd + 33, fd + 34)  # cambio forzado por slot

    # σ_f: activos rivales a ↔ b
    _swap(pf, lay['foe_0'][0], lay['foe_1'][0], mb)
    _swap(pf, fm, fm + 4 * FOE_MOVE_DIM, 4 * FOE_MOVE_DIM)
    for k in range(8):  # en cada movimiento propio: features por rival 0 ↔ 1
        o = om + k * MOVE_SLOT_DIM
        for i in (65, 68, 70, 74, 77, 79):
            _swap(pf, o + i, o + i + 1)
    for m in range(3):  # columnas 0 ↔ 1
        base = pr + m * n_cell
        for i in range(N_OUR_MONS):
            _swap(pf, base + i * N_FOE_MONS, base + i * N_FOE_MONS + 1)

    # σ_b: nuestras reservas 1 ↔ 2
    _swap(pb, lay['our_2'][0], lay['our_3'][0], mb)
    for m in range(3):
        base = pr + m * n_cell
        _swap(pb, base + 2 * N_FOE_MONS, base + 3 * N_FOE_MONS, N_FOE_MONS)
    return pm, pf, pb


@lru_cache(maxsize=1)
def obs_perms() -> np.ndarray:
    """int64[8, OBS_DIM]: ``obs' = obs[P[g]]``."""
    pm, pf, pb = _base_perms()
    out = np.zeros((N_ELEMENTS, OBS_DIM), dtype=np.int64)
    for g in range(N_ELEMENTS):
        p = np.arange(OBS_DIM, dtype=np.int64)
        if g & BIT_BENCH:
            p = p[pb]
        if g & BIT_FOE:
            p = p[pf]
        if g & BIT_OUR:
            p = p[pm]
        out[g] = p
    return out


@lru_cache(maxsize=1)
def slot_perms() -> np.ndarray:
    """int64[8, 40]: permutación de ids por slot (σ_f objetivos, σ_b cambios).
    σ_o no cambia ids (aliado y "sí mismo" son relativos al slot); cambia slots."""
    out = np.tile(np.arange(SLOT_ACTION_N, dtype=np.int64), (N_ELEMENTS, 1))
    for g in range(N_ELEMENTS):
        p = out[g]
        if g & BIT_FOE:
            for base in (0, MEGA_BASE):
                for mi in range(N_MOVE_ACTIONS // 4):
                    _swap(p, base + mi * 4 + 0, base + mi * 4 + 1)
        if g & BIT_BENCH:
            _swap(p, SWITCH_BASE, SWITCH_BASE + 1)
    return out


@lru_cache(maxsize=1)
def joint_perms() -> np.ndarray:
    """int64[8, 1600]: ``j' = J[g][j]`` y máscara conjunta ``m' = m[J[g]]``."""
    sp = slot_perms()
    a1 = np.repeat(np.arange(SLOT_ACTION_N), SLOT_ACTION_N)
    a2 = np.tile(np.arange(SLOT_ACTION_N), SLOT_ACTION_N)
    out = np.zeros((N_ELEMENTS, JOINT_ACTION_N), dtype=np.int64)
    for g in range(N_ELEMENTS):
        b1, b2 = sp[g][a1], sp[g][a2]
        if g & BIT_OUR:
            b1, b2 = b2, b1
        out[g] = b1 * SLOT_ACTION_N + b2
    return out


def bench_swap_valid(obs: np.ndarray) -> np.ndarray:
    """bool[N]: ambas reservas presentes y no debilitadas (σ_b aplicable)."""
    from showdown.rl.observation import obs_layout

    lay = obs_layout()
    x = np.asarray(obs).reshape(-1, OBS_DIM)
    ok = np.ones(x.shape[0], dtype=bool)
    for name in ('our_2', 'our_3'):
        d = lay[name][0] + OTS_MON_DIM
        ok &= (x[:, d + 0] > 0.5) & (x[:, d + 2] < 0.5)
    return ok


def sample_elements(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Un elemento aleatorio por fila (σ_b solo donde aplica). Incluye la identidad."""
    n = np.asarray(obs).reshape(-1, OBS_DIM).shape[0]
    g = rng.integers(0, N_ELEMENTS, size=n)
    g[~bench_swap_valid(obs)] &= ~BIT_BENCH
    return g.astype(np.int64)


def augment_obs(obs: np.ndarray, g: np.ndarray) -> np.ndarray:
    x = np.asarray(obs).reshape(-1, OBS_DIM)
    perms = obs_perms()[np.asarray(g, dtype=np.int64)]
    return np.take_along_axis(x, perms, axis=1)


def augment_joint(a: np.ndarray, g: np.ndarray) -> np.ndarray:
    return joint_perms()[np.asarray(g, dtype=np.int64), np.asarray(a, dtype=np.int64)]


def augment_joint_mask(mask: np.ndarray, g: np.ndarray) -> np.ndarray:
    m = np.asarray(mask).reshape(-1, JOINT_ACTION_N)
    return np.take_along_axis(m, joint_perms()[np.asarray(g, dtype=np.int64)], axis=1)


def _is_mega(a: np.ndarray) -> np.ndarray:
    return (a >= MEGA_BASE) & (a < MEGA_BASE + N_MOVE_ACTIONS)


def _is_switch(a: np.ndarray) -> np.ndarray:
    return (a >= SWITCH_BASE) & (a < SWITCH_BASE + 4)


def ppo_symmetry_batch(
    obs: np.ndarray, a1: np.ndarray, a2: np.ndarray, m1: np.ndarray, m2: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Versión permutada de un minibatch PPO (política auto-regresiva).

    σ_o invierte el orden de decisión: el nuevo slot 1 necesita la máscara
    **incondicional** del slot 2, que el buffer no guarda. Es exactamente ``m2``
    cuando ``a1`` no es mega ni cambio (la condición solo quita la mega del otro
    slot y el cambio ya tomado), así que σ_o se aplica solo a esas filas. La
    nueva máscara del slot 2 (= slot 1 original condicionado a ``a2``) se obtiene
    de ``m1`` quitando la mega si ``a2`` megaevoluciona y el cambio que ``a2`` toma.
    """
    a1 = np.asarray(a1, dtype=np.int64).reshape(-1)
    a2 = np.asarray(a2, dtype=np.int64).reshape(-1)
    m1 = np.asarray(m1, dtype=np.float32).reshape(-1, SLOT_ACTION_N)
    m2 = np.asarray(m2, dtype=np.float32).reshape(-1, SLOT_ACTION_N)
    g = sample_elements(obs, rng)
    swap_ok = ~(_is_mega(a1) | _is_switch(a1))
    g[~swap_ok] &= ~BIT_OUR
    sw = (g & BIT_OUR) > 0

    n1, n2 = a1.copy(), a2.copy()
    k1, k2 = m1.copy(), m2.copy()
    if sw.any():
        cond = m1[sw].copy()
        mega2 = _is_mega(a2[sw])
        cond[mega2, MEGA_BASE:MEGA_BASE + N_MOVE_ACTIONS] = 0.0
        rows = np.flatnonzero(_is_switch(a2[sw]))
        cond[rows, a2[sw][rows]] = 0.0
        n1[sw], n2[sw] = a2[sw], a1[sw]
        k1[sw], k2[sw] = m2[sw], cond
    sp = slot_perms()[g]
    rows_all = np.arange(a1.shape[0])
    return {
        'g': g,
        'obs': augment_obs(obs, g),
        'a1': sp[rows_all, n1],
        'a2': sp[rows_all, n2],
        'm1': np.take_along_axis(k1, sp, axis=1),
        'm2': np.take_along_axis(k2, sp, axis=1),
    }


def element_stats(g: np.ndarray) -> Dict[str, float]:
    g = np.asarray(g, dtype=np.int64).reshape(-1)
    if g.size == 0:
        return {}
    return {
        'sym_frac_our': float(((g & BIT_OUR) > 0).mean()),
        'sym_frac_foe': float(((g & BIT_FOE) > 0).mean()),
        'sym_frac_bench': float(((g & BIT_BENCH) > 0).mean()),
    }


def maybe_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()
