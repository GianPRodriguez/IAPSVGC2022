"""Borrado aleatorio de los Pokémon rivales no activos en la obs (memorizacion_equipos.md §8, b).

El análisis de §7 mostró que DDQN responde a la combinación completa del equipo rival (gana 60 %
contra equipos `eval` cuyos 6 sets conoce, frente a 88 % con sus equipos). Al aprender, cada uno
de los 4 Pokémon rivales no activos (bloques ``foe_2``–``foe_5`` y sus entradas en ``pairs``) se
pone a cero —la codificación de "ausente" de ``_mon_block(None)`` y ``_pair_block``— con
probabilidad ``p``, de forma independiente por muestra y por Pokémon. Así la red no puede usar la
hoja completa del rival como clave y tiene que decidir sobre todo con lo que está en juego.

Solo se borra la obs ``s`` del lote (la entrada de la regresión); ``s'`` y la inferencia ven la
obs completa. El sorteo usa el generador NumPy del agente (mismos lotes en CPU y GPU).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

from showdown.rl.constants import N_FOE_MONS, N_OUR_MONS

BENCH_SLOTS = tuple(range(2, N_FOE_MONS))  # foe_0 y foe_1 son los activos


@lru_cache(maxsize=1)
def bench_columns() -> Tuple[np.ndarray, ...]:
    """Columnas de la obs de cada Pokémon rival no activo (bloque + pares)."""
    from showdown.rl.observation import obs_layout
    lay = obs_layout()
    p0 = lay['pairs'][0]
    out = []
    for j in BENCH_SLOTS:
        a, b = lay[f'foe_{j}']
        cols = list(range(a, b))
        for k in range(3):  # daño us→foe, foe→us, velocidad: índice k*24 + i*6 + j
            for i in range(N_OUR_MONS):
                cols.append(p0 + k * N_OUR_MONS * N_FOE_MONS + i * N_FOE_MONS + j)
        out.append(np.asarray(cols, dtype=np.int64))
    return tuple(out)


def sample_drop(n: int, p: float, rng) -> np.ndarray:
    """(n, 4) booleano: qué Pokémon no activos se borran en cada muestra."""
    return rng.random((n, len(BENCH_SLOTS))) < float(p)


def foe_bench_dropout(s: np.ndarray, p: float, rng) -> Tuple[np.ndarray, float]:
    drop = sample_drop(s.shape[0], p, rng)
    out = s.copy()
    for k, cols in enumerate(bench_columns()):
        rows = np.flatnonzero(drop[:, k])
        if rows.size:
            out[np.ix_(rows, cols)] = 0.0
    return out, float(drop.mean())


def foe_bench_dropout_t(s, p: float, rng):
    """Versión torch (mismo sorteo que la NumPy)."""
    import torch
    drop = sample_drop(int(s.shape[0]), p, rng)
    keep = torch.ones_like(s)
    for k, cols in enumerate(bench_columns()):
        rows = np.flatnonzero(drop[:, k])
        if rows.size:
            r = torch.as_tensor(rows, device=s.device)
            c = torch.as_tensor(cols, device=s.device)
            keep[r.view(-1, 1), c.view(1, -1)] = 0.0
    return s * keep, float(drop.mean())
