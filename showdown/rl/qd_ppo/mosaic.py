"""Static heuristic foe mosaic 25/25/50 (docs/qd_ppo_spec.md)."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

MOSAIC_FOES: Tuple[str, ...] = ('random', 'greedy_solo', 'greedy_joint')
MOSAIC_PROBS: Tuple[float, ...] = (0.25, 0.25, 0.50)


def sample_mosaic_foe(rng: np.random.Generator) -> str:
    idx = int(rng.choice(len(MOSAIC_FOES), p=np.asarray(MOSAIC_PROBS, dtype=np.float64)))
    return MOSAIC_FOES[idx]


def mosaic_weights() -> dict:
    return {f: float(p) for f, p in zip(MOSAIC_FOES, MOSAIC_PROBS)}


# I 2.0 (25-11-2026): mosaico con las proporciones de las fases de PPO v2 (random 150k, solo 250k,
# joint 300k, expectimax 300k de 1M), para que continuar desde PPO v2 no olvide a expectimax.
MOSAICS = {
    'v1': (MOSAIC_FOES, MOSAIC_PROBS),
    'ppo_v2': (('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian'), (0.15, 0.25, 0.30, 0.30)),
    # Selección del desplegado y validación: los tres heurísticos por igual (como el val de PPO v2).
    'heur': (('greedy_solo', 'greedy_joint', 'expectimax_bayesian'), (1 / 3, 1 / 3, 1 / 3)),
}


def make_sampler(name: str):
    """Muestreador ``rng -> rival`` del mosaico ``name`` (ver ``MOSAICS``)."""
    foes, probs = MOSAICS[str(name)]
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()

    def _sample(rng: np.random.Generator) -> str:
        return foes[int(rng.choice(len(foes), p=p))]

    return _sample

