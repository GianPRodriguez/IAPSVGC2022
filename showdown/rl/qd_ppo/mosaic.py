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
