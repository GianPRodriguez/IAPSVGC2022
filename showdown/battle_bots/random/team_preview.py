"""Team Preview del bot Random — muestreo uniforme 4 de 6.

Contrato: docs/random_spec.md §3.
"""
from __future__ import annotations

import random
from typing import Optional, Sequence, Tuple

# Congelado: 4 índices distintos en {1..6}; orden de la muestra = orden enviado.
PREVIEW_PICK_COUNT: int = 4
PREVIEW_SLOT_LO: int = 1
PREVIEW_SLOT_HI: int = 6  # inclusive via range(lo, hi+1)


def sample_team_preview_order(
    rng: Optional[random.Random] = None,
) -> Tuple[int, ...]:
    """
    Devuelve 4 índices 1-based distintos uniformes.

    Leads = posiciones 0 y 1 de la tupla; banca = 2 y 3.
    Sin tipado / BST / roles.
    """
    r = rng if rng is not None else random
    return tuple(r.sample(range(PREVIEW_SLOT_LO, PREVIEW_SLOT_HI + 1), PREVIEW_PICK_COUNT))


def is_valid_preview_order(order: Sequence[int]) -> bool:
    if len(order) != PREVIEW_PICK_COUNT:
        return False
    s = set(int(x) for x in order)
    if len(s) != PREVIEW_PICK_COUNT:
        return False
    return all(PREVIEW_SLOT_LO <= i <= PREVIEW_SLOT_HI for i in s)
