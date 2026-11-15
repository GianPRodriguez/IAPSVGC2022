"""Diversidad de juego de los baselines: muestreo entre casi-empates y traza por decisión.

Los tres heurísticos (solo, joint, expectimax) son deterministas: ante el mismo estado
juegan siempre lo mismo. Para que sirvan como rivales de entrenamiento "diversos pero
buenos" se puede activar un muestreo uniforme entre las jugadas cuyo valor queda a menos
de ``margin`` del mejor (0 = determinista, el comportamiento por defecto).

``PS_DIVERSITY_LOG=<ruta>`` escribe una línea JSON por decisión en ``<ruta>.<pid>``;
``python -m showdown.inference_sim.analyze_diversity <ruta>`` las resume.
``PS_TIE_SEED=<n>`` fija la semilla del muestreo (por defecto, aleatoria por proceso).
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar('T')

_seed = os.environ.get('PS_TIE_SEED', '').strip()
TIE_RNG = random.Random(int(_seed) if _seed.lstrip('-').isdigit() else None)


def sample_near_ties(scored: Sequence[Tuple[float, T]], margin: float,
                     rng: Optional[random.Random] = None) -> Tuple[Optional[T], int]:
    """Elige al azar entre los elementos con puntaje >= mejor − margin.

    Devuelve ``(elegido, n_candidatos)``. Con ``margin <= 0`` o un solo candidato,
    devuelve el primer máximo (el mismo que el argmax determinista).
    """
    if not scored:
        return None, 0
    best_i = 0
    for i, (s, _x) in enumerate(scored):
        if s > scored[best_i][0]:
            best_i = i
    best_s, best_x = scored[best_i]
    if margin <= 0:
        return best_x, 1
    cands: List[T] = [x for s, x in scored if s >= best_s - margin]
    if len(cands) <= 1:
        return best_x, 1
    return (rng or TIE_RNG).choice(cands), len(cands)


def enabled() -> bool:
    return bool(os.environ.get('PS_DIVERSITY_LOG', '').strip())


def write(rec: Dict[str, Any]) -> None:
    """Agrega una decisión a ``PS_DIVERSITY_LOG.<pid>`` (no-op sin la variable)."""
    path = os.environ.get('PS_DIVERSITY_LOG', '').strip()
    if not path:
        return
    try:
        with open(f'{path}.{os.getpid()}', 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec) + chr(10))
    except OSError:
        pass
