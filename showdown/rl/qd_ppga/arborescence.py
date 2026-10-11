"""Arborescence parent selection α=0.5 (docs/qd_ppga_spec.md)."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from showdown.rl.qd_ppo.archive import Archive

ALPHA_DEFAULT = 0.5


def _bin_center(i: int, j: int, n1: int, n2: int) -> Tuple[float, float]:
    return ((float(i) + 0.5) / float(n1), (float(j) + 0.5) / float(n2))


def _l1(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def pick_frontier_target(
    archive: Archive,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """
    c*: uniform among empty cells if any; else filled cell with minimum f.
    """
    filled = set(archive.filled_keys())
    empty: List[Tuple[int, int]] = [
        (i, j)
        for i in range(archive.n1)
        for j in range(archive.n2)
        if (i, j) not in filled
    ]
    if empty:
        return empty[int(rng.integers(0, len(empty)))]
    # All filled → lowest fitness cell
    best_key = None
    best_f = float('inf')
    for key, elite in archive.cells.items():
        if elite.fitness < best_f:
            best_f = float(elite.fitness)
            best_key = key
    assert best_key is not None
    return best_key


def nearest_filled_parent(
    archive: Archive,
    c_star: Tuple[int, int],
) -> Tuple[Dict[str, Any], Tuple[int, int]]:
    """Parent = filled elite minimizing L1(BD, center(c*))."""
    target = _bin_center(c_star[0], c_star[1], archive.n1, archive.n2)
    best_key = None
    best_d = float('inf')
    for key, elite in archive.cells.items():
        d = _l1(elite.bd, target)
        if d < best_d:
            best_d = d
            best_key = key
    assert best_key is not None
    return copy.deepcopy(archive.cells[best_key].state_dict), best_key


def sample_arborescence_parent(
    archive: Archive,
    rng: np.random.Generator,
    *,
    alpha: float = ALPHA_DEFAULT,
    fallback_state: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[int, int]], str]:
    """
    Returns (actor_state_dict, parent_key, mode) where mode in
    {'empty','uniform','frontier'}.
    """
    keys = archive.filled_keys()
    if not keys:
        sd = copy.deepcopy(fallback_state) if fallback_state else None
        return sd, None, 'empty'
    if float(rng.random()) < float(alpha):
        key = keys[int(rng.integers(0, len(keys)))]
        return copy.deepcopy(archive.cells[key].state_dict), key, 'uniform'
    c_star = pick_frontier_target(archive, rng)
    sd, key = nearest_filled_parent(archive, c_star)
    return sd, key, 'frontier'
