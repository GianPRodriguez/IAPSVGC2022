"""Top-K joint sets I(s), J(s) (docs/mcts_puct_nash_spec.md)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from showdown.rl.mcts_puct.greedy_snapshot import score_joint
from showdown.rl.mcts_puct.simulator import flip_perspective, legal_joints
from showdown.rl.snapshot import BattleSnapshot

DEFAULT_K = 16


def topk_joints(snap: BattleSnapshot, k: int = DEFAULT_K) -> np.ndarray:
    """
    Top-K legal joints by score_joint; ties → smaller joint id.

    If |L| < K, return all legal joints.
    """
    legal = legal_joints(snap)
    if legal.size == 0:
        return legal
    k = max(1, int(k))
    ranked: List[Tuple[float, int]] = []
    for j in legal:
        ranked.append((float(score_joint(snap, int(j))), int(j)))
    ranked.sort(key=lambda t: (-t[0], t[1]))
    out = [j for _s, j in ranked[:k]]
    return np.asarray(out, dtype=np.int64)


def topk_sets(
    snap: BattleSnapshot,
    k: int = DEFAULT_K,
) -> Tuple[np.ndarray, np.ndarray]:
    """I(s) from self snapshot; J(s) from flipped (opp) snapshot."""
    I = topk_joints(snap, k=k)
    J = topk_joints(flip_perspective(snap), k=k)
    return I, J
