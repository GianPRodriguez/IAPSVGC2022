"""SM-MCTS nodes / cell PUCT (docs/mcts_puct_nash_spec.md)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from showdown.rl.mcts_puct_nash.topk import DEFAULT_K, topk_sets
from showdown.rl.snapshot import BattleSnapshot

CellKey = Tuple[int, int]  # (a_self_joint, a_opp_joint)


@dataclass
class CellStats:
    n: int = 0
    W: float = 0.0
    P: float = 0.0

    @property
    def Q(self) -> float:
        """Online Â_ij = W/n (0 if unvisited)."""
        return 0.0 if self.n <= 0 else float(self.W) / float(self.n)


@dataclass
class Node:
    state: BattleSnapshot
    I: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    J: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    cells: Dict[CellKey, CellStats] = field(default_factory=dict)
    children: Dict[CellKey, 'Node'] = field(default_factory=dict)
    expanded: bool = False
    N_sum: int = 0

    def expand_topk(self, k: int = DEFAULT_K) -> None:
        if self.expanded:
            return
        I, J = topk_sets(self.state, k=k)
        self.I = I
        self.J = J
        n_cells = int(I.size) * int(J.size)
        if n_cells <= 0:
            self.expanded = True
            return
        p = 1.0 / float(n_cells)
        for a_i in I:
            for b_j in J:
                self.cells[(int(a_i), int(b_j))] = CellStats(n=0, W=0.0, P=p)
        self.expanded = True

    def payoff_matrix(self) -> np.ndarray:
        """Â with rows = I, cols = J; Â_ij = Q of cell (I[i], J[j])."""
        n = int(self.I.size)
        m = int(self.J.size)
        A = np.zeros((n, m), dtype=np.float64)
        for i, a_i in enumerate(self.I):
            for j, b_j in enumerate(self.J):
                e = self.cells.get((int(a_i), int(b_j)))
                A[i, j] = 0.0 if e is None else e.Q
        return A


def untried_cells(node: Node) -> List[CellKey]:
    """Cells with n=0 not yet expanded to children (prefer exploration)."""
    out: List[CellKey] = []
    for key, e in node.cells.items():
        if e.n == 0:
            out.append(key)
    # Stable: smaller (a_i, b_j)
    out.sort()
    return out


def puct_select_cell(node: Node, c_puct: float) -> CellKey:
    """Argmax_{(i,j)} Q + c * P * sqrt(N_sum) / (1+n)."""
    if not node.cells:
        raise ValueError('PUCT select on empty cell set')
    best_key: Optional[CellKey] = None
    best_score = -1e30
    sqrt_n = float(np.sqrt(max(1, node.N_sum)))
    c = float(c_puct)
    for key, e in node.cells.items():
        u = e.Q + c * e.P * sqrt_n / (1.0 + float(e.n))
        if u > best_score + 1e-15:
            best_score = u
            best_key = key
        elif abs(u - best_score) <= 1e-15 and best_key is not None:
            if key < best_key:
                best_key = key
    assert best_key is not None
    return best_key
