"""MCTS nodes / PUCT selection (docs/mcts_puct_spec.md)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from showdown.rl.mcts_puct.simulator import legal_joints
from showdown.rl.snapshot import BattleSnapshot


@dataclass
class EdgeStats:
    N: int = 0
    W: float = 0.0
    P: float = 0.0

    @property
    def Q(self) -> float:
        return 0.0 if self.N <= 0 else float(self.W) / float(self.N)


@dataclass
class Node:
    state: BattleSnapshot
    edges: Dict[int, EdgeStats] = field(default_factory=dict)
    children: Dict[int, 'Node'] = field(default_factory=dict)
    expanded: bool = False
    N_sum: int = 0

    def expand_uniform(self) -> None:
        if self.expanded:
            return
        legal = legal_joints(self.state)
        if legal.size == 0:
            self.expanded = True
            return
        p = 1.0 / float(legal.size)
        for a in legal:
            self.edges[int(a)] = EdgeStats(N=0, W=0.0, P=p)
        self.expanded = True


def puct_select(node: Node, c_puct: float) -> int:
    """Argmax_a Q + c_puct * P * sqrt(N_sum) / (1+N)."""
    if not node.edges:
        raise ValueError('PUCT select on empty edge set')
    best_a = None
    best_score = -1e30
    sqrt_n = float(np.sqrt(max(1, node.N_sum)))
    c = float(c_puct)
    for a, e in node.edges.items():
        u = e.Q + c * e.P * sqrt_n / (1.0 + float(e.N))
        if u > best_score:
            best_score = u
            best_a = int(a)
    assert best_a is not None
    return best_a


def untried_actions(node: Node) -> List[int]:
    return [a for a, e in node.edges.items() if e.N == 0 and a not in node.children]
