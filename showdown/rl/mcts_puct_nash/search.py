"""SM-MCTS B1: cell PUCT + Nash backup (docs/mcts_puct_nash_spec.md)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from showdown.rl.mcts_puct.search import make_sim_rng, rollout
from showdown.rl.mcts_puct.simulator import cand1_return, step_sim
from showdown.rl.mcts_puct_nash.nash import fictitious_play
from showdown.rl.mcts_puct_nash.topk import DEFAULT_K
from showdown.rl.mcts_puct_nash.tree import (
    CellKey,
    Node,
    puct_select_cell,
    untried_cells,
)
from showdown.rl.snapshot import BattleSnapshot


@dataclass
class SearchConfig:
    n_sims: int = 50
    c_puct: float = 1.5
    k: int = DEFAULT_K
    max_turns: int = 40
    seed: int = 0
    t_fp_internal: int = 200
    t_fp_root: int = 500


@dataclass
class SearchTelemetry:
    sims: int = 0
    nodes: int = 0
    failures: int = 0
    avg_depth: float = 0.0


def _backup_nash(
    path: List[Tuple[Node, CellKey]],
    z: float,
    *,
    t_fp: int,
) -> None:
    """
    Deepest cell gets leaf z; then each ancestor gets v*(Â) from below.

    For each edge in reverse: n+=1, W+=z; then z ← NashValue(Â).
    """
    for node, key in reversed(path):
        cell = node.cells[key]
        cell.n += 1
        cell.W += float(z)
        node.N_sum += 1
        A = node.payoff_matrix()
        _sigma, _tau, v = fictitious_play(A, T=int(t_fp))
        z = float(v)


def run_simulation(
    root: Node,
    *,
    sim_id: int,
    cfg: SearchConfig,
    telemetry: Optional[SearchTelemetry] = None,
) -> None:
    rng = make_sim_rng(cfg.seed, int(root.state.turn), int(sim_id))
    node = root
    path: List[Tuple[Node, CellKey]] = []
    depth = 0

    try:
        while True:
            if node.state.finished:
                z = cand1_return(node.state)
                _backup_nash(path, z, t_fp=cfg.t_fp_internal)
                break

            if not node.expanded:
                node.expand_topk(k=cfg.k)

            untried = untried_cells(node)
            if untried:
                key = untried[0]
            elif node.cells:
                key = puct_select_cell(node, cfg.c_puct)
            else:
                z = 0.0
                _backup_nash(path, z, t_fp=cfg.t_fp_internal)
                break

            path.append((node, key))
            depth += 1
            a_self, a_opp = int(key[0]), int(key[1])

            if key not in node.children:
                child_state = step_sim(
                    node.state, a_self, a_opp, rng, max_turns=cfg.max_turns,
                )
                child = Node(state=child_state)
                node.children[key] = child
                if telemetry is not None:
                    telemetry.nodes += 1
                if child_state.finished:
                    z = cand1_return(child_state)
                else:
                    z = rollout(child_state, rng, max_turns=cfg.max_turns)
                _backup_nash(path, z, t_fp=cfg.t_fp_internal)
                break

            node = node.children[key]

        if telemetry is not None:
            telemetry.sims += 1
            n = max(1, telemetry.sims)
            telemetry.avg_depth += (depth - telemetry.avg_depth) / n
    except Exception:
        if telemetry is not None:
            telemetry.failures += 1
            telemetry.sims += 1
        _backup_nash(path, 0.0, t_fp=cfg.t_fp_internal)


def search(
    root_state: BattleSnapshot,
    cfg: Optional[SearchConfig] = None,
) -> Tuple[int, Node, SearchTelemetry, np.ndarray]:
    """
    Run N_sim simulations; return (a*, root, tel, σ*_root).

    Play: a* = argmax_i σ*_root with T_fp = t_fp_root.
    """
    cfg = cfg or SearchConfig()
    root = Node(state=root_state)
    root.expand_topk(k=cfg.k)
    tel = SearchTelemetry(nodes=1)

    for k in range(int(cfg.n_sims)):
        run_simulation(root, sim_id=k, cfg=cfg, telemetry=tel)

    if root.I.size == 0:
        from showdown.rl.action_space import encode_joint_action
        from showdown.rl.constants import PASS_ID
        empty = np.zeros(0, dtype=np.float64)
        return encode_joint_action(PASS_ID, PASS_ID), root, tel, empty

    A = root.payoff_matrix()
    sigma, _tau, _v = fictitious_play(A, T=int(cfg.t_fp_root))
    # argmax_i σ*; ties → smaller joint id (I already ascending on ties from topk)
    best_i = int(np.argmax(sigma))
    # Stable tie-break: among max σ, smallest joint id
    max_s = float(sigma[best_i])
    candidates = [
        i for i in range(int(root.I.size))
        if abs(float(sigma[i]) - max_s) <= 1e-12
    ]
    best_i = min(candidates, key=lambda i: int(root.I[i]))
    return int(root.I[best_i]), root, tel, sigma
