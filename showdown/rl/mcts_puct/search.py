"""MCTS simulate / rollout / backup (docs/mcts_puct_spec.md)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from showdown.rl.mcts_puct.greedy_snapshot import greedy_joint_snapshot
from showdown.rl.mcts_puct.simulator import (
    cand1_return,
    flip_perspective,
    sample_uniform_joint,
    step_sim,
)
from showdown.rl.mcts_puct.tree import Node, puct_select, untried_actions
from showdown.rl.snapshot import BattleSnapshot


def make_sim_rng(global_seed: int, turn: int, sim_id: int) -> np.random.Generator:
    """Local RNG: hash(global_seed, turn, sim_id) → SeedSequence."""
    payload = f'{int(global_seed)}|{int(turn)}|{int(sim_id)}'.encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    seed_int = int.from_bytes(digest[:8], 'little')
    return np.random.default_rng(seed_int)


@dataclass
class SearchConfig:
    n_sims: int = 200
    c_puct: float = 1.5
    max_turns: int = 40
    seed: int = 0


@dataclass
class SearchTelemetry:
    sims: int = 0
    nodes: int = 0
    failures: int = 0
    avg_depth: float = 0.0


def rollout(
    snap: BattleSnapshot,
    rng: np.random.Generator,
    *,
    max_turns: int = 40,
) -> float:
    """Both sides greedy_joint_snapshot until terminal; return Cand. 1 z."""
    s = snap
    guard = 0
    while not s.finished and guard < int(max_turns) + 2:
        guard += 1
        a_self = greedy_joint_snapshot(s)
        a_opp = greedy_joint_snapshot(flip_perspective(s))
        s = step_sim(s, a_self, a_opp, rng, max_turns=max_turns)
    return cand1_return(s)


def _backup(path: List[Tuple[Node, int]], z: float) -> None:
    for node, a in path:
        e = node.edges[int(a)]
        e.N += 1
        e.W += float(z)
        node.N_sum += 1


def run_simulation(
    root: Node,
    *,
    sim_id: int,
    cfg: SearchConfig,
    telemetry: Optional[SearchTelemetry] = None,
) -> None:
    rng = make_sim_rng(cfg.seed, int(root.state.turn), int(sim_id))
    node = root
    path: List[Tuple[Node, int]] = []
    depth = 0

    try:
        while True:
            if node.state.finished:
                z = cand1_return(node.state)
                _backup(path, z)
                break

            if not node.expanded:
                node.expand_uniform()

            untried = untried_actions(node)
            if untried:
                a = int(untried[0])
            elif node.edges:
                a = puct_select(node, cfg.c_puct)
            else:
                z = 0.0
                _backup(path, z)
                break

            path.append((node, a))
            depth += 1

            if a not in node.children:
                a_opp = sample_uniform_joint(flip_perspective(node.state), rng)
                child_state = step_sim(
                    node.state, a, a_opp, rng, max_turns=cfg.max_turns,
                )
                child = Node(state=child_state)
                node.children[a] = child
                if telemetry is not None:
                    telemetry.nodes += 1
                if child_state.finished:
                    z = cand1_return(child_state)
                else:
                    z = rollout(child_state, rng, max_turns=cfg.max_turns)
                _backup(path, z)
                break

            node = node.children[a]
        if telemetry is not None:
            telemetry.sims += 1
            n = max(1, telemetry.sims)
            telemetry.avg_depth += (depth - telemetry.avg_depth) / n
    except Exception:
        if telemetry is not None:
            telemetry.failures += 1
            telemetry.sims += 1
        _backup(path, 0.0)


def search(
    root_state: BattleSnapshot,
    cfg: Optional[SearchConfig] = None,
) -> Tuple[int, Node, SearchTelemetry]:
    """Run N_sim simulations; return visit-greedy joint action."""
    cfg = cfg or SearchConfig()
    root = Node(state=root_state)
    root.expand_uniform()
    tel = SearchTelemetry(nodes=1)
    for k in range(int(cfg.n_sims)):
        run_simulation(root, sim_id=k, cfg=cfg, telemetry=tel)

    if not root.edges:
        from showdown.rl.action_space import encode_joint_action
        from showdown.rl.constants import PASS_ID
        return encode_joint_action(PASS_ID, PASS_ID), root, tel

    best_a = max(root.edges.items(), key=lambda kv: (kv[1].N, kv[1].Q))[0]
    return int(best_a), root, tel
