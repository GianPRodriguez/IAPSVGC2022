"""Acceptance tests for MCTS–PUCT/NashSpec (docs/mcts_puct_nash_spec.md)."""
from __future__ import annotations

import numpy as np

from showdown.rl.action_space import legal_joint_mask
from showdown.rl.constants import SPEC_VERSION
from showdown.rl.env import MockBattleBackend, VGCEnv
from showdown.rl.mcts_puct.search import make_sim_rng
from showdown.rl.mcts_puct_nash.nash import fictitious_play, nash_value
from showdown.rl.mcts_puct_nash.search import SearchConfig, search
from showdown.rl.mcts_puct_nash.topk import topk_joints, topk_sets
from showdown.rl.mcts_puct_nash.tree import CellStats, Node, puct_select_cell
from showdown.rl.reward import RewardCandidate
from showdown.rl.snapshot import BattleSnapshot


def test_spec_version_nash_aligned():
    assert SPEC_VERSION == 'v1.6'


def test_cells_only_from_legal_topk():
    env = VGCEnv(backend=MockBattleBackend(max_turns=8), reward_candidate=RewardCandidate.SPARSE)
    env.reset(seed=0)
    assert env._snap is not None
    mask = legal_joint_mask(env._snap)
    I, J = topk_sets(env._snap, k=16)
    assert I.size > 0 and J.size > 0
    for a in I:
        assert mask[int(a)] > 0.5
    joint, root, _tel, sigma = search(env._snap, SearchConfig(n_sims=8, seed=0, k=8))
    assert mask[joint] > 0.5
    for a_i, b_j in root.cells:
        assert mask[int(a_i)] > 0.5
    assert sigma.size == root.I.size
    assert abs(float(sigma.sum()) - 1.0) < 1e-6


def test_matching_pennies_fp():
    A = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    sigma, tau, v = fictitious_play(A, T=500, W=50)
    assert abs(float(sigma[0]) - 0.5) < 0.08
    assert abs(float(sigma[1]) - 0.5) < 0.08
    assert abs(float(tau[0]) - 0.5) < 0.08
    assert abs(float(tau[1]) - 0.5) < 0.08
    assert abs(v) < 0.08
    assert abs(nash_value(A, T=500)) < 0.08


def test_backup_parent_is_nash_value():
    """After updating a child cell, parent backup z becomes v*(Â)."""
    snap = BattleSnapshot(turn=1)
    node = Node(state=snap)
    node.I = np.array([0, 1], dtype=np.int64)
    node.J = np.array([10, 11], dtype=np.int64)
    node.expanded = True
    # Two cells: one visited with z=+1, one unvisited → Â = [[1,0],[0,0]] roughly
    node.cells[(0, 10)] = CellStats(n=0, W=0.0, P=0.25)
    node.cells[(0, 11)] = CellStats(n=0, W=0.0, P=0.25)
    node.cells[(1, 10)] = CellStats(n=0, W=0.0, P=0.25)
    node.cells[(1, 11)] = CellStats(n=0, W=0.0, P=0.25)

    from showdown.rl.mcts_puct_nash.search import _backup_nash

    path = [(node, (0, 10))]
    _backup_nash(path, z=1.0, t_fp=200)
    assert node.cells[(0, 10)].n == 1
    assert node.cells[(0, 10)].W == 1.0
    A = node.payoff_matrix()
    _s, _t, v = fictitious_play(A, T=200)
    # After one backup, N_sum=1 and Â has a single +1 entry → v* ≈ that game value
    assert abs(v - nash_value(A, T=200)) < 1e-9
    assert -1.0 <= v <= 1.0


def test_rng_reproducible_per_sim_id():
    env = VGCEnv(backend=MockBattleBackend(max_turns=8), reward_candidate=RewardCandidate.SPARSE)
    env.reset(seed=0)
    snap = env._snap
    assert snap is not None
    j1, _r1, t1, s1 = search(snap, SearchConfig(n_sims=8, seed=7, k=8))
    j2, _r2, t2, s2 = search(snap, SearchConfig(n_sims=8, seed=7, k=8))
    assert j1 == j2
    assert t1.sims == t2.sims == 8
    np.testing.assert_allclose(s1, s2)
    r1 = make_sim_rng(7, int(snap.turn), 3)
    r2 = make_sim_rng(7, int(snap.turn), 3)
    assert r1.integers(0, 10**9) == r2.integers(0, 10**9)


def test_puct_prefer_unvisited():
    snap = BattleSnapshot(turn=1)
    node = Node(state=snap)
    node.expanded = True
    node.N_sum = 10
    node.cells[(0, 0)] = CellStats(n=5, W=2.5, P=0.5)  # Q=0.5
    node.cells[(1, 1)] = CellStats(n=0, W=0.0, P=0.5)   # unvisited → high U
    key = puct_select_cell(node, c_puct=1.5)
    assert key == (1, 1)


def test_topk_size_and_tie_break():
    env = VGCEnv(backend=MockBattleBackend(max_turns=8), reward_candidate=RewardCandidate.SPARSE)
    env.reset(seed=3)
    assert env._snap is not None
    I = topk_joints(env._snap, k=4)
    assert 1 <= I.size <= 4
    # Sorted by score desc then id asc — ids unique
    assert len(set(int(x) for x in I)) == I.size


def test_smoke_agent_mock_game():
    from showdown.rl.mcts_puct_nash.agent import MCTSPuctNashAgent, MCTSPuctNashConfig

    agent = MCTSPuctNashAgent(MCTSPuctNashConfig(n_sims=8, seed=0, k=8))
    env = VGCEnv(
        backend=MockBattleBackend(max_turns=40),
        reward_candidate=RewardCandidate.SPARSE,
        max_episode_turns=40,
    )
    env.reset(seed=0)
    steps = 0
    try:
        while steps < 60:
            assert env._snap is not None
            a1, a2 = agent.act_slots(env._snap)
            _o, _r, term, trunc, _i = env.step((a1, a2))
            steps += 1
            if term or trunc:
                break
    finally:
        env.close()
    assert steps >= 1
    assert agent.last_telemetry is not None
    assert agent.last_telemetry.sims == 8
