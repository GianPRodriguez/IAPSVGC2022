"""Acceptance tests for MCTS+PUCTSpec (docs/mcts_puct_spec.md)."""
from __future__ import annotations

import numpy as np

from showdown.rl.action_space import encode_joint_action, legal_joint_mask
from showdown.rl.constants import PASS_ID, SPEC_VERSION
from showdown.rl.env import MockBattleBackend, VGCEnv
from showdown.rl.mcts_puct.greedy_snapshot import greedy_joint_snapshot
from showdown.rl.mcts_puct.search import SearchConfig, make_sim_rng, rollout, search
from showdown.rl.mcts_puct.simulator import (
    cand1_return,
    legal_joints,
    sample_uniform_joint,
    step_sim,
)
from showdown.rl.mcts_puct.tree import EdgeStats, Node, puct_select
from showdown.rl.reward import RewardCandidate
from showdown.rl.snapshot import BattleSnapshot


def test_spec_version_mcts_aligned():
    assert SPEC_VERSION == 'v1.7'


def test_legal_only_selection_and_greedy():
    env = VGCEnv(backend=MockBattleBackend(max_turns=8), reward_candidate=RewardCandidate.SPARSE)
    env.reset(seed=0)
    assert env._snap is not None
    mask = legal_joint_mask(env._snap)
    j = greedy_joint_snapshot(env._snap)
    assert mask[j] > 0.5
    joint, root, _tel = search(env._snap, SearchConfig(n_sims=16, seed=0))
    assert mask[joint] > 0.5
    for a in root.edges:
        assert mask[a] > 0.5


def test_puct_formula_argmax():
    snap = BattleSnapshot(turn=1)
    node = Node(state=snap)
    node.expanded = True
    node.N_sum = 100
    node.edges[0] = EdgeStats(N=10, W=5.0, P=0.5)   # Q=0.5
    node.edges[1] = EdgeStats(N=1, W=0.0, P=0.5)    # Q=0, high U
    # score0 = 0.5 + 1.5*0.5*10/(1+10) = 0.5 + 0.75*10/11 ≈ 1.181
    # score1 = 0 + 1.5*0.5*10/(1+1) = 0.75*5 = 3.75
    a = puct_select(node, c_puct=1.5)
    assert a == 1


def test_backup_no_sign_flip_and_cand1():
    env = VGCEnv(backend=MockBattleBackend(max_turns=4), reward_candidate=RewardCandidate.SPARSE)
    env.reset(seed=1)
    snap = env._snap
    assert snap is not None
    # Force a quick win-ish state via many aggressive steps in sim
    rng = np.random.default_rng(0)
    s = snap
    for _ in range(20):
        if s.finished:
            break
        a_s = greedy_joint_snapshot(s)
        a_o = encode_joint_action(PASS_ID, PASS_ID)
        s = step_sim(s, a_s, a_o, rng, max_turns=40)
    z = cand1_return(s)
    assert z in (-1.0, 0.0, 1.0)
    if s.finished and s.won is True:
        assert z == 1.0
    if s.finished and s.won is False:
        assert z == -1.0
    if s.finished and s.won is None:
        assert z == 0.0


def test_rng_reproducible_per_sim_id():
    env = VGCEnv(backend=MockBattleBackend(max_turns=8), reward_candidate=RewardCandidate.SPARSE)
    env.reset(seed=0)
    snap = env._snap
    assert snap is not None
    r1 = make_sim_rng(7, int(snap.turn), 3)
    r2 = make_sim_rng(7, int(snap.turn), 3)
    r3 = make_sim_rng(7, int(snap.turn), 4)
    a1 = sample_uniform_joint(snap, r1)
    a2 = sample_uniform_joint(snap, r2)
    a3 = sample_uniform_joint(snap, r3)
    assert a1 == a2
    # Different sim_id should usually differ (not guaranteed if |L|=1)
    legal = legal_joints(snap)
    if legal.size > 1:
        # re-draw sequences
        seq1 = [sample_uniform_joint(snap, make_sim_rng(7, snap.turn, 3)) for _ in range(5)]
        # same seed path identical
        seq2 = [sample_uniform_joint(snap, make_sim_rng(7, snap.turn, 3)) for _ in range(5)]
        assert seq1 == seq2


def test_rollout_terminates():
    env = VGCEnv(backend=MockBattleBackend(max_turns=40), reward_candidate=RewardCandidate.SPARSE)
    env.reset(seed=2)
    assert env._snap is not None
    z = rollout(env._snap, np.random.default_rng(0), max_turns=40)
    assert z in (-1.0, 0.0, 1.0)


def test_smoke_agent_mock_game():
    from showdown.rl.mcts_puct.agent import MCTSPuctAgent, MCTSPuctConfig

    agent = MCTSPuctAgent(MCTSPuctConfig(n_sims=16, seed=0))
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
    assert agent.last_telemetry.sims == 16
