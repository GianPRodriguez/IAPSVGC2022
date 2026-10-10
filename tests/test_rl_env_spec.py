"""Unit tests for Spec v1.4 RL package (offline, no PS)."""
from __future__ import annotations

import numpy as np

from showdown.rl.action_space import (
    SLOT_ACTION_N,
    apply_logit_mask,
    decode_slot_action,
    encode_slot_action,
    resolve_action_or_fallback,
    softmax_masked,
)
from showdown.rl.behavioral import BehavioralTracker, classify_action
from showdown.rl.constants import OBS_DIM, PASS_ID, SPEC_VERSION
from showdown.rl.env import MockBattleBackend, VGCEnv
from showdown.rl.observation import build_observation
from showdown.rl.reward import RewardCandidate, compute_reward
from showdown.rl.snapshot import (
    BattleSnapshot,
    MonSnapshot,
    SideSnapshot,
    SlotAction,
    TeamMaterial,
)


def test_spec_version_and_obs_dim():
    assert SPEC_VERSION == 'v1.4'
    assert OBS_DIM == 1024


def test_reward_sparse_terminal():
    z = TeamMaterial(0, 400.0)
    r = compute_reward(
        candidate=RewardCandidate.SPARSE,
        prev_our=z, prev_foe=z, cur_our=z, cur_foe=z,
        finished=True, won=True, tied=False,
    )
    assert r.reward == 1.0
    r2 = compute_reward(
        candidate=RewardCandidate.SPARSE,
        prev_our=z, prev_foe=z, cur_our=z, cur_foe=z,
        finished=True, won=False, tied=False,
    )
    assert r2.reward == -1.0
    r3 = compute_reward(
        candidate=RewardCandidate.SPARSE,
        prev_our=z, prev_foe=z, cur_our=z, cur_foe=z,
        finished=True, won=None, tied=True,
    )
    assert r3.reward == 0.0


def test_reward_dense_faint_and_tie():
    prev_foe = TeamMaterial(faint_count=0, hp_pct_total=400.0)
    cur_foe = TeamMaterial(faint_count=1, hp_pct_total=300.0)
    our = TeamMaterial(0, 400.0)
    r = compute_reward(
        candidate=RewardCandidate.DENSE,
        prev_our=our, prev_foe=prev_foe, cur_our=our, cur_foe=cur_foe,
        finished=False, won=None, tied=False,
    )
    # 0.15 faint + 0.05 * (100/100) = 0.15 + 0.05 = 0.20
    assert abs(r.reward - 0.20) < 1e-6

    tie = compute_reward(
        candidate=RewardCandidate.DENSE,
        prev_our=our, prev_foe=our, cur_our=our, cur_foe=our,
        finished=True, won=None, tied=True,
    )
    assert abs(tie.reward - (-0.25)) < 1e-6


def test_reward_skip_buffer_zero():
    z = TeamMaterial(0, 0.0)
    r = compute_reward(
        candidate=RewardCandidate.DENSE,
        prev_our=z, prev_foe=z, cur_our=z, cur_foe=z,
        finished=True, won=True, tied=False, skip_buffer=True,
    )
    assert r.reward == 0.0
    assert r.skip_buffer is True


def test_action_encode_decode_roundtrip():
    aid = encode_slot_action(kind='move', move_index=2, target=1)
    dec = decode_slot_action(aid)
    assert dec.kind == 'move'
    assert dec.move_index == 2
    assert dec.target == 1
    sw = encode_slot_action(kind='switch', switch_index=3)
    assert decode_slot_action(sw).switch_index == 3


def test_logit_mask_and_fallback():
    logits = np.zeros(SLOT_ACTION_N)
    logits[3] = 5.0
    mask = np.zeros(SLOT_ACTION_N)
    mask[PASS_ID] = 1.0
    masked = apply_logit_mask(logits, mask)
    assert masked[3] < -1e8
    probs = softmax_masked(logits, mask)
    assert abs(probs[PASS_ID] - 1.0) < 1e-6
    resolved, fb = resolve_action_or_fallback(3, mask)
    assert fb is True
    assert resolved == PASS_ID


def test_bd_hard_switch_vs_uturn():
    assert classify_action(SlotAction('switch'), force_switch=False) == 'hard_switch'
    assert classify_action(SlotAction('switch'), force_switch=True) == 'ignore'
    assert classify_action(SlotAction('move', move_index=0), 'uturn') == 'offensive'
    assert classify_action(SlotAction('move'), 'protect') == 'status_setup'
    tr = BehavioralTracker()
    tr.on_turn_actions(
        [SlotAction('move'), SlotAction('switch')],
        ['flareblitz', ''],
        force_switch=(False, False),
    )
    bd1, bd2 = tr.bd_vector()
    assert bd2 > 0
    assert 0.0 <= bd1 <= 1.0


def test_observation_shape():
    backend = MockBattleBackend()
    snap = backend.reset(seed=1)
    obs = build_observation(snap)
    assert obs.shape == (1024,)
    assert obs.dtype == np.float32


def test_vgcenv_episode_smoke():
    env = VGCEnv(reward_candidate=RewardCandidate.DENSE)
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBS_DIM,)
    assert info['spec_version'] == 'v1.4'
    assert info['action_mask_slot1'].shape == (SLOT_ACTION_N,)
    done = False
    trunc = False
    steps = 0
    while not (done or trunc) and steps < 20:
        # Always pick legal: PASS, PASS or first legal move
        m1 = info['action_mask_slot1']
        a0 = int(np.flatnonzero(m1 > 0.5)[0])
        a1 = PASS_ID
        obs, reward, done, trunc, info = env.step([a0, a1])
        assert isinstance(reward, float)
        if info.get('skip_buffer'):
            # L2 must zero reward
            assert reward == 0.0
        steps += 1
    assert steps >= 1
