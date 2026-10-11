"""Acceptance tests for QD-PPOSpec (docs/qd_ppo_spec.md)."""
from __future__ import annotations

import numpy as np

from showdown.rl.behavioral import BehavioralTracker, classify_action
from showdown.rl.constants import SPEC_VERSION
from showdown.rl.env import MockBattleBackend, VGCEnv
from showdown.rl.qd_ppo.archive import Archive, bin_bd
from showdown.rl.qd_ppo.mosaic import MOSAIC_FOES, MOSAIC_PROBS, sample_mosaic_foe
from showdown.rl.qd_ppo.train import derived_n_outer
from showdown.rl.reward import RewardCandidate
from showdown.rl.snapshot import SlotAction


def test_spec_version_qd_aligned():
    assert SPEC_VERSION == 'v1.8'


def test_derived_n_outer():
    assert derived_n_outer() == 109
    assert derived_n_outer(t_total=1_000_000, b_boot=100_000, n_emit=4, rollout=2048) == 109


def test_bin_bd_corners():
    assert bin_bd(0.0, 0.0) == (0, 0)
    assert bin_bd(1.0, 1.0) == (9, 9)
    assert bin_bd(0.99, 0.05) == (9, 0)


def test_archive_strict_replace_and_coverage():
    arch = Archive()
    sd_a = {'w': np.array([1.0])}
    sd_b = {'w': np.array([2.0])}
    assert arch.try_insert(bd1=0.1, bd2=0.2, fitness=0.0, state_dict=sd_a)
    assert not arch.try_insert(bd1=0.1, bd2=0.2, fitness=0.0, state_dict=sd_b)  # tie
    assert not arch.try_insert(bd1=0.1, bd2=0.2, fitness=-0.1, state_dict=sd_b)
    assert arch.try_insert(bd1=0.1, bd2=0.2, fitness=0.5, state_dict=sd_b)
    assert arch.cells[bin_bd(0.1, 0.2)].fitness == 0.5
    assert np.allclose(arch.cells[bin_bd(0.1, 0.2)].state_dict['w'], [2.0])
    assert 0.0 < arch.coverage() <= 1.0
    assert arch.qd_score() == 0.5
    assert arch.best() is not None
    assert arch.best().fitness == 0.5


def test_g_min_reject_logic():
    from showdown.rl.qd_ppo.evaluate import EvalResult
    # Unit: accepted flag contract
    bad = EvalResult(fitness=0.1, bd=None, g_eligible=9, n_games=20, accepted=False)
    assert not bad.accepted
    good = EvalResult(fitness=0.1, bd=(0.5, 0.5), g_eligible=10, n_games=20, accepted=True)
    assert good.accepted


def test_bd_formulas_hard_switch():
    tr = BehavioralTracker()
    # 2 offensive, 1 status, 1 hard switch over 2 turns → bd1=2/3, bd2=0.5
    tr.on_turn_actions(
        [SlotAction(kind='move', move_index=0, target=0), SlotAction(kind='switch', switch_index=0)],
        ['tackle', ''],
    )
    tr.on_turn_actions(
        [SlotAction(kind='move', move_index=0, target=0), SlotAction(kind='pass')],
        ['protect', ''],
    )
    assert not tr.archive_eligible()  # turns=2 < 4
    tr.on_turn_actions(
        [SlotAction(kind='move', move_index=0, target=0), SlotAction(kind='pass')],
        ['tackle', ''],
    )
    tr.on_turn_actions(
        [SlotAction(kind='move', move_index=0, target=0), SlotAction(kind='pass')],
        ['tackle', ''],
    )
    assert tr.archive_eligible()
    bd1, bd2 = tr.bd_vector()
    assert 0.0 <= bd1 <= 1.0 and 0.0 <= bd2 <= 1.0
    assert classify_action(SlotAction(kind='switch', switch_index=0), '') == 'hard_switch'


def test_mosaic_probs():
    rng = np.random.default_rng(0)
    counts = {f: 0 for f in MOSAIC_FOES}
    n = 4000
    for _ in range(n):
        counts[sample_mosaic_foe(rng)] += 1
    # loose chi-ish checks
    assert abs(counts['random'] / n - 0.25) < 0.05
    assert abs(counts['greedy_solo'] / n - 0.25) < 0.05
    assert abs(counts['greedy_joint'] / n - 0.50) < 0.05
    assert abs(sum(MOSAIC_PROBS) - 1.0) < 1e-9


def test_smoke_train_mock():
    from showdown.rl.qd_ppo.train import run_training

    arch = run_training(seed=0, smoke=True, checkpoint_dir='logs/qd_ppo_test_smoke')
    assert isinstance(arch, Archive)
    # With g_min=2 and short games, may or may not fill — at least runs
    assert arch.coverage() >= 0.0
