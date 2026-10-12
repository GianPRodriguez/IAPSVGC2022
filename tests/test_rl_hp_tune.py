"""Tests del tune HP RL (sin train largo / sin torch obligatorio)."""
from __future__ import annotations

from showdown.inference_sim.rl_hp_tune import (
    TrialOutcome,
    ddqn_candidates,
    override_deltas,
    parse_mean_wr,
    pick_best,
    ppo_combat_candidates,
    preview_nash_candidates,
    preview_ppo_candidates,
    preview_qd_pga_candidates,
    qd_combat_candidates,
)


def test_candidate_grids():
    assert len(ddqn_candidates()) == 6
    assert len(ppo_combat_candidates()) == 6
    assert len(preview_ppo_candidates()) == 6
    assert len(qd_combat_candidates()) == 6
    assert len(preview_nash_candidates()) == 3
    assert len(preview_qd_pga_candidates()) == 6
    assert 'lr' in ddqn_candidates()[0].overrides
    assert 'gamma' in ddqn_candidates()[0].overrides
    assert 'clip_eps' in ppo_combat_candidates()[0].overrides
    assert qd_combat_candidates()[0].overrides == ppo_combat_candidates()[0].overrides
    assert list(preview_nash_candidates()[0].overrides) == ['lr']
    assert preview_qd_pga_candidates()[0].overrides == preview_ppo_candidates()[0].overrides


def test_override_deltas_tracks_param_change():
    d0 = override_deltas(None, {'lr': 1e-4, 'clip_eps': 0.1})
    assert d0['lr']['from'] is None and d0['lr']['to'] == 1e-4
    d1 = override_deltas({'lr': 1e-4, 'clip_eps': 0.1}, {'lr': 3e-4, 'clip_eps': 0.1})
    assert abs(d1['lr']['delta'] - 2e-4) < 1e-12
    assert d1['clip_eps']['delta'] == 0.0


def test_parse_mean_wr():
    assert parse_mean_wr('MEAN_WR: 0.425\n') == 0.425
    assert parse_mean_wr('MEAN_SCORE: -0.12\n') == -0.12
    assert parse_mean_wr('no score') is None


def test_pick_best():
    outs = [
        TrialOutcome(1, {'lr': 1e-4}, 0.3, True, 1.0, 0, 'a'),
        TrialOutcome(2, {'lr': 3e-4}, 0.5, True, 1.0, 0, 'b'),
        TrialOutcome(3, {'lr': 1e-3}, None, False, 1.0, 1, 'c'),
    ]
    b = pick_best(outs)
    assert b is not None and b.trial_index == 2


def test_dry_run_ddqn_cli():
    from showdown.inference_sim.run_ddqn_hp_tune import main
    assert main(['--dry-run', '--max-trials', '2', '--seed', '0']) == 0


def test_dry_run_ppo_and_preview_cli():
    from showdown.inference_sim.run_ppo_hp_tune import main as ppo_main
    from showdown.inference_sim.run_team_preview_ppo_hp_tune import main as prev_main
    assert ppo_main(['--dry-run', '--max-trials', '2']) == 0
    assert prev_main(['--dry-run', '--max-trials', '2']) == 0


def test_dry_run_qd_combat_clis():
    from showdown.inference_sim.run_qd_pga_hp_tune import main as pga_main
    from showdown.inference_sim.run_qd_ppo_hp_tune import main as ppo_qd_main
    from showdown.inference_sim.run_qd_ppga_hp_tune import main as ppga_main

    assert ppo_qd_main(['--dry-run', '--max-trials', '2']) == 0
    assert pga_main(['--dry-run', '--max-trials', '2']) == 0
    assert ppga_main(['--dry-run', '--max-trials', '2']) == 0


def test_dry_run_preview_nash_and_qd_pga_clis():
    from showdown.inference_sim.run_team_preview_nash_hp_tune import main as nash_main
    from showdown.inference_sim.run_team_preview_qd_pga_hp_tune import main as qd_prev_main

    assert nash_main(['--dry-run', '--max-trials', '2']) == 0
    assert qd_prev_main(['--dry-run', '--max-trials', '2']) == 0
