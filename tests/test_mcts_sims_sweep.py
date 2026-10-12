"""Tests del barrido N_sim / criterio de meseta (sin Showdown)."""
from __future__ import annotations

from showdown.inference_sim.mcts_sims_sweep import (
    PlateauCriteria,
    SweepPoint,
    build_tracking_log,
    default_sims_grid,
    env_key_for_agent,
    format_tracking_text,
    step_transition,
    suggest_plateau,
)


def test_default_grids():
    assert default_sims_grid('mcts_puct') == (50, 100, 200, 400)
    assert default_sims_grid('mcts_puct_nash') == (25, 50, 100, 200)
    assert env_key_for_agent('mcts_puct') == 'MCTS_SIMS'
    assert env_key_for_agent('mcts_puct_nash') == 'MCTS_NASH_SIMS'


def test_plateau_when_gain_small_and_time_jumps():
    pts = [
        SweepPoint(50, wr_raw=48.0, pair_wr=50.0, elapsed_s=100.0),
        SweepPoint(100, wr_raw=51.0, pair_wr=52.0, elapsed_s=200.0),
        SweepPoint(200, wr_raw=51.5, pair_wr=52.5, elapsed_s=400.0),  # +0.5pp, ×2 time
        SweepPoint(400, wr_raw=51.8, pair_wr=53.0, elapsed_s=900.0),
    ]
    d = suggest_plateau(
        pts,
        PlateauCriteria(delta_wr_pp=1.5, time_ratio=1.8, prefer_metric='wr_raw'),
    )
    # 100→200: gain 0.5 < 1.5 and time 2.0 ≥ 1.8 → meseta en 100
    assert d.recommended_n_sims == 100
    assert 'meseta' in d.reason


def test_best_metric_when_no_plateau():
    pts = [
        SweepPoint(50, wr_raw=40.0, pair_wr=40.0, elapsed_s=10.0),
        SweepPoint(100, wr_raw=45.0, pair_wr=45.0, elapsed_s=12.0),  # big gain, little time
        SweepPoint(200, wr_raw=50.0, pair_wr=50.0, elapsed_s=14.0),
    ]
    d = suggest_plateau(
        pts,
        PlateauCriteria(delta_wr_pp=1.5, time_ratio=1.8, prefer_metric='wr_raw'),
    )
    assert d.recommended_n_sims == 200
    assert 'sin meseta' in d.reason


def test_pair_wr_metric():
    pts = [
        SweepPoint(50, wr_raw=60.0, pair_wr=40.0, elapsed_s=10.0),
        SweepPoint(100, wr_raw=55.0, pair_wr=50.0, elapsed_s=12.0),
    ]
    d = suggest_plateau(
        pts,
        PlateauCriteria(delta_wr_pp=1.5, time_ratio=1.8, prefer_metric='pair_wr'),
    )
    assert d.recommended_n_sims == 100


def test_step_transition_and_tracking_log():
    a = SweepPoint(50, wr_raw=48.0, pair_wr=50.0, elapsed_s=100.0)
    b = SweepPoint(100, wr_raw=51.0, pair_wr=52.0, elapsed_s=200.0)
    tr0 = step_transition(None, a, prefer_metric='wr_raw')
    tr1 = step_transition(a, b, prefer_metric='wr_raw')
    assert tr0['n_sims_prev'] is None
    assert tr1['n_sims_delta'] == 50
    assert abs(tr1['delta_wr_raw_pp'] - 3.0) < 1e-9
    assert abs(tr1['time_ratio'] - 2.0) < 1e-9

    crit = PlateauCriteria(delta_wr_pp=1.5, time_ratio=1.8, prefer_metric='wr_raw')
    d = suggest_plateau([a, b], crit)
    tracking = build_tracking_log(
        [a, b],
        d,
        agent='mcts_puct',
        criteria=crit,
        player_b='greedy_joint',
        seed=10,
        teams=40,
        run_id='test_run',
        dry_run=True,
    )
    assert tracking['varying_hyperparam'] == 'n_sims'
    assert tracking['frozen_search_hyperparams']['c_puct'] == 1.5
    assert len(tracking['steps']) == 2
    text = format_tracking_text(tracking)
    assert 'N_sim = 50' in text or 'N_sim 50' in text
    assert 'recommended_n_sims' in text
