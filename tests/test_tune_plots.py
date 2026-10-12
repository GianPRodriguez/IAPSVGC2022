"""Tests de gráficas tune/ledger (SVG stdlib, sin matplotlib obligatorio)."""
from __future__ import annotations

import json
from pathlib import Path

from showdown.inference_sim.tune_plots import (
    TrialPoint,
    append_ledger_entry,
    plot_hp_tune_run,
    plot_ledger,
    trials_from_hp_history,
    write_trial_charts,
)


def test_trials_from_hp_history():
    rows = [
        {'event': 'tune_start'},
        {
            'event': 'hp_trial', 'trial_index': 1, 'score': 0.3, 'accepted': True,
            'overrides': {'lr': 1e-4}, 'best_score_so_far': 0.3,
        },
        {
            'event': 'hp_trial', 'trial_index': 2, 'score': 0.5, 'accepted': True,
            'overrides': {'lr': 3e-4}, 'best_score_so_far': 0.5,
        },
        {'event': 'tune_done', 'best_score': 0.5},
    ]
    trials = trials_from_hp_history(rows)
    assert len(trials) == 2
    assert trials[1].score == 0.5
    assert trials[1].best_so_far == 0.5


def test_write_trial_charts_svg(tmp_path: Path):
    trials = [
        TrialPoint(1, 0.2, True, 'a', 0.2),
        TrialPoint(2, 0.4, True, 'b', 0.4),
        TrialPoint(3, 0.3, False, 'c', 0.4),
    ]
    stem = tmp_path / 'demo'
    arts = write_trial_charts(trials, stem=stem, title='demo')
    svg = Path(arts['svg'])
    assert svg.is_file()
    text = svg.read_text(encoding='utf-8')
    assert '<svg' in text
    assert 'demo' in text


def test_plot_hp_tune_run_and_ledger(tmp_path: Path, monkeypatch):
    import showdown.inference_sim.tune_plots as tp

    fig_dir = tmp_path / 'figures'
    ledger = fig_dir / 'hp_tune_ledger.jsonl'
    monkeypatch.setattr(tp, 'FIGURES_DIR', fig_dir)
    monkeypatch.setattr(tp, 'LEDGER_PATH', ledger)

    hist = tmp_path / 'ddqn_hp_tune_history.jsonl'
    rows = [
        {'event': 'tune_start', 'run_id': 'r1'},
        {
            'event': 'hp_trial', 'trial_index': 1, 'score': 0.4, 'accepted': True,
            'overrides': {'lr': 1e-4, 'gamma': 0.95}, 'best_score_so_far': 0.4,
        },
        {
            'event': 'hp_trial', 'trial_index': 2, 'score': 0.55, 'accepted': True,
            'overrides': {'lr': 3e-4, 'gamma': 0.95}, 'best_score_so_far': 0.55,
        },
        {'event': 'tune_done', 'run_id': 'r1', 'best_score': 0.55},
    ]
    hist.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')

    info = plot_hp_tune_run(
        agent='ddqn',
        history_path=hist,
        run_id='r1',
        best_score=0.55,
        best_overrides={'lr': 3e-4, 'gamma': 0.95},
        update_ledger=True,
    )
    assert Path(info['latest_figures']['svg']).is_file()
    assert ledger.is_file()
    # second run appends
    append_ledger_entry(
        agent='ddqn', run_id='r2', best_score=0.6,
        best_overrides={'lr': 1e-3}, n_trials=2, source='x',
    )
    arts = plot_ledger(agent='ddqn')
    assert Path(arts['svg']).is_file()
    lines = ledger.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) >= 2


def test_dry_run_cli_plot():
    from showdown.inference_sim.run_plot_tune_history import main
    # ledger-only always ok even if empty
    assert main(['--ledger-only']) == 0
