"""Tests for final train reporte.txt / reporte.json."""
from __future__ import annotations

from pathlib import Path

from showdown.rl.train_report import (
    SUITE_GAMES_DEFAULT,
    SuiteFoeResult,
    TrainReport,
    render_reporte_txt,
    resolve_suite_games,
    write_train_report,
)


def test_resolve_suite_games():
    assert resolve_suite_games(suite_games=None, smoke=True) == 0
    assert resolve_suite_games(suite_games=None, smoke=False) == SUITE_GAMES_DEFAULT
    assert resolve_suite_games(suite_games=200, smoke=True) == 200
    assert resolve_suite_games(suite_games=0, smoke=False) == 0


def test_render_and_write_reporte(tmp_path: Path):
    report = TrainReport(
        agent='qd_ppo',
        seed=0,
        global_steps=1000,
        t_total=1_000_000,
        elapsed_sec=12.5,
        steps_per_sec=80.0,
        metrics={'coverage': 0.12, 'best_f': 0.3},
        suite=[
            SuiteFoeResult(
                foe='random', n_games=4, wins=2, losses=1, ties=1,
                mean_return=0.1, win_rate=0.5,
            ),
        ],
        suite_games_per_foe=4,
        artifacts={'checkpoint_dir': str(tmp_path)},
        notes=['unit test'],
    )
    text = render_reporte_txt(report)
    assert '# Train Report — qd_ppo' in text
    assert 'coverage: 0.12' in text
    assert 'MEAN_WR: 0.500' in text
    assert 'reporte.json' in text
    txt_p, json_p = write_train_report(tmp_path, report)
    assert txt_p.is_file()
    assert json_p.is_file()
    body = txt_p.read_text(encoding='utf-8')
    assert 'Outcome metrics' in body
