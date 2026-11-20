"""Resumen de la cola de medición de MCTS+PUCT v2 (showdown/inference_sim/mcts_g_summary.py)."""
from __future__ import annotations

from showdown.inference_sim.mcts_g_summary import load, paired_diff, render, wilson


def _report(wins: int, losses: int, elapsed: float = 100.0, errors: int = 0) -> str:
    n = wins + losses
    return (
        '=== Robust E2E Report ===\n'
        f'elapsed_s: {elapsed}\n'
        f'player_a_wins: {wins}\nplayer_b_wins: {losses}\n'
        f'series_errors: {errors}\n'
        f'win_rate_player_a: {100.0 * wins / n:.1f}%\n'
        'win_rate_pairs: 50.0%\n'
    )


def test_wilson_bounds():
    lo, hi = wilson(50, 100)
    assert 40 < lo < 50 < hi < 60
    assert wilson(0, 0) == (0.0, 0.0)


def test_load_render_and_paired(tmp_path):
    rep = tmp_path / 'reports'
    rep.mkdir()
    (rep / 'v2_n200_s9101.txt').write_text(_report(150, 150), encoding='utf-8')
    (rep / 'v2_n200_s9102.txt').write_text(_report(160, 140), encoding='utf-8')
    (rep / 'v2_n50_s9101.txt').write_text(_report(140, 160), encoding='utf-8')
    (rep / 'v2_n50_s9102.txt').write_text(_report(145, 155, errors=2), encoding='utf-8')
    (rep / 'ignored.txt').write_text('x', encoding='utf-8')
    res = load(tmp_path)
    assert set(res) == {'v2_n200', 'v2_n50'}
    assert res['v2_n50'][9102]['series_errors'] == 2
    pd = paired_diff(res['v2_n50'], res['v2_n200'])
    assert pd is not None and pd[2] == 2 and pd[0] < 0
    text = render(res)
    assert '| v2_n200 | 9101–9102 (2) | 600 | 310 |' in text
    assert '| v2_n50 |' in text and '(2 sem.)' in text


def _nash(d1, d2, holm1, holm2):
    lines = [
        '--- McNemar P1 vs P0',
        f'emparejadas=3400 A_mejor=10 B_mejor=5 diferencia={d1} pts p=0.01',
        '--- McNemar P2 vs P0',
        f'emparejadas=3400 A_mejor=10 B_mejor=5 diferencia={d2} pts p=0.02',
        '--- Holm (m=2; P1, P2)',
        f'p=0.01 p_holm=0.02 {holm1}',
        f'p=0.02 p_holm=0.02 {holm2}',
    ]
    return '\n'.join(lines) + '\n'


def test_nash_decision_rule():
    from showdown.inference_sim.mcts_g_summary import nash_decision
    assert nash_decision(_nash('+1.50', '+2.00', 'rechaza H0', 'rechaza H0')) == 'P2'
    assert nash_decision(_nash('+1.50', '+2.00', 'rechaza H0', 'no rechaza')) == 'P1'
    assert nash_decision(_nash('-1.50', '+2.00', 'rechaza H0', 'no rechaza')) is None  # empeora
    assert nash_decision(_nash('+1.50', '+2.00', 'no rechaza', 'no rechaza')) is None
    assert nash_decision('basura') is None

