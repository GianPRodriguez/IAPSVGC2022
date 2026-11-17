"""Resumen del kick-off oficial (showdown/rl/kickoff_report.py)."""
import json

from showdown.rl import kickoff_report as KR


def _diag(path, wins):
    res = {f: {'win': w, 'loss': 200 - w} for f, w in zip(KR.FOES, wins)}
    path.write_text(json.dumps({'resultados': res}), encoding='utf-8')


def test_report_means_ci_and_comparison(tmp_path):
    d = tmp_path / 'diag'
    d.mkdir()
    for s in range(5):
        _diag(d / f'ddqn_s{s}_final_eval.json', (190, 120 + s, 125, 120))
        _diag(d / f'ddqn_s{s}_best_eval.json', (190, 110, 110, 110))
        _diag(d / f'ddqn_s{s}_final_own.json', (196, 186, 186, 170))
        _diag(d / f'ppo_s{s}_final_eval.json', (198, 165, 160 + s, 160))
    _diag(d / 'ppo_s0_best_eval.json', (198, 160, 160, 160))
    assert KR.main([str(tmp_path)]) == 0
    md = (tmp_path / 'kickoff_summary16.md').read_text(encoding='utf-8')
    assert '| DDQN | 5 |' in md and '| PPO | 5 |' in md
    # DDQN: (120+s + 125 + 120) / 600, media s=0..4 → 367/600 = 61,2 %
    assert '61.2 %' in md
    assert '## 4. PPO frente a DDQN' in md and '+20.0 puntos' in md  # 487/600 − 367/600


def test_mean_ci_matches_t_student():
    m, hw = KR.mean_ci([0.6, 0.62, 0.64, 0.66, 0.68])
    assert abs(m - 0.64) < 1e-9 and abs(hw - 2.776 * 0.0316228 / 5 ** 0.5) < 1e-4


def test_missing_data_does_not_crash(tmp_path):
    (tmp_path / 'diag').mkdir()
    assert KR.main([str(tmp_path)]) == 0
    assert 'sin datos' in (tmp_path / 'kickoff_summary16.md').read_text(encoding='utf-8')
