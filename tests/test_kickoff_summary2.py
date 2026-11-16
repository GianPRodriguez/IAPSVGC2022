"""El resumen de la segunda cola se genera aunque falten etapas."""
import json
import sys

sys.path.insert(0, '.')

from showdown.rl.ddqn import kickoff_summary2 as KS2


def _run(d, mean):
    d.mkdir(parents=True)
    rows = [{'foe': f, 'win_rate': mean} for f in ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')]
    (d / 'train_metrics.jsonl').write_text(
        json.dumps({'kind': 'eval', 'step': 400000, 'mean_wr': mean, 'rows': rows}) + '\n', encoding='utf-8')


def test_summary2_pairs_lr_by_seed(tmp_path):
    q1, q2 = tmp_path / 'kickoff', tmp_path / 'kickoff2'
    _run(q1 / 'tune' / 'T0_base', 0.30)
    _run(q1 / 'tune' / 'T1_lr1e-4', 0.38)
    _run(q1 / 'ablation' / 'struct_sym_s1', 0.28)
    _run(q2 / 'lr_confirm' / 'lr1e-4_s1', 0.33)
    (q2 / 'preview_v2_ent02').mkdir(parents=True)
    (q2 / 'preview_v2_ent02' / 'reporte.txt').write_text(
        'MEAN_WR: 0.520000\nBASE_WR: 0.505000\nDELTA_WR: +0.015000 (SE 0.030000)\n', encoding='utf-8')
    assert KS2.main([str(q2), str(q1)]) == 0
    md = (q2 / 'kickoff_summary2.md').read_text(encoding='utf-8')
    assert '| 0 | 30.0 % | 38.0 % | +8.0 |' in md and '| 1 | 28.0 % | 33.0 % | +5.0 |' in md
    assert '| 2 | — | — | — |' in md
    assert 'preview_v2_ent02 | 0.520000 | 0.505000 |' in md
    assert 'Diferencia media: +6.5 puntos en 2 semillas (2 de 2' in md
