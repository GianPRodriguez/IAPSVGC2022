"""El resumen nocturno previo al kick-off se genera aunque falten etapas."""
import json
import sys

sys.path.insert(0, '.')

from showdown.rl.ddqn import kickoff_summary as KS


def _eval(step, wrs):
    foes = ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')
    return {'kind': 'eval', 'step': step, 'mean_wr': sum(wrs) / 4,
            'rows': [{'foe': f, 'win_rate': w} for f, w in zip(foes, wrs)]}


def _run(d, wrs_by_step):
    d.mkdir(parents=True)
    with open(d / 'train_metrics.jsonl', 'w', encoding='utf-8') as fh:
        fh.write(json.dumps({'kind': 'window', 'step': 1, 'learner': {'mean_q': 0.9, 'td_error': 0.15}}) + '\n')
        for step, wrs in wrs_by_step:
            fh.write(json.dumps(_eval(step, wrs)) + '\n')


def test_summary_with_partial_results(tmp_path):
    _run(tmp_path / 'tune' / 'T0_base', [(200000, [0.6, 0.1, 0.1, 0.1]), (400000, [0.7, 0.2, 0.2, 0.2])])
    _run(tmp_path / 'tune' / 'T1_lr1e-4', [(400000, [0.5, 0.1, 0.0, 0.1])])
    _run(tmp_path / 'ablation' / 'flat_sym_s0', [(400000, [0.5, 0.1, 0.1, 0.1])])
    _run(tmp_path / 'ablation' / 'flat_sym_s1', [(400000, [0.6, 0.1, 0.1, 0.1])])
    (tmp_path / 'queue.log').write_text('2026-11-16 00:00:00 inicio\n', encoding='utf-8')
    assert KS.main([str(tmp_path)]) == 0
    md = (tmp_path / 'kickoff_summary.md').read_text(encoding='utf-8')
    assert 'Mejor configuración (media final):** `T0_base`' in md
    assert '| flat_sym | 2 |' in md
    assert 'Sin datos' in md          # etapas que faltan no rompen el resumen
