"""El resumen de la quinta cola se genera aunque falten etapas."""
import json
import sys

sys.path.insert(0, '.')

from showdown.rl.ddqn import kickoff_summary5 as KS5


def test_summary5_partial(tmp_path):
    q2, q3, q5 = tmp_path / 'k2', tmp_path / 'k3', tmp_path / 'k5'
    (q5 / 'preview_check').mkdir(parents=True)
    ev = {'mean_wr': 0.9, 'base_wr': 0.6, 'delta_wr': 0.3, 'delta_se': 0.03}
    (q5 / 'preview_check' / 'preview_check.json').write_text(json.dumps({
        'split_train': ev, 'split_eval': ev, 'formacion_fija': {**ev, 'traídos': ['A', 'B', 'C', 'D']},
        'formaciones_train': [{'traídos': ['A', 'B', 'C', 'D'], 'frac': 0.4}],
        'formaciones_distintas_train': 7, 'formaciones_distintas_eval': 9,
        'replays': {'dir': 'x', 'contextos': []}}), encoding='utf-8')
    assert KS5.main([str(q5), str(q2), str(q3)]) == 0
    md = (q5 / 'kickoff_summary5.md').read_text(encoding='utf-8')
    assert '| Rivales con equipos `eval` (no vistos) | 90.0 % | 60.0 % | +30.0 (± 3.0) |' in md
    assert 'Sin datos.' in md
