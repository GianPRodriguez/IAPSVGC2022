"""Informe del entrenamiento contra el nivel 3 (protocolo §4.20): pareo, pruebas T / N y Holm."""
from __future__ import annotations

import json

from scripts.analysis import l3_report as R  # noqa: E402


def _write(diag, fam, arm, k, wins_by_foe):
    partidas, res = [], {}
    for foe, wins in wins_by_foe.items():
        res[foe] = {'win': sum(wins), 'loss': len(wins) - sum(wins)}
        partidas += [{'rival': foe, 'res': 'win' if w else 'loss'} for w in wins]
    (diag / f'ev_{fam}_{arm}_s{k}.json').write_text(json.dumps({'resultados': res, 'partidas': partidas}))


def test_report_detects_gain_vs_level3_and_noninferiority(tmp_path):
    diag = tmp_path / 'diag'
    diag.mkdir()
    import random
    rng = random.Random(0)
    for fam in R.FAMS:
        for k in R.SEEDS:
            base = {f: [int(rng.random() < 0.85) for _ in range(200)] for f in R.HEUR}
            l3_c = [int(rng.random() < 0.63) for _ in range(200)]
            l3_l = [1 if (x or rng.random() < 0.35) else 0 for x in l3_c]       # L3 gana más contra el nivel 3
            _write(diag, fam, 'C', k, {**base, R.L3: l3_c})
            _write(diag, fam, 'L3', k, {**base, R.L3: l3_l})
            _write(diag, fam, 'prev', k, {**base, R.L3: l3_c})
    txt = R.build(tmp_path)
    assert txt.count('**se confirma**') == 4                                   # T y N en las dos familias
    t = R.paired(diag, 'ppo', 'C', 'L3', (R.L3,))
    assert t['n'] == 600 and t['diff'] > 0.05 and t['p_sup'] < 1e-6
    n = R.paired(diag, 'ppo', 'C', 'L3', R.HEUR)
    assert n['diff'] == 0 and n['p_ni'] < 0.01
    assert 'nivel 3' in txt and '| ppo | prev | s0 |' in txt


def test_holm_and_missing_data(tmp_path):
    (tmp_path / 'diag').mkdir()
    txt = R.build(tmp_path)
    assert 'sin datos' in txt
    h = R.holm({'a': 0.001, 'b': 0.04})
    assert h['a'][1] and h['b'][1]
    h = R.holm({'a': 0.03, 'b': 0.04})
    assert not h['a'][1] and not h['b'][1]
