"""Fase 1 de expectimax: analizador de términos de V (contrafactual de amenaza y puerta)."""
import json
import sys

sys.path.insert(0, '.')

from showdown.inference_sim import analyze_vterms as av


def _rec(pairs, accepted=False, p1_v=100.0, best_v=100.0):
    return {'pairs': pairs, 'accepted': accepted, 'p1_v': p1_v, 'best_v': best_v,
            'w_tempo': 1.0, 'w_threat': 0.0, 'margin': 40.0}


def _p(off, threat, p1=False, best=False):
    return {'off': off, 'vgc': 0.0, 'tempo': 0.0, 'threat': threat, 'syn': 0.0,
            'p1': p1, 'best': best, 'kinds': ['damage', 'damage']}


def test_threat_counterfactual_and_gate(tmp_path):
    # Decisión 1: el par de más ofensa también recibe mucha amenaza → w alto lo cambia.
    r1 = _rec([_p(300, 200, p1=True, best=True), _p(280, 0)])
    # Decisión 2: la amenaza no cambia el orden.
    r2 = _rec([_p(300, 0, best=True), _p(100, 50, p1=True)], accepted=True, best_v=300, p1_v=100)
    path = tmp_path / 'vt.123'
    path.write_text('\n'.join(json.dumps(r) for r in (r1, r2)), encoding='utf-8')
    rows = av.load(str(tmp_path / 'vt'))
    assert len(rows) == 2
    assert av._argmax(r1['pairs'], r1, 0.0) == 0 and av._argmax(r1['pairs'], r1, 0.5) == 1
    text = '\n'.join(av.summarize(rows, weights=(0.05, 0.5)))
    assert 'w=0.05    0.0 %' in text and 'w=0.5    50.0 %' in text
    assert 'mejor par de phase2 = par de joint:  50.0 %' in text
    assert 'aceptado y distinto de joint:  50.0 %' in text
