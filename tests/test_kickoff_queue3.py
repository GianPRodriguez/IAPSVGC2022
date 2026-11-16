"""Tercera cola: clasificación de jugadas, reglas de elección y resumen."""
import json
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, '.')

from showdown.rl.ddqn import kickoff_summary3 as KS3
from showdown.rl.ddqn.policy_diag import classify


def _move(mid, target, bp):
    return SimpleNamespace(id=mid, target=SimpleNamespace(name=target), base_power=bp)


def _battle(moves, force=(False, False)):
    return SimpleNamespace(available_moves=[moves, moves], force_switch=list(force))


def _act(kind, move_index=0, target=0):
    return SimpleNamespace(kind=kind, move_index=move_index, target=target)


def test_classify_categories():
    from showdown.rl.constants import PASS_ID
    only_pass = np.zeros(40)
    only_pass[PASS_ID] = 1
    with_opts = np.ones(40)
    b = _battle([_move('woodhammer', 'NORMAL', 120), _move('earthquake', 'ALL_ADJACENT', 100),
                 _move('hypervoice', 'ALL_ADJACENT_FOES', 90), _move('helpinghand', 'ADJACENT_ALLY', 0),
                 _move('protect', 'SELF', 0)])
    assert classify(b, 0, _act('pass'), only_pass) == 'pass_obligatorio'
    assert classify(b, 0, _act('pass'), with_opts) == 'pass_con_opciones'
    assert classify(b, 0, _act('switch'), with_opts) == 'cambio_voluntario'
    assert classify(_battle([], (True, False)), 0, _act('switch'), with_opts) == 'cambio_forzado'
    assert classify(b, 0, _act('move', 0, 1), with_opts) == 'ataque_rival'
    assert classify(b, 0, _act('move', 0, 2), with_opts) == 'ataque_aliado'
    assert classify(b, 0, _act('move', 1, 3), with_opts) == 'area_con_aliado'
    assert classify(b, 0, _act('move', 2, 3), with_opts) == 'area_rivales'
    assert classify(b, 0, _act('move', 3, 2), with_opts) == 'apoyo_aliado'
    assert classify(b, 0, _act('mega', 4, 3), with_opts) == 'protect'


def _run(d, mean):
    d.mkdir(parents=True)
    rows = [{'foe': f, 'win_rate': mean} for f in ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')]
    (d / 'train_metrics.jsonl').write_text(
        json.dumps({'kind': 'eval', 'step': 400000, 'mean_wr': mean, 'rows': rows}) + '\n', encoding='utf-8')


def test_choose_rules(tmp_path):
    q1, q2, q3 = tmp_path / 'k1', tmp_path / 'k2', tmp_path / 'k3'
    _run(q1 / 'tune' / 'T0_base', 0.30)
    _run(q1 / 'tune' / 'T1_lr1e-4', 0.38)
    _run(q1 / 'ablation' / 'struct_sym_s1', 0.28)
    _run(q2 / 'lr_confirm' / 'lr1e-4_s1', 0.25)          # en contra, pero la media sigue > 0
    _run(q3 / 'combo_lr1e-4_b512', 0.36)                 # no supera a lr 1e-4 solo
    assert KS3.choose(q3, q1, q2).startswith('lr=1e-4 batch=256')
    _run(q1 / 'ablation' / 'struct_sym_s2', 0.40)
    _run(q2 / 'lr_confirm' / 'lr1e-4_s2', 0.20)          # ahora la media es negativa
    assert KS3.choose(q3, q1, q2).startswith('lr=3e-4 batch=256')


def test_summary3_with_partial_results(tmp_path):
    q1, q2, q3 = tmp_path / 'k1', tmp_path / 'k2', tmp_path / 'k3'
    _run(q1 / 'tune' / 'T0_base', 0.30)
    q3.mkdir(parents=True)
    (q3 / 'load.csv').write_text('ts,etapa,cpu_ocupada,cpu_otros_usuarios\n1,diag,40,120\n2,diag,60,80\n',
                                 encoding='utf-8')
    assert KS3.main([str(q3), str(q1), str(q2)]) == 0
    md = (q3 / 'kickoff_summary3.md').read_text(encoding='utf-8')
    assert '| diag | 2 | 50 % | 100 / 120 |' in md
    assert '| base lr 3e-4 | 30.0 % |' in md
