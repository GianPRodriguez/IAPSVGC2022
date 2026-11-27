"""F1 de la familia QD 2.0 (protocolo §4.11): enfrentamientos, estadística y humo con el simulador de prueba."""
from __future__ import annotations

import math

import numpy as np

from showdown.rl.qd_ppo.confirm_f1 import (
    FOES, balanced_matchups, compute_stats, holm, paired, wilson, wins_of,
)


def test_balanced_matchups_reproducible_and_balanced():
    a = balanced_matchups(7, 5, 'eval')
    b = balanced_matchups(7, 5, 'eval')
    assert a == b and len(a) == 15
    assert [m['opponent'] for m in a] == [f for f in FOES for _ in range(5)]
    assert all(m['split'] == 'eval' for m in a)
    assert balanced_matchups(8, 5, 'eval') != a


def test_wins_and_wilson():
    assert wins_of([1.0, -1.0, 0.0, None]) == [1, 0, 0, None]
    lo, hi = wilson(80, 100)
    assert 0.70 < lo < 0.80 < hi < 0.88


def test_paired_noninferiority_and_superiority():
    rng = np.random.default_rng(0)
    c = [int(x) for x in rng.random(20000) < 0.8]
    same = list(c)
    t = paired(same, c, shift=0.015)
    assert t['diff'] == 0 and t['p'] == 0.0           # sin diferencias: se rechaza H0 (d ≤ −1,5)
    worse = [x if rng.random() > 0.1 else 0 for x in c]   # ~−8 pts
    assert paired(worse, c, shift=0.015)['p'] > 0.99
    better = [1 if rng.random() < 0.3 else x for x in c]
    assert paired(better, c)['p'] < 1e-6
    t = paired([1, None, 0], [0, 1, None])
    assert t['n'] == 1 and t['p'] == 1.0              # caídas fuera; n < 2 no prueba nada


def test_holm_step_down():
    h = holm({'a': 0.001, 'b': 0.02, 'c': 0.04, 'd': 0.3})
    assert h['a']['rejected'] and h['b']['rejected'] is False   # 0,02 > 0,05/3
    assert h['c']['rejected'] is False and h['d']['rejected'] is False
    assert math.isclose(h['a']['p_holm'], 0.004)
    assert h['b']['p_holm'] >= h['a']['p_holm']


def test_compute_stats_structure():
    rng = np.random.default_rng(1)
    n = 3 * 400
    foes = [f for f in FOES for _ in range(400)]
    base = [int(x) for x in rng.random(n) < 0.8]
    res = {'foes': foes, 'wins': {
        'control': base, 'I': list(base), 'K': [0 if rng.random() < 0.2 else x for x in base],
        'J': [1 if (f == 'expectimax_bayesian' and rng.random() < 0.5) else x for x, f in zip(base, foes)],
        'init': list(base)}}
    st = compute_stats(res)
    assert set(st['tests']) == {'N_I', 'N_J', 'N_K', 'S_J'}
    assert st['tests']['N_I']['rejected'] and not st['tests']['N_K']['rejected']
    assert st['tests']['S_J']['rejected']
    assert st['rates']['I']['3 heur.']['n'] == n
    assert 'K - control | greedy_joint' in st['descriptive']
    assert 'I - init | 3 heur.' in st['descriptive']


def test_deploy_and_confirm_mock(tmp_path):
    """Humo: archivos con dos élites, despliegue y confirmación con el simulador de prueba."""
    import json

    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    from showdown.rl.qd_ppo.archive import Archive
    from showdown.rl.qd_ppo.confirm_f1 import run_confirm, run_deploy
    from showdown.rl.qd_ppo.train2 import _state

    base = PPOAgent(PPOConfig(device='cpu'))
    base._ensure_torch()
    ck = tmp_path / 'ctrl.pt'
    base.save(ck)
    paths = {}
    for name in ('I', 'J'):
        arch = Archive()
        for i in range(2):
            a = PPOAgent(PPOConfig(device='cpu', seed=i + 1))
            a._ensure_torch()
            arch.try_insert(bd1=0.1 + 0.5 * i, bd2=0.5, fitness=0.1 * i, state_dict=_state(a))
        paths[name] = str(arch.save(tmp_path / f'{name}.pt'))
    dep = run_deploy(paths, cfg_checkpoint=str(ck), out=tmp_path / 'dep', top=2, games=6, mock=True,
                     num_envs=2, chunk=3)
    assert set(dep['chosen']) == {'I', 'J'} and len(dep['rows']) == 4
    assert (tmp_path / 'dep' / 'deploy_I.pt').is_file()
    res = run_confirm({'I': str(tmp_path / 'dep' / 'deploy_I.pt'), 'control': str(ck)},
                      cfg_checkpoint=str(ck), out=tmp_path / 'conf', per_foe=2, mock=True, num_envs=2,
                      chunk=4)
    assert len(res['wins']['I']) == 6 and len(res['wins']['control']) == 6
    saved = json.loads((tmp_path / 'conf' / 'results.json').read_text(encoding='utf-8'))
    assert saved['foes'] == [f for f in FOES for _ in range(2)]
