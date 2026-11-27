"""F2–F4 de la familia QD 2.0 (protocolo §4.12–§4.14): selector, fusión de resultados, exportación, mosaico."""
from __future__ import annotations

import json

import numpy as np


def test_logistic_and_cv_recover_rule():
    from showdown.rl.qd_ppo.selector_f2 import cv_lambda, fit_all, fit_logistic, predict
    rng = np.random.default_rng(0)
    n = 1200
    x = rng.integers(0, 2, n).astype(float)
    X = np.c_[x, rng.random(n), np.ones(n)]
    # élite 0 gana si x = 1, élite 1 si x = 0
    Y = np.c_[(rng.random(n) < np.where(x == 1, 0.9, 0.4)), (rng.random(n) < np.where(x == 1, 0.4, 0.9))].astype(float)
    V = np.ones_like(Y, dtype=bool)
    w = fit_logistic(X, Y[:, 0], 1.0)
    assert w[0] > 1.0
    groups = [str(i % 60) for i in range(n)]
    lam, cv = cv_lambda(X, Y, V, groups, lambdas=(1.0, 10.0), k=3)
    assert cv[str(lam)] > 0.8
    choice = np.argmax(predict(fit_all(X, Y, V, lam), X), axis=1)
    assert np.mean(choice == (x == 0)) > 0.95


def test_opp_team_matches_backend_draw():
    from showdown.rl.foe_pool import trainer_defaults
    from showdown.rl.pokeenv_backend import _default_team_metas
    from showdown.rl.qd_ppo.selector_f2 import opp_team, team_key
    with trainer_defaults():
        for sp in ('fit', 'eval'):
            for seed in (3, 12345, 2**30):
                assert team_key(opp_team(seed, sp)) == team_key(_default_team_metas(seed=seed, split=sp)[1])


def test_features_and_analyze_end_to_end():
    """Con resultados sintéticos que dependen del equipo rival, el selector supera a la mejor élite única."""
    from showdown.rl.foe_pool import trainer_defaults
    from showdown.rl.qd_ppo.confirm_f1 import balanced_matchups
    from showdown.rl.qd_ppo.selector_f2 import analyze, build_features, opp_team
    from showdown.rl.team_roles import team_role_counts

    with trainer_defaults():
        m_fit = balanced_matchups(1, 300, 'fit')
        m_ev = balanced_matchups(2, 200, 'eval')
        rng = np.random.default_rng(3)

        def wins(ms):
            out = {'A': [], 'B': [], 'control': []}
            for m in ms:
                t = opp_team(m['seed'], m['split'])
                tw = team_role_counts(t.mons)[5] > 0      # el rival tiene Tailwind
                out['A'].append(int(rng.random() < (0.9 if tw else 0.5)))
                out['B'].append(int(rng.random() < (0.5 if tw else 0.9)))
                out['control'].append(int(rng.random() < 0.7))
            return out

        X, vocab = build_features([opp_team(m['seed'], m['split']) for m in m_fit[:5]])
        assert X.shape[1] == len(vocab) + 11 and np.all(X[:, -1] == 1)
        res = {'repertoire': ['A', 'B'], 'fit_split': 'fit', 'eval_split': 'eval',
               'fit': {'foes': [m['opponent'] for m in m_fit], 'seeds': [m['seed'] for m in m_fit],
                       'wins': {k: v for k, v in wins(m_fit).items() if k != 'control'}},
               'eval': {'foes': [m['opponent'] for m in m_ev], 'seeds': [m['seed'] for m in m_ev],
                        'wins': wins(m_ev)}}
        a = analyze(res)
    sel = a['rates']['selector']['3 heur.']['rate']
    single = [v for k, v in a['rates'].items() if k.startswith('mejor élite única')][0]['3 heur.']['rate']
    assert sel > single + 0.05 and sel > a['rates']['control']['3 heur.']['rate']
    assert set(a['chosen_counts']) == {'A', 'B'}


def test_merge_results_and_stats_cli(tmp_path):
    from showdown.rl.qd_ppo.confirm_f1 import main, merge_results
    foes = ['greedy_solo'] * 3
    base = {'foes': foes, 'seeds': [1, 2, 3]}
    (tmp_path / 'a.json').write_text(json.dumps({**base, 'wins': {'control': [1, 0, 1]}}))
    (tmp_path / 'b.json').write_text(json.dumps({**base, 'wins': {'control': [1, 1, 1]}}))
    m = merge_results([str(tmp_path / 'a.json'), str(tmp_path / 'b.json')],
                      {f'{tmp_path / "b.json"}:control': 'pvqd'})
    assert m['wins'] == {'control': [1, 0, 1], 'pvqd': [1, 1, 1]}
    (tmp_path / 'c.json').write_text(json.dumps({'foes': foes, 'seeds': [9, 9, 9], 'wins': {'x': [1, 1, 1]}}))
    try:
        merge_results([str(tmp_path / 'a.json'), str(tmp_path / 'c.json')])
        raise AssertionError('debió fallar')
    except ValueError:
        pass
    assert main(['stats', '--results', str(tmp_path / 'a.json'), '--results', str(tmp_path / 'b.json'),
                 '--rename', f'{tmp_path / "b.json"}:control=pvqd', '--ni', 'pvqd', '--sup', '',
                 '--out', str(tmp_path / 'st')]) == 0
    st = json.loads((tmp_path / 'st.json').read_text())
    assert set(st['tests']) == {'N_pvqd'}


def test_export_elites_and_pool_mosaic(tmp_path):
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    from showdown.rl.qd_ppo.archive import Archive
    from showdown.rl.qd_ppo.confirm_f1 import export_elites
    from showdown.rl.qd_ppo.mosaic import MOSAICS, make_sampler
    from showdown.rl.qd_ppo.train2 import _state

    base = PPOAgent(PPOConfig(device='cpu'))
    base._ensure_torch()
    ck = tmp_path / 'c.pt'
    base.save(ck)
    arch = Archive()
    arch.try_insert(bd1=0.1, bd2=0.1, fitness=0.2, state_dict=_state(base))
    arch.try_insert(bd1=0.9, bd2=0.9, fitness=0.1, state_dict=_state(base))
    arch.save(tmp_path / 'a.pt')
    keys = sorted(arch.cells)
    rows = [{'agent': 'K', 'cell': list(keys[0]), 'win_rate': 0.8, 'fitness': 0.5},
            {'agent': 'K', 'cell': list(keys[1]), 'win_rate': 0.9, 'fitness': 0.6},
            {'agent': 'J', 'cell': list(keys[0]), 'win_rate': 0.99, 'fitness': 0.9}]
    (tmp_path / 'd.json').write_text(json.dumps({'rows': rows}))
    paths = export_elites(str(tmp_path / 'a.pt'), str(tmp_path / 'd.json'), 'K', 5, str(ck), tmp_path / 'ex')
    assert len(paths) == 2 and f'_{keys[1][0]}_{keys[1][1]}' in paths[0].name
    PPOAgent(PPOConfig(device='cpu')).load(paths[0])
    foes, probs = MOSAICS['pool']
    assert 'ppo_v1' in foes and abs(sum(probs) - 1) < 1e-9
    s = make_sampler('pool')
    assert {s(np.random.default_rng(i)) for i in range(200)} == set(foes)


def test_team_roles_generic():
    from showdown.inference_sim.team_factory import load_team_file
    from showdown.rl.team_roles import ROLE_KEYS, team_role_counts
    c = dict(zip(ROLE_KEYS, team_role_counts(load_team_file('showdown/rl/data/ddqn_team.txt').mons)))
    assert c['support'] == 3 and c['physical'] == 3 and c['special'] == 3
    assert c['fake_out'] == 2 and c['tailwind'] == 1 and c['intimidate'] == 1 and c['redirect'] == 1


def test_selector_play_mock(tmp_path):
    from showdown.rl.ppo.agent import PPOAgent, PPOConfig
    from showdown.rl.qd_ppo.archive import Archive
    from showdown.rl.qd_ppo.selector_f2 import run_play
    from showdown.rl.qd_ppo.train2 import _state

    base = PPOAgent(PPOConfig(device='cpu'))
    base._ensure_torch()
    ck = tmp_path / 'c.pt'
    base.save(ck)
    arch = Archive()
    for i in range(3):
        a = PPOAgent(PPOConfig(device='cpu', seed=i + 1))
        a._ensure_torch()
        arch.try_insert(bd1=0.1 + 0.3 * i, bd2=0.5, fitness=0.1, state_dict=_state(a))
    arch.save(tmp_path / 'K.pt')
    rows = [{'name': f'K:{k[0]},{k[1]}', 'agent': 'K', 'cell': list(k), 'win_rate': 0.5 + 0.1 * n, 'fitness': 0.0}
            for n, k in enumerate(sorted(arch.cells))]
    (tmp_path / 'd.json').write_text(json.dumps({'rows': rows}))
    res = run_play(deploy_json=str(tmp_path / 'd.json'), archives={'K': str(tmp_path / 'K.pt')}, control=str(ck),
                   init=None, cfg_checkpoint=str(ck), out=tmp_path / 'f2', r=2, per_foe_fit=2, per_foe_eval=2,
                   mock=True, num_envs=2, chunk=3)
    assert res['repertoire'] == [rows[2]['name'], rows[1]['name']]
    assert set(res['eval']['wins']) == {rows[2]['name'], rows[1]['name'], 'control'}
    assert len(res['fit']['wins'][rows[1]['name']]) == 6
    # F2b: datos del selector en val y la parte eval reutilizada de la corrida anterior
    res2 = run_play(deploy_json=str(tmp_path / 'd.json'), archives={'K': str(tmp_path / 'K.pt')}, control=str(ck),
                    init=None, cfg_checkpoint=str(ck), out=tmp_path / 'f2b', r=2, per_foe_fit=1, per_foe_eval=2,
                    mock=True, num_envs=2, chunk=3, fit_split='val', fit_seed_block=7,
                    eval_from=str(tmp_path / 'f2' / 'results.json'))
    assert res2['fit_split'] == 'val' and res2['eval'] == res['eval']
    assert len(res2['fit']['wins'][rows[1]['name']]) == 3
