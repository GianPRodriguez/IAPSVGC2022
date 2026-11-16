"""Separación de equipos 200/100 (checklist DDQN, punto 11)."""
import json
import sys

import pytest

sys.path.insert(0, '.')

from showdown.inference_sim import team_split as TS


def _data():
    return json.loads(TS.SPLIT_PATH.read_text(encoding='utf-8'))


def test_split_file_is_200_100_and_disjoint():
    d = _data()
    assert len(d['train']) == 200 and len(d['eval']) == 100
    assert not set(d['train']) & set(d['eval'])


def test_split_covers_the_json_pool():
    raw = json.loads((TS.SPLIT_PATH.parent / 'metavgc_regmc_teams.json').read_text(encoding='utf-8'))
    keys = {TS.team_key(t) for t in raw}
    d = _data()
    assert set(d['train']) | set(d['eval']) <= keys


def test_split_is_reproducible():
    class M:
        def __init__(self, s):
            self.species = s

    class T:
        def __init__(self, k, species):
            self.source_url, self.name = k, k
            self.mons = [M(s) for s in species]
    teams = [T(f'u{i}', [f'mon{i}', 'Incineroar']) for i in range(30)]
    a, b = TS.make_split(teams, n_train=20), TS.make_split(list(reversed(teams)), n_train=20)
    assert a == b and len(a['train']) == 20 and len(a['eval']) == 10


def test_clones_stay_in_the_same_split():
    """Equipos con las mismas 6 especies nunca quedan repartidos entre train y eval."""
    raw = json.loads((TS.SPLIT_PATH.parent / 'metavgc_regmc_teams.json').read_text(encoding='utf-8'))
    d = _data()
    tr, ev = set(d['train']), set(d['eval'])
    from showdown.inference_sim.team_factory import canonicalize_species

    def key(t):
        return '|'.join(sorted(''.join(c for c in canonicalize_species(m['species']).lower() if c.isalnum())
                               for m in t['mons']))
    g_tr = {key(t) for t in raw if TS.team_key(t) in tr}
    g_ev = {key(t) for t in raw if TS.team_key(t) in ev}
    assert g_tr and g_ev and not (g_tr & g_ev)


def test_env_selects_split(monkeypatch):
    monkeypatch.delenv('PS_TEAM_SPLIT', raising=False)
    assert TS.current_split() == 'all' and TS.prior_split() == 'all'
    monkeypatch.setenv('PS_TEAM_SPLIT', 'eval')
    assert TS.prior_split() == 'train'   # la evaluación usa el meta conocido (train)
    monkeypatch.setenv('PS_TEAM_SPLIT', 'nope')
    with pytest.raises(ValueError):
        TS.current_split()


def test_prior_and_worlds_follow_split(monkeypatch):
    from showdown import meta_sp_prior as MP
    from showdown.battle_bots.expectimax_bayesian import belief as BL
    monkeypatch.setenv('PS_TEAM_SPLIT', 'all')
    n_all = MP.load_meta_sp_prior(force=True)
    w_all = BL._load_meta(force=True)
    monkeypatch.setenv('PS_TEAM_SPLIT', 'eval')
    n_train = MP.load_meta_sp_prior()
    w_train = BL._load_meta()
    assert 0 < n_train < n_all and 0 < w_train < w_all
    # ~2/3 de los Pokémon (200 de 300 equipos)
    assert 0.55 < n_train / n_all < 0.78
    monkeypatch.setenv('PS_TEAM_SPLIT', 'all')
    assert MP.load_meta_sp_prior() == n_all


def test_rl_backend_defaults_to_train(monkeypatch):
    from showdown.rl.pokeenv_backend import rl_team_split
    monkeypatch.delenv('PS_TEAM_SPLIT', raising=False)
    assert rl_team_split() == 'train'
    monkeypatch.setenv('PS_TEAM_SPLIT', 'all')
    assert rl_team_split() == 'all'


def test_conclusion_uses_eval_teams():
    from showdown.inference_sim.conclusion_eval.constants import (
        CONCLUSION_TEAM_SPLIT, DEFAULT_SNAPSHOT_N)
    assert CONCLUSION_TEAM_SPLIT == 'eval' and DEFAULT_SNAPSHOT_N == 78 and DEFAULT_SNAPSHOT_N % 2 == 0
