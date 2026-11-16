"""Pool de rivales del entrenamiento: validación reservada, subconjuntos y equipos sintéticos (16-11-2026)."""
import json
import random
import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, '.')

from showdown.inference_sim import team_split as ts
from showdown.rl import foe_pool


def _fake(n):
    return [SimpleNamespace(name=f't{i}', source_url=f'u{i}', mons=[]) for i in range(n)]


@pytest.fixture
def clean_env(monkeypatch):
    for k in ('RL_HOLDOUT_VAL', 'RL_TRAIN_SUBSET', 'RL_FOE_SYNTH', 'RL_FOE_SYNTH_FILE'):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _patch_pool(monkeypatch, teams):
    import showdown.inference_sim.team_factory as tf
    monkeypatch.setattr(tf, 'active_teams', lambda split=None: list(teams))


def test_default_sampling_identical_to_previous(clean_env):
    teams = _fake(20)
    _patch_pool(clean_env, teams)
    from showdown.rl.pokeenv_backend import _default_team_metas
    for seed in range(30):
        rng = np.random.default_rng(seed)  # lógica anterior, copiada
        i, j = int(rng.integers(0, 20)), int(rng.integers(0, 19))
        j += j >= i
        a, b = _default_team_metas(seed=seed)
        assert (a, b) == (teams[i], teams[j])


def test_holdout_switches_training_and_validation_split(clean_env):
    clean_env.setenv('PS_TEAM_SPLIT', 'train')
    assert foe_pool.training_split() == 'train' and foe_pool.validation_split() is None
    clean_env.setenv('RL_HOLDOUT_VAL', '1')
    assert foe_pool.training_split() == 'fit' and foe_pool.validation_split() == 'val'
    clean_env.setenv('PS_TEAM_SPLIT', 'eval')  # diagnóstico con eval: no se toca
    assert foe_pool.training_split() == 'eval'


def test_fit_and_val_keys(clean_env):
    data = {'train': ['a', 'b', 'c', 'd'], 'eval': ['e'], 'val': ['b']}
    clean_env.setattr(ts, '_load', lambda: data)
    assert ts.split_keys('val') == {'b'} and ts.split_keys('fit') == {'a', 'c', 'd'}
    assert ts.split_keys('train') == {'a', 'b', 'c', 'd'}
    clean_env.setattr(ts, '_load', lambda: {'train': ['a'], 'eval': ['e']})
    with pytest.raises(RuntimeError):
        ts.split_keys('fit')


def test_subset_nested_and_order_independent():
    teams = _fake(60)
    s50, s20 = foe_pool.subset(teams, 50), foe_pool.subset(teams, 20)
    assert len(s50) == 50 and all(t in s50 for t in s20)
    assert foe_pool.subset(list(reversed(teams)), 20) == s20
    assert foe_pool.subset(teams, 0) == teams


def test_subset_only_applies_to_training_splits(clean_env):
    _patch_pool(clean_env, _fake(40))
    clean_env.setenv('RL_TRAIN_SUBSET', '10')
    assert len(foe_pool.base_pool('fit')) == 10 and len(foe_pool.base_pool('train')) == 10
    assert len(foe_pool.base_pool('eval')) == 40 and len(foe_pool.base_pool('val')) == 40


def test_synthetic_mix_only_in_training(clean_env, tmp_path):
    synth = tmp_path / 's.json'
    synth.write_text(json.dumps({'teams': []}), encoding='utf-8')
    fake_synth = _fake(3)
    clean_env.setattr(foe_pool, 'synth_pool', lambda: fake_synth)
    real = object()
    rng = np.random.default_rng(0)
    assert foe_pool.maybe_synthetic_foe(rng, 'fit', real) is real          # apagado
    clean_env.setenv('RL_FOE_SYNTH', '1')
    assert foe_pool.maybe_synthetic_foe(rng, 'fit', real) in fake_synth
    assert foe_pool.maybe_synthetic_foe(rng, 'eval', real) is real         # evaluación intacta
    assert foe_pool.maybe_synthetic_foe(rng, 'val', real) is real
    clean_env.setenv('RL_FOE_SYNTH', '0.5')
    hits = sum(foe_pool.maybe_synthetic_foe(rng, 'fit', real) is not real for _ in range(2000))
    assert 900 < hits < 1100
    clean_env.setenv('RL_FOE_SYNTH', '2')
    with pytest.raises(ValueError):
        foe_pool.synth_prob()


# ---------------------------------------------------------------- recombinación
def _team(name, specs):
    from showdown.inference_sim.team_factory import MonSet, TeamMeta
    mons = [MonSet(species=s, item=it, ability='Intimidate', nature='Jolly', moves=['Protect', 'Tackle'],
                   sp={'hp': 2, 'atk': 32, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 32}) for s, it in specs]
    return TeamMeta(name=name, source_url=name, mons=mons)


BASE = [('Incineroar', 'Sitrus Berry'), ('Rillaboom', 'Assault Vest'), ('Gholdengo', 'Choice Specs'),
        ('Arcanine', 'Leftovers'), ('Garchomp', 'Life Orb'), ('Primarina', 'Mystic Water')]
DONORS = [('Arcanine-Hisui', 'Choice Band'), ('Kingambit', 'Sitrus Berry'), ('Dragonite', 'Choice Band'),
          ('Talonflame', 'Sharp Beak'), ('Sinistcha', 'Rocky Helmet'), ('Basculegion', 'Focus Sash')]


def test_recombine_respects_species_and_item_clause():
    from showdown.rl.synth_teams import dex_num, recombine
    assert dex_num('Arcanine') == dex_num('Arcanine-Hisui')
    base, donor_team = _team('base', BASE), _team('donor', DONORS)
    donors = base.mons + donor_team.mons
    rng = random.Random(0)
    for _ in range(200):
        t, swapped = recombine(base, donors, rng.choice((1, 2, 3)), rng)
        if t is None:
            continue
        nums = [dex_num(m.species) for m in t.mons]
        items = [m.item for m in t.mons]
        assert len(set(nums)) == 6 and len(set(items)) == 6
        assert sum(m.species != b.species for m, b in zip(t.mons, base.mons)) == len(swapped)


def test_generate_excludes_forbidden_combinations_and_keeps_full_sets():
    from showdown.rl.synth_teams import generate
    fit = [_team('base', BASE), _team('donor', DONORS)]
    forbidden = {ts.clone_key(t) for t in fit}
    teams, meta, _ = generate(fit, forbidden, n=40, seed=1)
    assert teams and len(teams) == len(meta)
    keys = [ts.clone_key(t) for t in teams]
    assert len(set(keys)) == len(keys) and not (set(keys) & forbidden)
    real_sets = {json.dumps(asdict(m), sort_keys=True) for t in fit for m in t.mons}
    for t in teams:  # cada Pokémon es un set real completo, sin mezclar piezas
        assert all(json.dumps(asdict(m), sort_keys=True) in real_sets for m in t.mons)
    again, _, _ = generate(fit, forbidden, n=40, seed=1)
    assert [ts.clone_key(t) for t in again] == keys  # determinista


def test_synth_file_roundtrip(clean_env, tmp_path):
    from showdown.rl.synth_teams import generate
    fit = [_team('base', BASE), _team('donor', DONORS)]
    teams, meta, _ = generate(fit, set(), n=5, seed=2)
    rows = [{'name': t.name, 'source_url': t.source_url, **m, 'mons': [asdict(x) for x in t.mons]}
            for t, m in zip(teams, meta)]
    p = tmp_path / 'synth.json'
    p.write_text(json.dumps({'teams': rows}), encoding='utf-8')
    clean_env.setenv('RL_FOE_SYNTH_FILE', str(p))
    foe_pool._load_synth.cache_clear()
    loaded = foe_pool.synth_pool()
    assert [t.name for t in loaded] == [t.name for t in teams]
    assert [m.species for m in loaded[0].mons] == [m.species for m in teams[0].mons]
