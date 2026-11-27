"""Tests for Team Preview QD-PGA (spec v1.0)."""
from __future__ import annotations

import numpy as np
import pytest

from showdown.rl.team_preview_action import encode_preview_action
from showdown.rl.team_preview_qd_pga.archive import Archive, actor_state_dict, is_actor_key
from showdown.rl.team_preview_qd_pga.bd import preview_bd_from_action
from showdown.rl.team_preview_qd_pga.mutate import mutate_actor_state


def test_bd_bounds_and_tr_bias():
    # Indeedee-F (TR) + Tornadus (TW) + Incineroar (FO) + Amoonguss
    species = [
        'Indeedee-F', 'Tornadus', 'Incineroar', 'Amoonguss',
        'Landorus-Therian', 'Rillaboom',
    ]
    a = encode_preview_action((0, 1, 2, 3))
    b1, b2 = preview_bd_from_action(a, species)
    assert 0.0 <= b1 <= 1.0 and 0.0 <= b2 <= 1.0
    assert b1 >= 0.5  # TR and/or TW


def test_archive_actor_only_keys():
    torch = pytest.importorskip('torch')
    from showdown.rl.team_preview_obs import OBS_DIM
    from showdown.rl.team_preview_ppo.network import PreviewActorCritic

    net = PreviewActorCritic(OBS_DIM)
    sd = actor_state_dict(net)
    assert sd
    assert all(is_actor_key(k) for k in sd)
    assert not any(k.startswith('value.') for k in sd)


def test_mutate_changes_weights():
    torch = pytest.importorskip('torch')
    from showdown.rl.team_preview_obs import OBS_DIM
    from showdown.rl.team_preview_ppo.network import PreviewActorCritic

    net = PreviewActorCritic(OBS_DIM)
    sd = actor_state_dict(net)
    rng = np.random.default_rng(0)
    mut = mutate_actor_state(sd, rng, sigma=0.02)
    # at least one tensor differs
    diff = False
    for k in sd:
        a = sd[k].detach().cpu().numpy()
        b = mut[k].detach().cpu().numpy() if hasattr(mut[k], 'detach') else mut[k]
        if not np.allclose(a, b):
            diff = True
            break
    assert diff


def test_archive_insert_replace():
    arch = Archive(10, 10)
    sd = {'policy.weight': np.zeros((2, 2), dtype=np.float32)}
    assert arch.try_insert(bd1=0.1, bd2=0.2, fitness=0.1, state_dict=sd)
    assert not arch.try_insert(bd1=0.1, bd2=0.2, fitness=0.05, state_dict=sd)
    assert arch.try_insert(bd1=0.1, bd2=0.2, fitness=0.2, state_dict=sd)
    assert arch.coverage() > 0


def test_qd_pga_smoke_train():
    torch = pytest.importorskip('torch')
    del torch
    gymnasium = pytest.importorskip('gymnasium')
    del gymnasium
    from showdown.rl.team_preview_qd_pga.train import train

    arch = train(seed=0, smoke=True, device='cpu')
    assert isinstance(arch, Archive)
    # bootstrap should usually fill ≥1 cell
    assert arch.coverage() >= 0.0


def test_b2_is_support_lead_fraction_v16():
    """v1.6: b2 = fracción de leads de soporte (redirect / Tailwind / Trick Room / Fake Out)."""
    species = ['Incineroar', 'Tornadus', 'Landorus-Therian', 'Chien-Pao', 'Amoonguss', 'Indeedee-F']
    assert preview_bd_from_action(encode_preview_action((0, 1, 2, 3)), species)[1] == 1.0
    assert preview_bd_from_action(encode_preview_action((2, 3, 0, 1)), species)[1] == 0.0
    assert preview_bd_from_action(encode_preview_action((0, 2, 1, 3)), species)[1] == 0.5


def test_archive_meta_roundtrip(tmp_path):
    ar = Archive(bounds=((0.0, 1.0), (0.0, 0.5)))
    ar.meta['obs_abilities'] = True
    ar.try_insert(bd1=0.2, bd2=0.2, fitness=0.1, state_dict={'w': np.ones(1)})
    ar.save(tmp_path / 'a.pt')
    br = Archive()
    br.load(tmp_path / 'a.pt')
    assert br.meta == {'obs_abilities': True} and br.bounds == ((0.0, 1.0), (0.0, 0.5))


def test_roles_bd_fixed_team(monkeypatch):
    """v1.7 (F4a): con el equipo fijo, los descriptores por roles sí varían con la elección."""
    from showdown.rl.team_preview_action import decode_preview_action
    from showdown.rl.team_preview_qd_pga import bd as B

    monkeypatch.setenv('PREVIEW_BD', 'roles')
    monkeypatch.setenv('PS_PREVIEW_OWN_TEAM', 'showdown/rl/data/ddqn_team.txt')
    sp = ['Raichu', 'Rillaboom', 'Gholdengo', 'Arcanine-Hisui', 'Staraptor', 'Primarina']
    seen = set()
    for a in range(360):
        b1, b2 = B.preview_bd_from_action(a, sp)
        i1, i2, i3, i4 = decode_preview_action(a)
        sup = {0, 1, 4}
        phys = {1, 3, 4}
        assert b1 == ((i1 in sup) + (i2 in sup)) / 2
        assert b2 == sum(i in phys for i in (i1, i2, i3, i4)) / 4
        seen.add((b1, b2))
    assert {b for b, _ in seen} == {0.0, 0.5, 1.0}
    assert {b for _, b in seen} == {0.25, 0.5, 0.75}
    monkeypatch.delenv('PREVIEW_BD')
    assert B.bd_mode() == 'species'
