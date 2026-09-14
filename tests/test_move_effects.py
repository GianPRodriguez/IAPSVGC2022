"""Tests: schema MoveEffectFeatures + scoring extendido (NN / expectimax)."""
from __future__ import annotations

from showdown.battle_bots.greedy_solo.greedy import CalcPoke
from showdown.battle_bots.move_mechanics import (
    BOOST_SELF_UNIT,
    STATUS_VALUE,
    expected_effect_score,
    expected_move_effect_score,
    feature_keys,
    is_protect_move,
    parse_move_effects,
)


def _mon(species: str, *, types=(), ability=None, status='', item=None, sub=False):
    return CalcPoke(
        species=species,
        types=tuple(types),
        ability=ability,
        status=status,
        item=item,
        has_substitute=sub,
        stats={'hp': 100, 'atk': 100, 'def': 100, 'spa': 100, 'spd': 100, 'spe': 100},
        cur_hp=100,
    )


def test_feature_keys_stable_length():
    keys = feature_keys()
    f = parse_move_effects('Flamethrower')
    assert len(keys) == len(f.to_dense_vector())
    assert len(f.to_feature_dict()) == len(keys)


def test_parse_sample_moves():
    bulk = parse_move_effects('Bulk Up')
    assert bulk.self_boosts.get('atk', 0) >= 1
    assert bulk.self_boosts.get('def', 0) >= 1

    charm = parse_move_effects('Charm')
    assert charm.foe_boosts.get('atk', 0) <= -2

    vs = parse_move_effects('Volt Switch')
    assert vs.self_switch is True

    bb = parse_move_effects('Brave Bird')
    assert bb.recoil_frac > 0.2

    wow = parse_move_effects('Will-O-Wisp')
    assert wow.primary_status == 'brn'

    taunt = parse_move_effects('Taunt')
    assert taunt.volatile_status == 'taunt'

    web = parse_move_effects('Sticky Web')
    assert 'stickyweb' in (web.side_condition or '')

    ks = parse_move_effects("King's Shield")
    assert ks.is_protect or is_protect_move("King's Shield")
    assert is_protect_move('Protect')

    fm = parse_move_effects('Follow Me')
    assert fm.priority >= 2 or fm.move_id == 'followme'


def test_regression_status_effects_floor():
    atk = _mon('breloom', types=('Grass', 'Fighting'))
    foe = _mon('charizard', types=('Fire', 'Flying'))
    old = expected_effect_score('Sleep Powder', attacker=atk, defender=foe)
    new = expected_move_effect_score('Sleep Powder', attacker=atk, defender=foe)
    assert new.score >= old.score - 1e-9
    assert new.score > 30.0

    ft = expected_effect_score('Flamethrower', attacker=atk, defender=foe)
    ft2 = expected_move_effect_score('Flamethrower', attacker=atk, defender=foe)
    assert ft2.score >= ft.score - 1e-9

    swipe = expected_effect_score(
        'Breaking Swipe', attacker=atk, defender=foe,
    )
    swipe2 = expected_move_effect_score(
        'Breaking Swipe', attacker=atk, defender=foe,
    )
    assert swipe2.score >= swipe.score - 1e-9


def test_scoring_setup_charm_pivot_heal_taunt():
    atk = _mon('incineroar', types=('Fire', 'Dark'))
    foe = _mon('rillaboom', types=('Grass',))
    base = 4.0  # _STATUS_BASE

    bulk = expected_move_effect_score('Bulk Up', attacker=atk, defender=None)
    assert bulk.score > base
    assert bulk.score >= BOOST_SELF_UNIT * 1.5  # atk+def

    charm = expected_move_effect_score('Charm', attacker=atk, defender=foe)
    assert charm.score > base

    pivot = expected_move_effect_score('Volt Switch', attacker=atk, defender=foe)
    # damage secundarios 0 + pivot bonus * acc
    assert pivot.score >= 10.0

    roost = expected_move_effect_score('Roost', attacker=atk, defender=None)
    assert roost.score > base

    taunt = expected_move_effect_score('Taunt', attacker=atk, defender=foe)
    assert taunt.score > base


def test_draco_meteor_self_drop_penalty():
    atk = _mon('latios', types=('Dragon', 'Psychic'))
    foe = _mon('incineroar', types=('Fire', 'Dark'))
    feats = parse_move_effects('Draco Meteor')
    assert feats.self_boosts.get('spa', 0) <= -2
    sc = expected_move_effect_score('Draco Meteor', attacker=atk, defender=foe)
    # Con self-drop, score de efectos (sin daño) no es enorme positivo por setup
    assert sc.score < 5.0  # penalti ~8, sin secundario fuerte


def test_belly_drum_hp_cost_parsed():
    feats = parse_move_effects('Belly Drum')
    assert feats.hp_cost_frac >= 0.49
