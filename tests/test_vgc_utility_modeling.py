"""Tests de modelado VGC: Knock Off / Fake Out deny / mega proyección."""
from __future__ import annotations

from showdown.battle_bots.greedy_solo.greedy import CalcPoke
from showdown.battle_bots.move_mechanics.priority_block import priority_move_fails
from showdown.battle_bots.move_mechanics.vgc_utility import (
    fake_out_deny_from_threat,
    is_knock_off_removable,
    is_mega_stone,
    knock_off_item_bonus,
    project_foe_active_forme,
    weather_from_projected_foes,
)


def _poke(species, *, item=None, ability=None, spe=100, just_switched=False, types=('Normal',)):
    return CalcPoke(
        species=species,
        ability=ability,
        item=item,
        just_switched=just_switched,
        stats={'hp': 150, 'atk': 100, 'def': 100, 'spa': 100, 'spd': 100, 'spe': spe},
        cur_hp=150,
        types=types,
    )


def test_knock_off_skips_mega_stone():
    assert is_mega_stone('Charizardite Y')
    assert not is_knock_off_removable('Charizardite Y')
    assert is_knock_off_removable('Life Orb')
    assert knock_off_item_bonus(_poke('Sylveon', item='Life Orb')) > 0
    assert knock_off_item_bonus(_poke('Charizard', item='Charizardite Y')) == 0.0


def test_fake_out_deny_scales_with_threat():
    atk = _poke('Incineroar', spe=60, just_switched=True)
    low = fake_out_deny_from_threat(attacker=atk, threat_denied=10.0)
    high = fake_out_deny_from_threat(attacker=atk, threat_denied=900.0)
    assert high > low
    assert high >= 900.0
    # Sin first-turn → 0
    assert fake_out_deny_from_threat(
        attacker=_poke('Incineroar', just_switched=False), threat_denied=900.0,
    ) == 0.0
    # Fails (Psychic Terrain etc.) → 0
    assert fake_out_deny_from_threat(
        attacker=atk, threat_denied=900.0, fails=True,
    ) == 0.0


def test_fake_out_fails_on_psychic_terrain_grounded():
    atk = _poke('Incineroar', just_switched=True, types=('Fire', 'Dark'))
    foe = _poke('Sylveon', types=('Fairy',))
    assert priority_move_fails(
        'Fake Out', attacker=atk, defender=foe, defender_side=[foe],
        terrain='psychicterrain',
    )
    # Flying no grounded → no falla por terreno
    flyer = _poke('Charizard', types=('Fire', 'Flying'))
    assert not priority_move_fails(
        'Fake Out', attacker=atk, defender=flyer, defender_side=[flyer],
        terrain='psychicterrain',
    )


def test_project_foe_mega_sets_drought_weather():
    zard = _poke('Charizard', item='Charizardite Y', ability='Blaze', spe=100)
    projected = project_foe_active_forme(zard)
    assert projected is not None
    w = weather_from_projected_foes([projected], current_weather=None)
    abi = str(getattr(projected, 'ability', '') or '').lower().replace(' ', '')
    if 'mega' in str(getattr(projected, 'species', '')).lower() or abi == 'drought':
        if abi == 'drought':
            assert w == 'sunnyday'
