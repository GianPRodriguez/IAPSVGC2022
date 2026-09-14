"""Entry abilities on switch: Intimidate, weather/terrain, Screen Cleaner, etc."""
from __future__ import annotations

from showdown.battle_bots.greedy_solo.greedy import CalcPoke
from showdown.battle_bots.move_mechanics import (
    INTIMIDATE_PER_FOE,
    entry_ability_breakdown,
    entry_ability_switch_value,
    intimidate_can_affect,
)


def _mon(species, *, ability=None, types=(), boosts=None, sub=False):
    return CalcPoke(
        species=species,
        ability=ability,
        types=tuple(types),
        boosts=dict(boosts or {}),
        has_substitute=sub,
        stats={'hp': 100, 'atk': 100, 'def': 100, 'spa': 100, 'spd': 100, 'spe': 100},
        cur_hp=100,
    )


def test_intimidate_two_foes():
    incin = _mon('incineroar', ability='Intimidate', types=('Fire', 'Dark'))
    f1 = _mon('fluttermane', types=('Ghost', 'Fairy'))
    f2 = _mon('rillaboom', types=('Grass',))
    v = entry_ability_switch_value(incin, [f1, f2])
    assert abs(v - 2 * INTIMIDATE_PER_FOE) < 1e-6


def test_intimidate_skips_immune_and_defiant():
    incin = _mon('incineroar', ability='Intimidate')
    clear = _mon('registeel', ability='Clear Body')
    defiant = _mon('baxcalibur', ability='Defiant')
    normal = _mon('ursaluna', ability='Guts')
    assert not intimidate_can_affect(clear)
    assert not intimidate_can_affect(defiant)
    assert intimidate_can_affect(normal)
    v = entry_ability_switch_value(incin, [clear, defiant, normal])
    assert abs(v - INTIMIDATE_PER_FOE) < 1e-6


def test_intimidate_skips_substitute_and_floor():
    incin = _mon('incineroar', ability='Intimidate')
    sub = _mon('mimikyu', sub=True)
    floored = _mon('charizard', boosts={'atk': -6})
    assert not intimidate_can_affect(sub)
    assert not intimidate_can_affect(floored)
    assert entry_ability_switch_value(incin, [sub, floored]) == 0.0


def test_drizzle_sets_rain_once():
    pelo = _mon('pelipper', ability='Drizzle')
    assert entry_ability_switch_value(pelo, [], weather=None) > 0
    assert entry_ability_switch_value(pelo, [], weather='RainDance') == 0.0
    assert entry_ability_switch_value(pelo, [], weather='rain') == 0.0


def test_electric_surge_and_already_up():
    tapu = _mon('tapukoko', ability='Electric Surge')
    assert entry_ability_switch_value(tapu, [], terrain=None) > 0
    assert entry_ability_switch_value(
        tapu, [], terrain='electricterrain',
    ) == 0.0


def test_screen_cleaner_needs_screens():
    mrw = _mon('mr-rime', ability='Screen Cleaner')
    assert entry_ability_switch_value(mrw, []) == 0.0
    v = entry_ability_switch_value(
        mrw, [], opp_reflect=True, opp_light_screen=True,
    )
    assert v > 0
    parts = entry_ability_breakdown(
        mrw, [], opp_reflect=True, opp_light_screen=True,
    )
    assert 'screencleaner' in parts


def test_costar_copies_ally_boosts():
    flamigo = _mon('flamigo', ability='Costar')
    ally = _mon('dragonite', boosts={'atk': 2, 'spe': 1})
    flat = _mon('dragonite')
    assert entry_ability_switch_value(flamigo, [], ally=flat) == 0.0
    assert entry_ability_switch_value(flamigo, [], ally=ally) > 0


def test_dauntless_shield_self_boost():
    zam = _mon('zamazenta', ability='Dauntless Shield')
    assert entry_ability_switch_value(zam, []) > 0
    maxed = _mon('zamazenta', ability='Dauntless Shield', boosts={'def': 6})
    assert entry_ability_switch_value(maxed, []) == 0.0


def test_no_ability_zero():
    plain = _mon('pikachu', ability=None)
    assert entry_ability_switch_value(plain, [_mon('x')]) == 0.0
