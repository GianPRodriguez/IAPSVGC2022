"""Tests for status/secondary effect scoring and accuracy scaling."""
from __future__ import annotations

from showdown.battle_bots.greedy_solo.greedy import CalcPoke
from showdown.battle_bots.move_mechanics import (
    STATUS_VALUE,
    accuracy_scale_damage,
    expected_effect_score,
    move_accuracy,
    powder_can_affect,
    status_can_affect,
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


def test_move_accuracy_from_dex_and_overrides():
    assert move_accuracy('Thunderbolt') == 1.0
    assert abs(move_accuracy('Stone Edge') - 0.8) < 1e-9
    assert abs(move_accuracy('Focus Blast') - 0.7) < 1e-9
    assert abs(move_accuracy('Sleep Powder') - 0.75) < 1e-9
    assert move_accuracy('Dire Claw') == 1.0
    assert abs(accuracy_scale_damage(100.0, 'Stone Edge') - 80.0) < 1e-9


def test_sleep_powder_values_grass_immunity():
    atk = _mon('breloom', types=('Grass', 'Fighting'))
    normal = _mon('charizard', types=('Fire', 'Flying'))
    grass = _mon('rillaboom', types=('Grass',))
    sc_ok = expected_effect_score('Sleep Powder', attacker=atk, defender=normal)
    sc_grass = expected_effect_score('Sleep Powder', attacker=atk, defender=grass)
    assert sc_ok.score > 30.0  # 55 * 0.75
    assert sc_grass.score == 0.0
    assert not powder_can_affect(grass)


def test_spore_beats_chip_magnitude():
    atk = _mon('breloom', types=('Grass', 'Fighting'))
    foe = _mon('incineroar', types=('Fire', 'Dark'))
    spore = expected_effect_score('Spore', attacker=atk, defender=foe)
    # Spore 100% sleep ≈ 55 > typical mid chip after accuracy.
    assert spore.score >= STATUS_VALUE['slp'] * 0.99


def test_thunder_wave_limber_and_ground():
    atk = _mon('pikachu', types=('Electric',))
    limber = _mon('persian', types=('Normal',), ability='Limber')
    ground = _mon('garchomp', types=('Dragon', 'Ground'))  # TWave can hit Ground in gen6+
    # Electric types immune to para; Limber immune.
    assert not status_can_affect('par', defender=_mon('zapdos', types=('Electric', 'Flying')))
    assert expected_effect_score('Thunder Wave', attacker=atk, defender=limber).score == 0.0
    # Ground is NOT immune to Thunder Wave in modern gens (ignoreImmunity).
    sc = expected_effect_score('Thunder Wave', attacker=atk, defender=ground)
    assert sc.score > 0.0


def test_dire_claw_secondary_expectation():
    atk = _mon('sneasler', types=('Fighting', 'Poison'))
    foe = _mon('fluttermane', types=('Ghost', 'Fairy'))
    sc = expected_effect_score('Dire Claw', attacker=atk, defender=foe)
    # 50% * average(psn, par, slp) values
    expected = 0.5 * (STATUS_VALUE['psn'] + STATUS_VALUE['par'] + STATUS_VALUE['slp']) / 3.0
    assert abs(sc.score - expected) < 1e-6


def test_nuzzle_guaranteed_para_on_hit():
    atk = _mon('morpeko', types=('Electric', 'Dark'))
    foe = _mon('ursaluna', types=('Ground', 'Normal'))
    sc = expected_effect_score('Nuzzle', attacker=atk, defender=foe)
    assert abs(sc.score - STATUS_VALUE['par']) < 1e-6


def test_already_statused_no_extra_value():
    atk = _mon('breloom', types=('Grass', 'Fighting'))
    foe = _mon('incineroar', types=('Fire', 'Dark'), status='brn')
    assert expected_effect_score('Spore', attacker=atk, defender=foe).score == 0.0


def test_fakeout_skips_flinch_double_count():
    atk = _mon('incineroar', types=('Fire', 'Dark'))
    foe = _mon('rillaboom', types=('Grass',))
    # Fake Out flinch is scored elsewhere; effect score must be 0 here.
    assert expected_effect_score('Fake Out', attacker=atk, defender=foe).score == 0.0
