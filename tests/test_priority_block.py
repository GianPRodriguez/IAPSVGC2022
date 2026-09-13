"""Tests for priority blocking (Psychic Terrain, Armor Tail / Dazzling)."""
from __future__ import annotations

from showdown.battle_bots.greedy_solo.greedy import CalcPoke, MoveOption, _actions_from_ranges
from showdown.battle_bots.greedy_joint.joint import _atoms_from_side
from showdown.battle_bots.move_mechanics import (
    effective_priority,
    is_grounded,
    priority_move_fails,
)


def _mon(**kwargs) -> CalcPoke:
    base = dict(
        nature='Jolly',
        cur_hp=150,
        stats={'hp': 150, 'atk': 150, 'def': 100, 'spa': 100, 'spd': 100, 'spe': 150},
    )
    base.update(kwargs)
    return CalcPoke(**base)


def test_extreme_speed_priority_value():
    assert effective_priority('Extreme Speed') > 0
    assert effective_priority('Thunderbolt') == 0


def test_psychic_terrain_blocks_priority_vs_grounded():
    attacker = _mon(species='Arcanine', types=('Fire',))
    grounded = _mon(species='Incineroar', types=('Fire', 'Dark'))
    flyer = _mon(species='Corviknight', types=('Flying', 'Steel'))
    assert is_grounded(grounded)
    assert not is_grounded(flyer)

    assert priority_move_fails(
        'Extreme Speed',
        attacker=attacker,
        defender=grounded,
        terrain='psychicterrain',
    )
    assert not priority_move_fails(
        'Extreme Speed',
        attacker=attacker,
        defender=flyer,
        terrain='psychicterrain',
    )
    assert not priority_move_fails(
        'Extreme Speed',
        attacker=attacker,
        defender=grounded,
        terrain=None,
    )


def test_armor_tail_blocks_priority_even_vs_flyer():
    attacker = _mon(species='Arcanine', types=('Fire',))
    flyer = _mon(species='Corviknight', types=('Flying', 'Steel'), ability='Armor Tail')
    assert priority_move_fails(
        'Extreme Speed',
        attacker=attacker,
        defender=flyer,
        defender_side=[flyer],
        terrain=None,
    )


def test_mold_breaker_bypasses_armor_tail_not_terrain():
    breaker = _mon(species='Haxorus', types=('Dragon',), ability='Mold Breaker')
    armored = _mon(species='Farigiraf', types=('Normal', 'Psychic'), ability='Armor Tail')
    grounded = _mon(species='Incineroar', types=('Fire', 'Dark'))
    assert not priority_move_fails(
        'Extreme Speed',
        attacker=breaker,
        defender=armored,
        defender_side=[armored],
    )
    assert priority_move_fails(
        'Extreme Speed',
        attacker=breaker,
        defender=grounded,
        terrain='psychicterrain',
    )


def test_atoms_skip_extreme_speed_under_psychic_terrain():
    atk = _mon(species='Arcanine', types=('Fire',))
    foe = _mon(species='Incineroar', types=('Fire', 'Dark'))
    move = MoveOption(1, 'Extreme Speed', 'normal', 'Physical')
    other = MoveOption(2, 'Flare Blitz', 'normal', 'Physical')
    meta = [
        ('L', move, 1, foe, 0),
        ('L', other, 1, foe, 0),
    ]
    atoms = _atoms_from_side(
        [move, other], meta, [(120, 140), (80, 100)], [(1, foe)],
        can_mega=False, attacker=atk, terrain='psychicterrain',
    )
    names = {a.move_name for a in atoms if a.utility == 'damage'}
    assert 'Extreme Speed' not in names
    assert 'Flare Blitz' in names


def test_actions_skip_priority_under_psychic_terrain():
    atk = _mon(species='Arcanine', types=('Fire',))
    foe = _mon(species='Incineroar', types=('Fire', 'Dark'))
    move = MoveOption(1, 'Extreme Speed', 'normal', 'Physical')
    other = MoveOption(2, 'Flare Blitz', 'normal', 'Physical')
    meta = [
        ('L', move, 1, foe, 0),
        ('L', other, 1, foe, 0),
    ]
    actions = _actions_from_ranges(
        [move, other], meta, [(120, 140), (80, 100)],
        can_mega=False, attacker=atk, terrain='psychicterrain',
        foes_for_priority=[foe],
    )
    by_slot = {a.move_slot: a for a in actions}
    assert 1 not in by_slot  # Extreme Speed descartado
    assert 2 in by_slot
    assert by_slot[2].damage_pct > 0


def test_fakeout_also_blocked_by_psychic_terrain():
    atk = _mon(species='Incineroar', types=('Fire', 'Dark'), ability='Intimidate', just_switched=True)
    foe = _mon(species='Rillaboom', types=('Grass',))
    move = MoveOption(1, 'Fake Out', 'normal', 'Physical')
    atoms = _atoms_from_side(
        [move], [('L', move, 1, foe, 0)], [(40, 50)], [(1, foe)],
        can_mega=False, attacker=atk, terrain='psychicterrain',
    )
    assert all(a.utility != 'fakeout' for a in atoms)
    assert not any(a.move_name == 'Fake Out' and a.utility == 'damage' for a in atoms)
