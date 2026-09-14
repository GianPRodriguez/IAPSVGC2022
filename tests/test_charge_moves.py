"""Tests for two-turn charge move timing (Electro Shot, Solar Beam, …)."""
from __future__ import annotations

import asyncio

from showdown.battle_bots.move_mechanics import (
    CHARGE_MOVES,
    RECHARGE_MOVES,
    charge_boost_score,
    gate_damage_range,
    is_charge_move,
    is_must_recharge_from_volatiles,
    is_recharge_move,
    is_two_turn_charging_from_volatiles,
    item_after_charge,
    power_herb_cost,
    recharge_lock_penalty,
    resolve_charge,
    resolve_recharge,
)
from showdown.battle_bots.greedy_solo.greedy import CalcPoke, MoveOption, _actions_from_ranges
from showdown.battle_bots.greedy_joint.joint import _atoms_from_side, _utility_kind
from showdown.damage_calc_bridge import set_damage_range_override


def test_charge_move_catalog_includes_archaludon_staples():
    assert is_charge_move('Electro Shot')
    assert is_charge_move('solarbeam')
    assert is_charge_move('Solar Blade')
    assert is_charge_move('Meteor Beam')
    assert is_charge_move('Geomancy')
    assert is_charge_move('Fly')
    assert 'electroshot' in CHARGE_MOVES
    assert not is_charge_move('Thunderbolt')


def test_electroshot_requires_two_turns_without_rain():
    res = resolve_charge('Electro Shot', weather=None, item=None)
    assert res.is_charge_move
    assert res.is_charging
    assert not res.fires_this_turn
    assert res.locked_next
    assert res.charge_boosts.get('spa') == 1
    assert gate_damage_range((100, 120), 'Electro Shot') is None


def test_electroshot_fires_in_rain():
    res = resolve_charge('electroshot', weather='RainDance')
    assert res.fires_this_turn
    assert not res.is_charging
    assert gate_damage_range((100, 120), 'electroshot', weather='raindance') == (100, 120)


def test_electroshot_power_herb_and_already_charging():
    herb = resolve_charge('Electro Shot', item='Power Herb')
    assert herb.fires_this_turn
    assert herb.consumes_power_herb
    assert item_after_charge('Power Herb', herb) is None
    assert power_herb_cost(herb) > 0
    locked = resolve_charge('Electro Shot', already_charging=True)
    assert locked.fires_this_turn
    assert not locked.is_charging
    assert not locked.consumes_power_herb
    # Rain dispara sin gastar Power Herb (orden PS).
    rain_herb = resolve_charge('Electro Shot', weather='raindance', item='Power Herb')
    assert rain_herb.fires_this_turn
    assert not rain_herb.consumes_power_herb
    assert item_after_charge('Power Herb', rain_herb) == 'Power Herb'


def test_solarbeam_sun_exception():
    dry = resolve_charge('Solar Beam')
    assert dry.is_charging and not dry.fires_this_turn
    sun = resolve_charge('Solar Beam', weather='sunnyday')
    assert sun.fires_this_turn
    blade = resolve_charge('solarblade', weather='DesolateLand')
    assert blade.fires_this_turn


def test_geomancy_and_fly_flags():
    geo = resolve_charge('Geomancy')
    assert geo.is_charging
    assert charge_boost_score(geo) == 6.0 * 6  # +2 spa/spd/spe
    fly = resolve_charge('Fly')
    assert fly.invulnerable
    assert fly.is_charging


def test_volatiles_detect_twoturnmove():
    assert is_two_turn_charging_from_volatiles({'twoturnmove': True})
    assert is_two_turn_charging_from_volatiles({'electroshot': 1})
    assert not is_two_turn_charging_from_volatiles({'substitute': True})
    assert not is_two_turn_charging_from_volatiles({})


def test_utility_kind_charge():
    move = MoveOption(1, 'Electro Shot', 'normal', 'Special')
    assert _utility_kind(move) == 'charge'
    assert _utility_kind(move, weather='raindance') == 'damage'
    assert _utility_kind(MoveOption(1, 'Geomancy', 'self', 'Status')) == 'charge'


def test_actions_gate_electroshot_without_rain():
    atk = CalcPoke(
        species='Archaludon', nature='Modest',
        stats={'hp': 150, 'atk': 100, 'def': 120, 'spa': 150, 'spd': 100, 'spe': 100},
        cur_hp=150, types=('Steel', 'Dragon'),
    )
    foe = CalcPoke(
        species='Incineroar', nature='Adamant',
        stats={'hp': 170, 'atk': 140, 'def': 110, 'spa': 80, 'spd': 100, 'spe': 80},
        cur_hp=170, types=('Fire', 'Dark'),
    )
    move = MoveOption(1, 'Electro Shot', 'normal', 'Special')
    meta = [('L', move, 1, foe, 0)]
    ranges = [(100, 120)]
    actions = _actions_from_ranges(
        [move], meta, ranges, can_mega=False, attacker=atk, weather=None,
    )
    assert len(actions) == 1
    assert actions[0].damage_pct == 0.0
    assert not actions[0].ko
    assert actions[0].score == charge_boost_score(resolve_charge('Electro Shot'))

    wet = _actions_from_ranges(
        [move], meta, ranges, can_mega=False, attacker=atk, weather='raindance',
    )
    assert wet[0].damage_pct > 0
    assert wet[0].ko is False or wet[0].damage_pct > 0


def test_atoms_gate_electroshot_without_rain():
    atk = CalcPoke(
        species='Archaludon', nature='Modest',
        stats={'hp': 150, 'atk': 100, 'def': 120, 'spa': 150, 'spd': 100, 'spe': 100},
        cur_hp=150, types=('Steel', 'Dragon'),
    )
    foe = CalcPoke(
        species='Incineroar', nature='Adamant',
        stats={'hp': 170, 'atk': 140, 'def': 110, 'spa': 80, 'spd': 100, 'spe': 80},
        cur_hp=170, types=('Fire', 'Dark'),
    )
    move = MoveOption(1, 'Electro Shot', 'normal', 'Special')
    meta = [('L', move, 1, foe, 0)]
    atoms = _atoms_from_side(
        [move], meta, [(100, 120)], [(1, foe)],
        can_mega=False, attacker=atk, weather=None,
    )
    assert atoms
    assert all(a.utility == 'charge' for a in atoms)
    assert all(a.total_damage_pct == 0.0 for a in atoms)

    wet = _atoms_from_side(
        [move], meta, [(100, 120)], [(1, foe)],
        can_mega=False, attacker=atk, weather='raindance',
    )
    assert wet
    assert wet[0].utility == 'damage'
    assert wet[0].total_damage_pct > 0


def test_joint_prefers_real_damage_over_dry_electroshot():
    """Sin lluvia, un hit inmediato debe ganar a Electro Shot fantasma."""
    from showdown.battle_bots.greedy_joint.joint import best_joint_double_action

    def fake_batch(reqs):
        out = []
        for req in reqs:
            move = str(req.get('move') or '').lower().replace(' ', '').replace('-', '')
            if move == 'electroshot':
                out.append((200, 240))  # calc mentiría; el gate debe anularlo
            else:
                out.append((60, 70))
        return out

    set_damage_range_override(fake_batch)
    try:
        left = CalcPoke(
            species='Archaludon', nature='Modest',
            stats={'hp': 150, 'atk': 80, 'def': 120, 'spa': 160, 'spd': 100, 'spe': 90},
            cur_hp=150, types=('Steel', 'Dragon'),
        )
        right = CalcPoke(
            species='Rillaboom', nature='Adamant',
            stats={'hp': 193, 'atk': 173, 'def': 115, 'spa': 72, 'spd': 100, 'spe': 109},
            cur_hp=193, types=('Grass',),
        )
        foes = [(1, CalcPoke(
            species='Pelipper', nature='Modest',
            stats={'hp': 140, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=140, types=('Water', 'Flying'),
        ))]
        best_l, best_r, _score = asyncio.run(
            best_joint_double_action(
                left, right,
                [MoveOption(1, 'Electro Shot', 'normal', 'Special'),
                 MoveOption(2, 'Flash Cannon', 'normal', 'Special')],
                [MoveOption(1, 'Wood Hammer', 'normal', 'Physical')],
                foes,
                weather=None,
            )
        )
        assert best_l is not None
        assert best_l.move_slot == 2  # Flash Cannon, no Electro Shot
    finally:
        set_damage_range_override(None)


def test_recharge_catalog_and_lock():
    assert is_recharge_move('Hyper Beam')
    assert is_recharge_move('Giga Impact')
    assert 'hyperbeam' in RECHARGE_MOVES
    assert not is_recharge_move('Electro Shot')
    res = resolve_recharge('Hyper Beam')
    assert res.must_recharge_next
    assert res.can_act
    assert recharge_lock_penalty(res, ko=False) > recharge_lock_penalty(res, ko=True) > 0
    locked = resolve_recharge('Thunderbolt', currently_recharging=True)
    assert locked.currently_recharging
    assert not locked.can_act
    assert is_must_recharge_from_volatiles({'mustrecharge': True})
    assert gate_damage_range((90, 110), 'Hyper Beam', must_recharge=True) is None


def test_actions_penalize_hyper_beam_and_gate_must_recharge():
    atk = CalcPoke(
        species='Regieleki', nature='Timid',
        stats={'hp': 120, 'atk': 80, 'def': 80, 'spa': 150, 'spd': 80, 'spe': 200},
        cur_hp=120, types=('Electric',),
    )
    foe = CalcPoke(
        species='Incineroar', nature='Adamant',
        stats={'hp': 170, 'atk': 140, 'def': 110, 'spa': 80, 'spd': 100, 'spe': 80},
        cur_hp=170, types=('Fire', 'Dark'),
    )
    hb = MoveOption(1, 'Hyper Beam', 'normal', 'Special')
    tb = MoveOption(2, 'Thunderbolt', 'normal', 'Special')
    meta = [
        ('L', hb, 1, foe, 0),
        ('L', tb, 1, foe, 0),
    ]
    ranges = [(100, 120), (100, 120)]
    actions = _actions_from_ranges(
        [hb, tb], meta, ranges, can_mega=False, attacker=atk, weather=None,
    )
    by_slot = {a.move_slot: a for a in actions}
    # Hyper Beam 90% acc vs Thunderbolt 100%: daño esperado menor + lock.
    assert by_slot[1].damage_pct < by_slot[2].damage_pct
    assert by_slot[1].score < by_slot[2].score  # Hyper Beam paga lock + acc

    atk.must_recharge = True
    locked_actions = _actions_from_ranges(
        [hb], [('L', hb, 1, foe, 0)], [(100, 120)],
        can_mega=False, attacker=atk, weather=None,
    )
    assert locked_actions[0].damage_pct == 0.0
    assert locked_actions[0].score == 0.0


def test_actions_power_herb_cost_vs_rain():
    atk = CalcPoke(
        species='Archaludon', nature='Modest', item='Power Herb',
        stats={'hp': 150, 'atk': 80, 'def': 120, 'spa': 160, 'spd': 100, 'spe': 90},
        cur_hp=150, types=('Steel', 'Dragon'),
    )
    foe = CalcPoke(
        species='Incineroar', nature='Adamant',
        stats={'hp': 170, 'atk': 140, 'def': 110, 'spa': 80, 'spd': 100, 'spe': 80},
        cur_hp=170, types=('Fire', 'Dark'),
    )
    move = MoveOption(1, 'Electro Shot', 'normal', 'Special')
    meta = [('L', move, 1, foe, 0)]
    ranges = [(100, 120)]
    with_herb = _actions_from_ranges(
        [move], meta, ranges, can_mega=False, attacker=atk, weather=None,
    )
    atk_rain = CalcPoke(
        species='Archaludon', nature='Modest', item='Power Herb',
        stats={'hp': 150, 'atk': 80, 'def': 120, 'spa': 160, 'spd': 100, 'spe': 90},
        cur_hp=150, types=('Steel', 'Dragon'),
    )
    with_rain = _actions_from_ranges(
        [move], meta, ranges, can_mega=False, attacker=atk_rain, weather='raindance',
    )
    assert with_herb[0].damage_pct > 0
    assert with_rain[0].damage_pct > 0
    assert with_rain[0].score > with_herb[0].score  # rain no gasta herb


def test_joint_atoms_recharge_timing_penalty():
    atk = CalcPoke(
        species='Regieleki', nature='Timid',
        stats={'hp': 120, 'atk': 80, 'def': 80, 'spa': 150, 'spd': 80, 'spe': 200},
        cur_hp=120, types=('Electric',),
    )
    foe = CalcPoke(
        species='Incineroar', nature='Adamant',
        stats={'hp': 170, 'atk': 140, 'def': 110, 'spa': 80, 'spd': 100, 'spe': 80},
        cur_hp=170, types=('Fire', 'Dark'),
    )
    move = MoveOption(1, 'Hyper Beam', 'normal', 'Special')
    atoms = _atoms_from_side(
        [move], [('L', move, 1, foe, 0)], [(100, 120)], [(1, foe)],
        can_mega=False, attacker=atk, weather=None,
    )
    assert atoms
    assert atoms[0].utility == 'damage'
    assert atoms[0].timing_penalty > 0
