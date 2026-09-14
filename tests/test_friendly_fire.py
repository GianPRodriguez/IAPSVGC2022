"""Friendly fire: allAdjacent/all dañan al compañero; Protect/inmune anulan."""
from __future__ import annotations

from showdown.battle_bots.greedy_joint.joint import JointAtom, _score_pair
from showdown.battle_bots.greedy_solo.greedy import CalcPoke
from showdown.battle_bots.move_mechanics import (
    ALLY_DAMAGE_WEIGHT,
    ALLY_KO_BONUS,
    friendly_fire_expectimax_adjust,
    friendly_fire_pair_penalty,
    hits_ally,
    move_target_type,
    partner_avoids_friendly_fire,
)


def _mon(species, *, types=(), ability=None, hp=100, cur=None):
    return CalcPoke(
        species=species,
        types=tuple(types),
        ability=ability,
        stats={'hp': hp, 'atk': 150, 'def': 80, 'spa': 80, 'spd': 80, 'spe': 80},
        cur_hp=hp if cur is None else cur,
    )


def test_hits_ally_target_types():
    assert hits_ally('allAdjacent')
    assert hits_ally('allAdjacentFoes') is False
    assert hits_ally('all')
    assert hits_ally('normal') is False
    # Dex fallback
    assert hits_ally(move_name='Earthquake')
    assert hits_ally(move_name='Rock Slide') is False
    assert hits_ally(move_name='Surf')
    assert move_target_type('Earthquake') == 'alladjacent'


def test_protect_and_switch_avoid_friendly_fire():
    assert partner_avoids_friendly_fire('protect')
    assert partner_avoids_friendly_fire('switch')
    assert partner_avoids_friendly_fire('damage') is False
    prot = _mon('incineroar', types=('Fire', 'Dark'))
    prot.is_protected = True
    assert partner_avoids_friendly_fire('damage', prot)


def test_penalty_zero_when_partner_protects():
    eq = JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Earthquake',
        damage_pct={1: 40.0}, hits_ally=True, ally_damage_pct=55.0, ally_ko=False,
    )
    protect = JointAtom(move_slot=1, target=0, utility='protect', move_name='Protect')
    attack = JointAtom(
        move_slot=2, target=1, utility='damage', move_name='Knock Off',
        damage_pct={1: 30.0},
    )
    left = _mon('garchomp', types=('Dragon', 'Ground'))
    right = _mon('incineroar', types=('Fire', 'Dark'))
    assert friendly_fire_pair_penalty(eq, protect, left, right) == 0.0
    pen = friendly_fire_pair_penalty(eq, attack, left, right)
    assert abs(pen - ALLY_DAMAGE_WEIGHT * 55.0) < 1e-6


def test_penalty_zero_when_ally_damage_zero_immune():
    """Flying / Levitate → calc 0; no pena aunque hits_ally."""
    eq = JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Earthquake',
        damage_pct={1: 40.0}, hits_ally=True, ally_damage_pct=0.0, ally_ko=False,
    )
    partner = JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Air Slash',
        damage_pct={1: 35.0},
    )
    left = _mon('garchomp', types=('Dragon', 'Ground'))
    right = _mon('corviknight', types=('Flying', 'Steel'))
    assert friendly_fire_pair_penalty(eq, partner, left, right) == 0.0


def test_ally_ko_penalty_huge():
    eq = JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Earthquake',
        damage_pct={1: 40.0}, hits_ally=True, ally_damage_pct=80.0, ally_ko=True,
    )
    partner = JointAtom(0, 0, utility='status')
    pen = friendly_fire_pair_penalty(eq, partner, None, None)
    assert pen >= ALLY_KO_BONUS


def test_score_pair_prefers_eq_plus_protect():
    """EQ sin Protect << EQ+Protect cuando el chip al ally es alto."""
    foes = [(1, _mon('fluttermane', types=('Ghost', 'Fairy'), hp=150))]
    eq = JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Earthquake',
        damage_pct={1: 45.0}, max_roll={1: 60},
        hits_ally=True, ally_damage_pct=50.0,
    )
    knock = JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Knock Off',
        damage_pct={1: 35.0}, max_roll={1: 50},
    )
    protect = JointAtom(move_slot=2, target=0, utility='protect', move_name='Protect')
    left = _mon('garchomp', types=('Dragon', 'Ground'))
    right = _mon('incineroar', types=('Fire', 'Dark'))

    s_naked = _score_pair(eq, knock, foes, left_poke=left, right_poke=right)
    s_prot = _score_pair(eq, protect, foes, left_poke=left, right_poke=right)
    assert s_prot > s_naked


def test_expectimax_adjust_raises_threat_and_pko():
    eq = JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Earthquake',
        hits_ally=True, ally_damage_pct=40.0, ally_ko=True,
    )
    pad = JointAtom(0, 0, utility='status')
    threat, pko = friendly_fire_expectimax_adjust(
        threat=10.0,
        p_ko_side={'OL': 0.1, 'OR': 0.0},
        left=eq,
        right=pad,
        soft_ko_bonus=55.0,
    )
    assert threat > 10.0
    assert pko['OR'] >= 0.99  # víctima del EQ de L
    assert abs(pko['OL'] - 0.1) < 1e-9
