"""Proyección residual expectimax: condenado + status → S'."""
from __future__ import annotations

from showdown.battle_bots.expectimax_bayesian.residual import (
    condemned_reweight,
    project_residual_into_v,
)
from showdown.battle_bots.greedy_joint.joint import JointAtom
from showdown.battle_bots.greedy_solo.greedy import CalcPoke
from showdown.battle_bots.move_mechanics import (
    THREAT_RELIEF,
    expected_effect_score,
    expected_threat_relief,
)


def _mon(species, *, types=(), hp=100, status=''):
    return CalcPoke(
        species=species,
        types=tuple(types),
        status=status,
        stats={'hp': hp, 'atk': 100, 'def': 100, 'spa': 100, 'spd': 100, 'spe': 100},
        cur_hp=hp,
    )


def test_direclaw_threat_relief_positive():
    atk = _mon('sneasler', types=('Fighting', 'Poison'))
    foe = _mon('fluttermane', types=('Ghost', 'Fairy'))
    relief = expected_threat_relief('Dire Claw', attacker=atk, defender=foe)
    # 50% * avg(psn,par,slp) relief
    expected = 0.5 * (
        THREAT_RELIEF['psn'] + THREAT_RELIEF['par'] + THREAT_RELIEF['slp']
    ) / 3.0
    assert abs(relief - expected) < 1e-6
    assert relief > expected_effect_score('Dire Claw', attacker=atk, defender=foe).score * 0.5


def test_condemned_prefers_lasting_over_chip():
    """Con P(KO)≈1 al atacante, lasting_extra crece y chip offense cae."""
    foe = _mon('incineroar', types=('Fire', 'Dark'), hp=170)
    # Dire Claw: poco daño, alto effect_bonus
    claw = JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Dire Claw',
        damage_pct={1: 25.0}, effect_bonus=16.0,
    )
    # Close Combat: más daño, sin lasting
    cc = JointAtom(
        move_slot=2, target=1, utility='damage', move_name='Close Combat',
        damage_pct={1: 45.0}, effect_bonus=0.0,
    )
    pad = JointAtom(0, 0, utility='status')
    atk = _mon('sneasler', types=('Fighting', 'Poison'))
    foes = [(1, foe)]
    share = {'L': 100.0, 'R': 0.0}
    p_dead = {'OL': 1.0, 'OR': 0.0}
    p_ko_foe = {1: 0.0}  # no KO rival

    off_claw, thr_claw, last_claw = project_residual_into_v(
        offense=40.0,
        threat=80.0,
        share=share,
        left=claw,
        right=pad,
        left_atk=atk,
        right_atk=None,
        foes=foes,
        p_ko_side=p_dead,
        p_ko_foe=p_ko_foe,
    )
    off_cc, thr_cc, last_cc = project_residual_into_v(
        offense=55.0,
        threat=80.0,
        share=share,
        left=cc,
        right=pad,
        left_atk=atk,
        right_atk=None,
        foes=foes,
        p_ko_side=p_dead,
        p_ko_foe=p_ko_foe,
    )
    # Chip descontado fuerte al condenado
    assert off_claw < 40.0 * 0.5
    assert off_cc < 55.0 * 0.5
    # Dire Claw: lasting + threat relief > CC
    assert last_claw > last_cc
    assert thr_claw < thr_cc
    v_claw = off_claw - 0.75 * thr_claw + last_claw
    v_cc = off_cc - 0.75 * thr_cc + last_cc
    assert v_claw > v_cc


def test_condemned_reweight_no_discount_if_we_ko():
    left = JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Close Combat',
        damage_pct={1: 90.0}, effect_bonus=0.0,
    )
    right = JointAtom(0, 0, utility='status')
    off, extra = condemned_reweight(
        offense=100.0,
        share={'L': 100.0, 'R': 0.0},
        left=left,
        right=right,
        p_ko_side={'OL': 1.0, 'OR': 0.0},
        p_ko_foe={1: 1.0},  # sí KO rival
        chip_discount=0.70,
    )
    assert abs(off - 100.0) < 1e-9
    assert extra == 0.0
