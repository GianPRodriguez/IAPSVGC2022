"""Smoke tests Greedy-Joint (producto cartesiano + daño combinado)."""
from __future__ import annotations

import asyncio

from showdown.battle_bots.greedy_solo.greedy import CalcPoke, MoveOption
from showdown.battle_bots.greedy_joint.joint import best_joint_double_action
from showdown.damage_calc_bridge import set_damage_range_override


def test_joint_returns_pair_with_positive_score():
    def fake_batch(reqs):
        out = []
        for req in reqs:
            foe = str((req.get('defender') or {}).get('species', '')).lower()
            if 'toxapex' in foe:
                out.append((40, 50))
            else:
                out.append((80, 100))
        return out

    set_damage_range_override(fake_batch)
    try:
        left = CalcPoke(
            species='Rillaboom', nature='Adamant',
            stats={'hp': 193, 'atk': 173, 'def': 115, 'spa': 72, 'spd': 100, 'spe': 109},
            cur_hp=193, types=('Grass', 'Dark'),
        )
        right = CalcPoke(
            species='Salamence', nature='Naive',
            stats={'hp': 171, 'atk': 176, 'def': 100, 'spa': 145, 'spd': 90, 'spe': 163},
            cur_hp=171, types=('Dragon', 'Flying'),
        )
        foes = [
            (1, CalcPoke(
                species='Pelipper', nature='Modest',
                stats={'hp': 140, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
                cur_hp=40, types=('Water', 'Flying'),
            )),
            (2, CalcPoke(
                species='Toxapex', nature='Calm',
                stats={'hp': 141, 'atk': 74, 'def': 181, 'spa': 73, 'spd': 187, 'spe': 55},
                cur_hp=141, types=('Poison', 'Water'),
            )),
        ]
        left_moves = [MoveOption(1, 'Wood Hammer', 'normal', 'Physical')]
        right_moves = [MoveOption(1, 'Double-Edge', 'normal', 'Physical')]

        best_l, best_r, score = asyncio.run(
            best_joint_double_action(
                left, right, left_moves, right_moves, foes,
            )
        )
        assert best_l is not None and best_r is not None
        assert best_l.target in (1, 2)
        assert best_r.target in (1, 2)
        assert score > 0
    finally:
        set_damage_range_override(None)


def test_joint_import_battlebot():
    from showdown.battle_bots.greedy_joint.main import BattleBot
    from showdown.battle_bots.greedy_solo.main import BattleBot as Solo

    assert BattleBot is not Solo
    assert BattleBot.__module__.endswith('greedy_joint.main')


def test_redirect_plus_ko_beats_double_attack_focus():
    """Follow Me + partner KO should outscore both attacking the same weak foe."""
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Pelipper', nature='Modest',
            stats={'hp': 140, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 101},
            cur_hp=30, types=('Water', 'Flying'),
        )),
        (2, CalcPoke(
            species='Toxapex', nature='Calm',
            stats={'hp': 141, 'atk': 74, 'def': 181, 'spa': 73, 'spd': 187, 'spe': 55},
            cur_hp=141, types=('Poison', 'Water'),
        )),
    ]
    redirect = J.JointAtom(
        move_slot=1, target=0, utility='redirect', move_name='Follow Me',
    )
    kill = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Wood Hammer',
        damage_pct={1: 90.0}, max_roll={1: 40}, ko_tids={1},
    )
    chip = J.JointAtom(
        move_slot=2, target=1, utility='damage', move_name='U-turn',
        damage_pct={1: 20.0}, max_roll={1: 10}, ko_tids=set(),
    )
    score_redir = J._score_pair(redirect, kill, foes)
    score_double = J._score_pair(chip, kill, foes)
    assert score_redir > score_double
    assert J._redirect_synergy(redirect, kill, foes) == J._REDIRECT_WITH_KO


def test_redirect_with_high_damage_gets_attack_bonus():
    from showdown.battle_bots.greedy_joint import joint as J

    foes = [
        (1, CalcPoke(
            species='Archaludon', nature='Sassy',
            stats={'hp': 170, 'atk': 125, 'def': 150, 'spa': 145, 'spd': 100, 'spe': 100},
            cur_hp=170, types=('Steel', 'Dragon'),
        )),
    ]
    redirect = J.JointAtom(move_slot=1, target=0, utility='redirect', move_name='Rage Powder')
    heavy = J.JointAtom(
        move_slot=1, target=1, utility='damage', move_name='Double-Edge',
        damage_pct={1: 55.0}, max_roll={1: 80}, ko_tids=set(),
    )
    assert J._redirect_synergy(redirect, heavy, foes) == J._REDIRECT_WITH_ATTACK


def test_utility_kind_classifies_redirect():
    from showdown.battle_bots.greedy_joint.joint import _utility_kind

    assert _utility_kind(MoveOption(1, 'Follow Me', 'self', 'Status')) == 'redirect'
    assert _utility_kind(MoveOption(1, 'Rage Powder', 'self', 'Status')) == 'redirect'
    assert _utility_kind(MoveOption(1, 'Protect', 'self', 'Status')) == 'protect'


def test_joint_atoms_exclude_switches():
    """Decision space is moves only; switches are never JointAtoms."""
    from showdown.battle_bots.greedy_joint.joint import JointAtom, _atoms_from_side

    moves = [
        MoveOption(1, 'Follow Me', 'self', 'Status'),
        MoveOption(2, 'Wood Hammer', 'normal', 'Physical'),
    ]
    atoms = _atoms_from_side(moves, meta=[], ranges=[], foes=[], can_mega=False)
    assert all(isinstance(a, JointAtom) for a in atoms)
    assert {a.utility for a in atoms} == {'redirect'}
    assert all(a.move_name != 'switch' for a in atoms)


def test_robust_player_factories():
    from poke_env.player import RandomPlayer

    from showdown.battle_bots.greedy_joint.pokeenv_player import GreedyJointPlayer
    from showdown.battle_bots.greedy_solo.pokeenv_player import GreedyDamagePlayer
    from showdown.inference_sim.validated_greedy_player import (
        ValidatedGreedyJointPlayer,
        ValidatedGreedyPlayer,
        opponent_player_class,
        validated_player_class,
    )

    assert validated_player_class('greedy_solo') is ValidatedGreedyPlayer
    assert validated_player_class('greedy_joint') is ValidatedGreedyJointPlayer
    assert opponent_player_class('random') is RandomPlayer
    assert opponent_player_class('greedy_solo') is GreedyDamagePlayer
    assert opponent_player_class('greedy_joint') is GreedyJointPlayer
    assert issubclass(ValidatedGreedyJointPlayer, GreedyJointPlayer)
    assert issubclass(ValidatedGreedyPlayer, GreedyDamagePlayer)
