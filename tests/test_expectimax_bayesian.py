"""Smoke tests Expectimax-Bayesian (chance, belief, anytime, A/B/C upgrades)."""
from __future__ import annotations

import asyncio

from showdown.battle_bots.expectimax_bayesian.chance import (
    ChanceNode,
    combined_p_ko,
    discrete_rolls,
    p_ko,
)
from showdown.battle_bots.expectimax_bayesian.belief import (
    OpponentProfile,
    resolve_opp_move_names,
    sample_belief_node,
)
from showdown.battle_bots.expectimax_bayesian.expectimax import (
    ExpectimaxConfig,
    ReserveOption,
    best_expectimax_bayesian_action,
)
from showdown.battle_bots.expectimax_bayesian.opp_policy import opp_policy_distribution
from showdown.battle_bots.expectimax_bayesian.speed_order import acts_before, race_scale
from showdown.battle_bots.expectimax_bayesian.utility import switch_preservation_value
from showdown.battle_bots.greedy_solo.greedy import CalcPoke, MoveOption
from showdown.damage_calc_bridge import set_damage_range_override


def test_discrete_rolls_count_and_bounds():
    rolls = discrete_rolls(85, 100)
    assert len(rolls) == 16
    assert rolls[0] == 85
    assert rolls[-1] == 100


def test_soft_p_ko_partial():
    node = ChanceNode(min_damage=40, max_damage=60, cur_hp=50, max_hp=100)
    assert 0.0 < node.p_ko < 1.0
    assert node.expected_damage == sum(node.rolls) / 16.0


def test_p_ko_guaranteed_and_impossible():
    assert p_ko((50, 60), 40) == 1.0
    assert p_ko((10, 20), 100) == 0.0


def test_combined_p_ko_two_hits():
    assert combined_p_ko([(30, 30), (30, 30)], 50) == 1.0
    assert combined_p_ko([(20, 20)], 50) == 0.0


def test_opp_policy_sums_to_one():
    moves = [
        MoveOption(1, 'Hurricane', 'normal', 'Special'),
        MoveOption(2, 'Protect', 'self', 'Status'),
        MoveOption(3, 'Ice Beam', 'normal', 'Special'),
        MoveOption(4, 'Roost', 'self', 'Status'),
    ]
    dist = opp_policy_distribution(moves, damage_by_slot={1: 80.0, 2: 0.0, 3: 40.0, 4: 0.0})
    assert dist
    total = sum(p for _, p in dist)
    assert abs(total - 1.0) < 1e-6


def test_opp_policy_prefers_higher_damage_when_weighted():
    """π_opp post-calc: más masa en el move con más daño (vs uniforme legacy)."""
    moves = [
        MoveOption(1, 'Hurricane', 'normal', 'Special'),
        MoveOption(2, 'Protect', 'self', 'Status'),
        MoveOption(3, 'Ice Beam', 'normal', 'Special'),
    ]
    dist = opp_policy_distribution(
        moves,
        damage_by_slot={1: 100.0, 2: 0.0, 3: 10.0},
        w_greedy=0.82,
        w_control=0.12,
        w_uniform=0.06,
    )
    by_name = {a.move_name.lower().replace(' ', ''): p for a, p in dist}
    assert by_name.get('hurricane', 0) > by_name.get('icebeam', 0)


def test_opp_policy_targeted_splits_mass_and_switch():
    """Targets ∝ daño; switch resta masa; OL+OR no suman >1 en ataques."""
    from collections import defaultdict

    from showdown.battle_bots.expectimax_bayesian.opp_policy import opp_policy_targeted

    moves = [
        MoveOption(1, 'Hurricane', 'normal', 'Special'),
        MoveOption(2, 'Protect', 'self', 'Status'),
        MoveOption(3, 'Ice Beam', 'normal', 'Special'),
    ]
    dist = opp_policy_targeted(
        moves,
        damage_by_slot_target={
            1: {1: 100.0, 2: 10.0},
            2: {1: 0.0, 2: 0.0},
            3: {1: 5.0, 2: 80.0},
        },
        available_targets=(1, 2),
        w_greedy=0.75,
        w_control=0.12,
        w_uniform=0.03,
        w_switch=0.10,
    )
    ol = dist['OL']
    or_ = dist['OR']
    assert ol and or_
    p_ol = sum(p for _, p in ol)
    p_or = sum(p for _, p in or_)
    # Ataques no cubren toda la masa (control+switch restan)
    assert p_ol + p_or < 0.95
    ol_by: dict = defaultdict(float)
    or_by: dict = defaultdict(float)
    for a, p in ol:
        ol_by[a.move_name.lower().replace(' ', '')] += p
    for a, p in or_:
        or_by[a.move_name.lower().replace(' ', '')] += p
    assert ol_by.get('hurricane', 0) > ol_by.get('icebeam', 0)
    assert or_by.get('icebeam', 0) > or_by.get('hurricane', 0)


def test_phase2_gate_helpers_match_scored_to_atom():
    from showdown.battle_bots.expectimax_bayesian.expectimax import (
        _scored_matches_atom,
    )
    from showdown.battle_bots.greedy_joint.joint import JointAtom
    from showdown.battle_bots.greedy_solo.greedy import ScoredAction

    atom = JointAtom(move_slot=2, target=1, mega=0, utility='damage', move_name='EQ')
    scored = ScoredAction(score=1.0, move_slot=2, target=1, mega=0)
    assert _scored_matches_atom(scored, atom)
    assert not _scored_matches_atom(ScoredAction(score=1.0, move_slot=3, target=1), atom)
    sw = JointAtom(move_slot=3, target=0, mega=0, utility='switch', move_name='Switch 3')
    assert _scored_matches_atom(
        ScoredAction(score=0.0, move_slot=3, target=0, kind='switch'), sw,
    )


def test_belief_node_normalized():
    node = sample_belief_node(
        species='Pelipper',
        nature='Modest',
        candidates={
            'hp': set(range(0, 33)),
            'atk': set(range(0, 33)),
            'def': set(range(0, 33)),
            'spa': set(range(0, 33)),
            'spd': set(range(0, 33)),
            'spe': set(range(0, 33)),
        },
        k=4,
    )
    assert node.profiles
    assert abs(sum(node.probabilities) - 1.0) < 1e-6


def test_resolve_opp_moves_prefers_ots():
    prof = OpponentProfile(
        species='pelipper', nature='modest', sp=(20, 0, 0, 32, 0, 12),
        moves=('Hurricane', 'Protect', 'Tailwind', 'U-turn'),
    )
    known = ('Hurricane', 'Ice Beam', 'Wide Guard', 'Protect')
    assert resolve_opp_move_names(prof, known) == known
    assert resolve_opp_move_names(prof, ()) == prof.moves


def test_acts_before_priority_and_spe():
    assert acts_before(100, 1, 200, 0, trick_room=False)
    assert acts_before(200, 0, 100, 0, trick_room=False)
    assert acts_before(100, 0, 200, 0, trick_room=True)
    assert not acts_before(100, 0, 100, 0, trick_room=False)


def test_race_scale_continuous():
    assert abs(race_scale(0.0) - 1.0) < 1e-9
    assert abs(race_scale(1.0) - 0.15) < 1e-9
    assert 0.15 < race_scale(0.5) < 1.0


def test_switch_preservation_continuous():
    high = switch_preservation_value(stay_p_ko=0.95, bench_incoming_pct=10.0, matchup=1.0)
    low = switch_preservation_value(stay_p_ko=0.05, bench_incoming_pct=80.0, matchup=-1.0)
    assert high > low
    a = switch_preservation_value(stay_p_ko=0.5, bench_incoming_pct=29.0, matchup=0.0)
    b = switch_preservation_value(stay_p_ko=0.5, bench_incoming_pct=31.0, matchup=0.0)
    assert abs(a - b) < 30.0


def test_expectimax_returns_pair_with_fallback_safe():
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
        reserves = [
            ReserveOption(
                slot=3,
                poke=CalcPoke(
                    species='Indeedee-F', nature='Bold',
                    stats={'hp': 160, 'atk': 50, 'def': 120, 'spa': 90, 'spd': 110, 'spe': 80},
                    cur_hp=160, types=('Psychic', 'Normal'),
                ),
                types=('Psychic', 'Normal'),
            ),
        ]
        cfg = ExpectimaxConfig(
            deadline_ms=5000.0,
            phase1_budget_ms=2000.0,
            phase2_end_ms=4800.0,
            k_profiles=3,
            max_worlds=4,
            max_our_pairs=12,
            enable_voluntary_switch=True,
        )
        best_l, best_r, score, meta = asyncio.run(
            best_expectimax_bayesian_action(
                left, right, left_moves, right_moves, foes,
                config=cfg,
                foe_known_moves={
                    1: ('Hurricane', 'Ice Beam', 'Wide Guard', 'Protect'),
                    2: ('Recover', 'Baneful Bunker', 'Scald', 'Toxic'),
                },
                reserves=reserves,
            )
        )
        assert best_l is not None and best_r is not None
        assert getattr(best_l, 'kind', 'move') in ('move', 'switch')
        assert getattr(best_r, 'kind', 'move') in ('move', 'switch')
        assert score != float('-inf')
        assert meta.elapsed_ms >= 0.0
        assert meta.phase.startswith('phase')
    finally:
        set_damage_range_override(None)


def test_expectimax_import_battlebot():
    from showdown.battle_bots.expectimax_bayesian.main import BattleBot
    from showdown.battle_bots.greedy_joint.main import BattleBot as Joint

    assert BattleBot is not Joint
    assert BattleBot.__module__.endswith('expectimax_bayesian.main')
