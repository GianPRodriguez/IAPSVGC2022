import sys
sys.path.insert(0, '.')

from showdown.battle import Battle, Pokemon
from showdown.battle_modifier import (
    _revert_special_pokemon,
    sidestart,
    switch_or_drag,
    transform,
)


class TestBattle(Battle):
    async def action_team_preview(self, battle):
        pass

    async def action_battle(self, battle):
        pass


def build_pkmn(name, is_opponent=True, hp=100):
    p = Pokemon(name, hp, is_opponent, '', '', None, None, None, None, None)
    p.HPA_battle_actual = hp
    return p


def test_update_pkmn_user_replaces_active_moves_from_request():
    battle = TestBattle()
    battle.user.active1 = build_pkmn('ditto', is_opponent=False, hp=155)
    battle.user.active2 = build_pkmn('torkoal', is_opponent=False, hp=177)
    battle.user.active1.moves = ['old1', 'old2', 'old3', 'old4']
    battle.user.active2.moves = ['old1', 'old2', 'old3', 'old4']
    battle.user.reserved = [
        build_pkmn('whimsicott', is_opponent=False, hp=155),
        build_pkmn('raichu', is_opponent=False, hp=167),
        build_pkmn('whimsicott', is_opponent=False, hp=155),
        build_pkmn('raichu', is_opponent=False, hp=167),
    ]

    request = {
        'active': [
            {'moves': [{'move': 'Struggle', 'id': 'struggle', 'disabled': False}]},
            {'moves': [
                {'move': 'Yawn', 'id': 'yawn', 'pp': 12},
                {'move': 'Body Slam', 'id': 'bodyslam', 'pp': 16},
                {'move': 'Will-O-Wisp', 'id': 'willowisp', 'pp': 16},
                {'move': 'Clear Smog', 'id': 'clearsmog', 'pp': 15},
            ]},
        ],
        'side': {
            'pokemon': [
                {'details': 'Ditto, L50', 'condition': '155/155', 'stats': {'atk': 61, 'def': 68, 'spa': 68, 'spd': 68, 'spe': 74}},
                {'details': 'Torkoal, L50, F', 'condition': '177/177', 'stats': {'atk': 137, 'def': 176, 'spa': 105, 'spd': 90, 'spe': 37}},
                {'details': 'Whimsicott, L50, F', 'condition': '155/155', 'stats': {'atk': 78, 'def': 111, 'spa': 105, 'spd': 95, 'spe': 184}},
                {'details': 'Raichu, L50, M', 'condition': '167/167', 'stats': {'atk': 99, 'def': 75, 'spa': 156, 'spd': 100, 'spe': 130}},
            ],
        },
    }

    battle.user.updatePkmnUser(request, False, None)

    assert battle.user.active1.moves == ['struggle', None, None, None]
    assert battle.user.active1.movesPP == [-1, -1, -1, -1]
    assert battle.user.active2.moves == ['yawn', 'bodyslam', 'willowisp', 'clearsmog']
    assert battle.user.active2.movesPP == [12, 16, 16, 15]


def test_ditto_agent_returns_with_original_current_stats_and_cleared_effects():
    side = TestBattle().user
    side.isUser = True
    ditto = build_pkmn('gengar', is_opponent=False, hp=80)
    ditto.is_ditto = True
    ditto.ATK_battle_actual = 123
    ditto.non_volatile_statuses['par'] = True
    ditto.volatile_statuses['confusion'] = True
    ditto.boosts['atk'] = 2

    reverted = _revert_special_pokemon(ditto, side)

    assert reverted.ATK_battle_base == -1
    assert reverted.ATK_battle_actual == 123
    assert reverted.non_volatile_statuses['par'] is True
    assert not reverted.volatile_statuses
    assert not reverted.boosts


def test_ditto_opponent_returns_with_template_base_and_unknown_current_stats():
    side = TestBattle().opponent
    ditto_template = build_pkmn('ditto', is_opponent=True, hp=100)
    side.opponents = [ditto_template]
    transformed = build_pkmn('gengar', is_opponent=True, hp=70)
    transformed.is_ditto = True
    transformed.ATK_battle_actual = 123
    transformed.non_volatile_statuses['par'] = True
    transformed.volatile_statuses['confusion'] = True
    transformed.boosts['atk'] = 2

    reverted = _revert_special_pokemon(transformed, side)

    assert reverted.ATK_battle_base == ditto_template.ATK_battle_base
    assert reverted.ATK_battle_actual == -1
    assert reverted.non_volatile_statuses['par'] is True
    assert not reverted.volatile_statuses
    assert not reverted.boosts


def test_zoroark_and_inferred_disguise_clear_reserve_effects():
    side = TestBattle().opponent
    zoroark_template = build_pkmn('zoroark', is_opponent=True)
    side.opponents = [zoroark_template]
    side.reserved = [build_pkmn('blastoise', is_opponent=True)]
    disguised = build_pkmn('blastoise', is_opponent=True)
    disguised.non_volatile_statuses['par'] = True
    disguised.volatile_statuses['confusion'] = True
    disguised.boosts['atk'] = 2

    reverted = _revert_special_pokemon(disguised, side)

    assert reverted.name == 'zoroark'
    assert reverted.is_zoroark is True
    assert reverted.non_volatile_statuses['par'] is True
    assert not reverted.volatile_statuses
    assert not reverted.boosts


def test_sidestart_sets_tailwind_to_three_turns():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    sidestart(battle, ['', '-sidestart', 'p1: Agent', 'move: Tailwind'])

    assert battle.user.side_conditions['tailwind'] == 3


def test_starmie_remains_in_reserve_after_uncertain_zoroark_switch_sequence():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    battle.opponent.active1 = build_pkmn('dragonite', is_opponent=True)
    battle.opponent.active2 = build_pkmn('starmie', is_opponent=True)
    battle.opponent.reserved = [
        build_pkmn('blastoise', is_opponent=True),
        build_pkmn('starmie', is_opponent=True),
    ]
    battle.opponent.opponents = [
        build_pkmn('zoroark', is_opponent=True),
        build_pkmn('dragonite', is_opponent=True),
        build_pkmn('starmie', is_opponent=True),
        build_pkmn('blastoise', is_opponent=True),
    ]

    switch_or_drag(battle, ['', 'switch', 'p2b: Dragonite', 'Dragonite, L50, M', '100/100'])
    switch_or_drag(battle, ['', 'switch', 'p2a: Starmie', 'Starmie, L50', '7/100'])

    reserved_names = [p.name for p in battle.opponent.reserved]
    assert 'starmie' in reserved_names
    assert 'zoroark' not in reserved_names


def test_user_ditto_transform_uses_target_base_and_current_minus_one():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    battle.user.active1 = build_pkmn('ditto', is_opponent=False)
    battle.opponent.active1 = build_pkmn('gengar', is_opponent=True)
    battle.user.active1.HPA_battle_actual = 37
    battle.user.active1.HPB_battle_base = 100
    for index, stat_name in enumerate(('ATK', 'DEF', 'SPA', 'SPD', 'VEL'), start=1):
        setattr(battle.opponent.active1, f'{stat_name}_battle_base', index * 10)
        setattr(battle.opponent.active1, f'{stat_name}_battle_actual', index * 100)

    transform(battle, ['', '-transform', 'p1a: Ditto', 'p2a: Gengar', '[from] ability: Imposter'])

    transformed = battle.user.active1
    assert transformed.name == 'gengar'
    assert transformed.HPA_battle_actual == 37
    assert transformed.HPB_battle_base == 100
    for stat_name in ('ATK', 'DEF', 'SPA', 'SPD', 'VEL'):
        assert getattr(transformed, f'{stat_name}_battle_base') == getattr(battle.opponent.active1, f'{stat_name}_battle_base')
        assert getattr(transformed, f'{stat_name}_battle_actual') == -1


def test_opponent_ditto_transform_keeps_its_base_minus_one_and_copies_current_stats():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    battle.opponent.active1 = build_pkmn('ditto', is_opponent=True)
    battle.user.active1 = build_pkmn('starmie', is_opponent=False)
    battle.opponent.active1.HPA_battle_actual = 29
    battle.opponent.active1.HPB_battle_base = 90
    for index, stat_name in enumerate(('ATK', 'DEF', 'SPA', 'SPD', 'VEL'), start=1):
        setattr(battle.user.active1, f'{stat_name}_battle_base', index * 10)
        setattr(battle.user.active1, f'{stat_name}_battle_actual', index * 100)

    transform(battle, ['', '-transform', 'p2a: Ditto', 'p1a: Starmie', '[from] ability: Imposter'])

    transformed = battle.opponent.active1
    assert transformed.name == 'starmie'
    assert transformed.HPA_battle_actual == 29
    assert transformed.HPB_battle_base == 90
    for stat_name in ('ATK', 'DEF', 'SPA', 'SPD', 'VEL'):
        assert getattr(transformed, f'{stat_name}_battle_base') == -1
        assert getattr(transformed, f'{stat_name}_battle_actual') == getattr(battle.user.active1, f'{stat_name}_battle_actual')
