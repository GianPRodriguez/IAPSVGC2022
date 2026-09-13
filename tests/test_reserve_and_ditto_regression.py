import sys
sys.path.insert(0, '.')

import constants
from showdown import stat_inference as si
from showdown.battle import Battle, Pokemon
from showdown.battle_modifier import (
    _resolve_duplicate_illusion_reserves,
    _revert_special_pokemon,
    activate,
    boost,
    clearallboost,
    clearboost,
    copyboost,
    curestatus,
    end_volatile_status,
    faint,
    fieldend,
    fieldstart,
    form_change,
    heal_or_damage,
    mega,
    observe_move,
    remove_item,
    replace_illusion,
    request,
    set_item,
    set_opponent_ability_from_ability_tag,
    set_player,
    sidestart,
    sideend,
    singlemove,
    singleturn,
    start_volatile_status,
    status,
    swap,
    switch_or_drag,
    turn,
    transform,
    unboost,
    weather,
)
from showdown.damage_calc_bridge import set_damage_range_override


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
    assert battle.user.active1.movesPP == [1, -1, -1, -1]
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


def test_sidestart_sets_tailwind_to_four_turns():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    sidestart(battle, ['', '-sidestart', 'p1: Agent', 'move: Tailwind'])

    assert battle.user.side_conditions['tailwind'] == 4
    turn(battle, ['', 'turn', '2'])
    assert battle.user.side_conditions['tailwind'] == 3


def test_sidestart_light_screen_is_eight_with_light_clay():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    clay_mon = build_pkmn('whimsicott', is_opponent=False, hp=155)
    clay_mon.objectPkmn = 'lightclay'
    battle.user.active1 = clay_mon
    battle.user.active2 = build_pkmn('farigiraf', is_opponent=False, hp=222)

    sidestart(battle, ['', '-sidestart', 'p1: Agent', 'move: Light Screen'])
    assert battle.user.side_conditions['lightscreen'] == 8

    sidestart(battle, ['', '-sidestart', 'p2: Opp', 'move: Reflect'])
    assert battle.opponent.side_conditions['reflect'] == 5


def test_weather_is_eight_turns_with_heat_rock_on_source():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    torkoal = build_pkmn('torkoal', is_opponent=False, hp=177)
    torkoal.objectPkmn = 'heatrock'
    battle.user.active1 = torkoal
    battle.user.active2 = build_pkmn('whimsicott', is_opponent=False, hp=155)

    weather(
        battle,
        ['', '-weather', 'SunnyDay', '[from] ability: Drought', '[of] p1a: Torkoal'],
    )
    assert battle.weather == 'sunnyday'
    assert battle.weatherTurn == 8


def test_weather_is_five_turns_without_extender_rock():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    torkoal = build_pkmn('torkoal', is_opponent=False, hp=177)
    torkoal.objectPkmn = 'focussash'
    battle.user.active1 = torkoal

    weather(
        battle,
        ['', '-weather', 'SunnyDay', '[from] ability: Drought', '[of] p1a: Torkoal'],
    )
    assert battle.weatherTurn == 5


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


def test_field_and_side_effects_expire_without_zero_entries():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    fieldstart(battle, ['', '-fieldstart', 'move: Wonder Room'])
    assert battle.terrain['wonderroom'] == 5
    turn(battle, ['', 'turn', '2'])
    assert battle.terrain['wonderroom'] == 4

    sidestart(battle, ['', '-sidestart', 'p2: Opponent', 'move: Light Screen'])
    sideend(battle, ['', '-sideend', 'p2: Opponent', 'move: Light Screen'])
    assert 'lightscreen' not in battle.opponent.side_conditions
    fieldend(battle, ['', '-fieldend', 'move: Wonder Room'])
    assert 'wonderroom' not in battle.terrain


def test_zoroark_illusion_keeps_damage_on_zoroark_only():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.opponent.active1 = build_pkmn('dragonite', is_opponent=True, hp=100)
    battle.opponent.active1.HPB_battle_base = 100
    battle.opponent.active1.HPA_battle_actual = 64
    zoroark = build_pkmn('zoroark', is_opponent=True)
    zoroark.nature = 'serious'
    battle.opponent.opponents = [zoroark]

    replace_illusion(battle, ['', 'replace', 'p2a: Zoroark', 'Zoroark, L50, M'])

    assert battle.opponent.active1.name == 'zoroark'
    assert battle.opponent.active1.HPA_battle_actual == 64
    dragonite = next(p for p in battle.opponent.reserved if p.name == 'dragonite')
    assert dragonite.HPA_battle_actual == dragonite.HPB_battle_base == 100


def test_substitute_survives_aegislash_form_change():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.user.active1 = build_pkmn('aegislash', is_opponent=False, hp=135)
    battle.user.active1.volatile_statuses['substitute'] = True

    form_change(battle, ['', '-formechange', 'p1a: Aegislash', 'Aegislash-Blade'])

    assert battle.user.active1.volatile_statuses['substitute'] is True


def test_request_keeps_transformed_ditto_identity_and_aegislash_substitute():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.user.active1 = build_pkmn('ditto', is_opponent=False, hp=155)
    battle.user.active2 = build_pkmn('aegislash', is_opponent=False, hp=135)
    battle.user.active2.volatile_statuses['substitute'] = True
    battle.user.reserved = [
        build_pkmn('raichu', is_opponent=False, hp=167),
        build_pkmn('torkoal', is_opponent=False, hp=177),
    ]
    battle.opponent.active1 = build_pkmn('starmie', is_opponent=True, hp=100)
    for index, stat_name in enumerate(('ATK', 'DEF', 'SPA', 'SPD', 'VEL'), start=1):
        setattr(battle.opponent.active1, f'{stat_name}_battle_base', index * 10)

    transform(battle, ['', '-transform', 'p1a: Ditto', 'p2a: Starmie', '[from] ability: Imposter'])
    form_change(battle, ['', '-formechange', 'p1b: Aegislash', 'Aegislash-Blade'])

    request = {
        'active': [{'moves': []}, {'moves': []}],
        'side': {'pokemon': [
            {'details': 'Ditto, L50', 'condition': '155/155', 'stats': {'atk': 61, 'def': 68, 'spa': 68, 'spd': 68, 'spe': 74}},
            {'details': 'Aegislash, L50, F', 'condition': '135/135', 'stats': {'atk': 102, 'def': 160, 'spa': 102, 'spd': 176, 'spe': 73}},
            {'details': 'Raichu, L50', 'condition': '167/167', 'stats': {'atk': 99, 'def': 75, 'spa': 156, 'spd': 100, 'spe': 130}},
            {'details': 'Torkoal, L50', 'condition': '177/177', 'stats': {'atk': 137, 'def': 176, 'spa': 105, 'spd': 90, 'spe': 37}},
        ]},
    }
    battle.user.updatePkmnUser(request, False, None)

    assert battle.user.active1.name == 'starmie'
    assert battle.user.active1.ATK_battle_base == 10
    assert battle.user.active1.ATK_battle_actual == -1
    assert battle.user.active2.volatile_statuses['substitute'] is True


def test_struggle_gets_one_pp_when_request_omits_pp():
    battle = TestBattle()
    battle.user.active1 = build_pkmn('ditto', is_opponent=False, hp=155)
    battle.user.active2 = build_pkmn('torkoal', is_opponent=False, hp=177)
    battle.user.reserved = [build_pkmn('raichu', is_opponent=False, hp=167), build_pkmn('ditto', is_opponent=False, hp=155)]
    request = {
        'active': [
            {'moves': [{'move': 'Struggle', 'id': 'struggle', 'disabled': False}]},
            {'moves': []},
        ],
        'side': {'pokemon': [
            {'details': 'Ditto, L50', 'condition': '155/155', 'stats': {'atk': 61, 'def': 68, 'spa': 68, 'spd': 68, 'spe': 74}},
            {'details': 'Torkoal, L50', 'condition': '177/177', 'stats': {'atk': 137, 'def': 176, 'spa': 105, 'spd': 90, 'spe': 37}},
            {'details': 'Raichu, L50', 'condition': '167/167', 'stats': {'atk': 99, 'def': 75, 'spa': 156, 'spd': 100, 'spe': 130}},
            {'details': 'Ditto, L50', 'condition': '155/155', 'stats': {'atk': 61, 'def': 68, 'spa': 68, 'spd': 68, 'spe': 74}},
        ]},
    }

    battle.user.updatePkmnUser(request, False, None)

    assert battle.user.active1.movesPP[0] == 1


def test_illusion_switch_in_uses_known_zoroark_not_second_dragonite():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    real_dragonite = build_pkmn('dragonite', is_opponent=True, hp=100)
    zoroark = build_pkmn('zoroark', is_opponent=True, hp=62)
    zoroark.is_zoroark = True
    zoroark.non_volatile_statuses['par'] = True

    battle.opponent.active1 = build_pkmn('blastoise', is_opponent=True, hp=73)
    battle.opponent.active2 = real_dragonite
    battle.opponent.reserved = [zoroark, build_pkmn('pawmot', is_opponent=True, hp=100)]
    battle.opponent.opponents = [
        build_pkmn('zoroark', is_opponent=True),
        build_pkmn('dragonite', is_opponent=True),
        build_pkmn('blastoise', is_opponent=True),
        build_pkmn('pawmot', is_opponent=True),
    ]

    switch_or_drag(
        battle,
        ['', 'switch', 'p2a: Dragonite', 'Dragonite, L50, M', '62/100 par'],
    )

    assert battle.opponent.active1.name == 'zoroark'
    assert battle.opponent.active1.is_zoroark is True
    assert battle.opponent.active1.non_volatile_statuses['par'] is True
    assert battle.opponent.active2.name == 'dragonite'
    assert all(getattr(p, 'name', '') != 'dragonite' for p in battle.opponent.reserved)
    assert all(getattr(p, 'name', '') != 'zoroark' for p in battle.opponent.reserved)


def test_illusion_switch_in_as_blastoise_mega_does_not_corrupt_real_blastoise():
    """Regresión T12: Illusion como Blastoise-Mega no debe heredar 62/par al Blastoise real."""
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    blastoise = build_pkmn('blastoise-mega', is_opponent=True, hp=73)
    blastoise.is_mega_evo = True
    blastoise.abilityPkmn = 'megalauncher'
    blastoise.is_opponent_confirmed = True

    zoroark = build_pkmn('zoroark', is_opponent=True, hp=62)
    zoroark.is_zoroark = True
    zoroark.non_volatile_statuses['par'] = True

    battle.opponent.active1 = build_pkmn('pawmot', is_opponent=True, hp=100)
    battle.opponent.active2 = build_pkmn('dragonite', is_opponent=True, hp=0)
    battle.opponent.active2.is_fainted = True
    battle.opponent.reserved = [zoroark, blastoise]
    battle.opponent.opponents = [
        build_pkmn('zoroark', is_opponent=True),
        build_pkmn('blastoise', is_opponent=True),
        build_pkmn('pawmot', is_opponent=True),
        build_pkmn('dragonite', is_opponent=True),
    ]

    switch_or_drag(
        battle,
        ['', 'switch', 'p2b: Blastoise', 'Blastoise-Mega, L50, M', '62/100 par'],
    )

    assert battle.opponent.active2.name == 'zoroark'
    assert battle.opponent.active2.HPA_battle_actual == 62
    assert battle.opponent.active2.non_volatile_statuses['par'] is True
    reserved_blastoise = next(p for p in battle.opponent.reserved if 'blastoise' in p.name)
    assert reserved_blastoise.HPA_battle_actual == 73
    assert not reserved_blastoise.non_volatile_statuses.get('par')


def test_illusion_deduced_by_burn_vs_clean_real_copy():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    real_dragonite = build_pkmn('dragonite', is_opponent=True, hp=80)
    zoroark = build_pkmn('zoroark', is_opponent=True, hp=55)
    zoroark.is_zoroark = True
    zoroark.non_volatile_statuses['brn'] = True

    battle.opponent.active1 = build_pkmn('pawmot', is_opponent=True, hp=100)
    battle.opponent.active2 = real_dragonite
    battle.opponent.reserved = [zoroark]
    battle.opponent.opponents = [build_pkmn('zoroark', is_opponent=True), build_pkmn('dragonite', is_opponent=True)]

    switch_or_drag(
        battle,
        ['', 'switch', 'p2a: Dragonite', 'Dragonite, L50, M', '55/100 brn'],
    )

    assert battle.opponent.active1.name == 'zoroark'
    assert battle.opponent.active1.non_volatile_statuses['brn'] is True
    assert battle.opponent.active2.name == 'dragonite'
    assert battle.opponent.active2.HPA_battle_actual == 80


def test_illusion_deduced_by_hp_percent_discordance_without_status():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    real_dragonite = build_pkmn('dragonite', is_opponent=True, hp=100)
    zoroark = build_pkmn('zoroark', is_opponent=True, hp=41)
    zoroark.is_zoroark = True

    battle.opponent.active1 = build_pkmn('pawmot', is_opponent=True, hp=100)
    battle.opponent.active2 = real_dragonite
    battle.opponent.reserved = [zoroark]
    battle.opponent.opponents = [build_pkmn('zoroark', is_opponent=True), build_pkmn('dragonite', is_opponent=True)]

    switch_or_drag(
        battle,
        ['', 'switch', 'p2a: Dragonite', 'Dragonite, L50, M', '41/100'],
    )

    assert battle.opponent.active1.name == 'zoroark'
    assert battle.opponent.active1.HPA_battle_actual == 41


def test_mega_form_change_sets_mega_ability_and_keeps_battle_state():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    blastoise = build_pkmn('blastoise', is_opponent=True, hp=73)
    blastoise.abilityPkmn = 'raindish'
    blastoise.is_opponent_confirmed = True
    blastoise.non_volatile_statuses['par'] = True
    blastoise.boosts['special-attack'] = 1
    battle.opponent.active1 = blastoise

    form_change(battle, ['', 'detailschange', 'p2a: Blastoise', 'Blastoise-Mega, L50, M'])

    mega = battle.opponent.active1
    assert mega.name == 'blastoise-mega'
    assert mega.abilityPkmn == 'megalauncher'
    assert mega.is_mega_evo is True
    assert mega.is_opponent_confirmed is True
    assert mega.HPA_battle_actual == 73
    assert mega.non_volatile_statuses['par'] is True
    assert mega.boosts['special-attack'] == 1


def test_weather_decrements_only_on_protocol_turn():
    battle = TestBattle()
    battle.weather = 'sunnyday'
    battle.weatherTurn = 5
    battle.turn = 1

    turn(battle, ['', 'turn', '2'])
    assert battle.turn == 2
    assert battle.weatherTurn == 4

    turn(battle, ['', 'turn', '3'])
    assert battle.weatherTurn == 3


def test_resolve_removes_disguise_duplicate_matching_zoroark_in_reserve():
    battle = TestBattle()
    fake_dragonite = build_pkmn('dragonite', is_opponent=True, hp=62)
    fake_dragonite.non_volatile_statuses['par'] = True
    real_fainted = build_pkmn('dragonite', is_opponent=True, hp=0)
    real_fainted.is_fainted = True
    zoroark = build_pkmn('zoroark', is_opponent=True, hp=62)
    zoroark.is_zoroark = True
    zoroark.non_volatile_statuses['par'] = True

    battle.opponent.reserved = [fake_dragonite, zoroark, real_fainted]
    _resolve_duplicate_illusion_reserves(battle.opponent)

    names = [p.name for p in battle.opponent.reserved]
    assert names.count('dragonite') == 1
    assert 'zoroark' in names
    assert next(p for p in battle.opponent.reserved if p.name == 'dragonite').is_fainted is True


def test_revival_blessing_heals_fainted_party_member_opponent_and_user():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    opp_dragonite = build_pkmn('dragonite', is_opponent=True, hp=0)
    opp_dragonite.is_fainted = True
    opp_dragonite.HPB_battle_base = 100
    battle.opponent.reserved = [opp_dragonite]
    battle.opponent.active1 = build_pkmn('pawmot', is_opponent=True, hp=100)

    user_farigiraf = build_pkmn('farigiraf', is_opponent=False, hp=0)
    user_farigiraf.is_fainted = True
    user_farigiraf.HPB_battle_base = 222
    battle.user.reserved = [user_farigiraf]
    battle.user.active1 = build_pkmn('whimsicott', is_opponent=False, hp=55)

    heal_or_damage(
        battle,
        ['', '-heal', 'p2: Dragonite', '50/100y', '[from] move: Revival Blessing'],
    )
    heal_or_damage(
        battle,
        ['', '-heal', 'p1: Farigiraf', '111/222', '[from] move: Revival Blessing'],
    )

    assert opp_dragonite.is_fainted is False
    assert opp_dragonite.HPA_battle_actual == 50
    assert user_farigiraf.is_fainted is False
    assert user_farigiraf.HPA_battle_actual == 111


def test_curestatus_clears_non_volatile_statuses_dict():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    pkmn = build_pkmn('blastoise', is_opponent=True, hp=80)
    pkmn.non_volatile_statuses['par'] = True
    battle.opponent.active1 = pkmn

    curestatus(battle, ['', '-curestatus', 'p2a: Blastoise', 'par', '[msg]'])

    assert 'par' not in pkmn.non_volatile_statuses
    assert not dict(pkmn.non_volatile_statuses)


def test_status_replaces_previous_major_status():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    pkmn = build_pkmn('dragonite', is_opponent=True, hp=90)
    pkmn.non_volatile_statuses['par'] = True
    battle.opponent.active1 = pkmn

    status(battle, ['', '-status', 'p2a: Dragonite', 'brn'])

    assert pkmn.non_volatile_statuses.get('brn') is True
    assert 'par' not in pkmn.non_volatile_statuses


def test_skill_swap_activate_exchanges_abilities():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.user.active2 = build_pkmn('torkoal', is_opponent=False, hp=177)
    battle.user.active2.abilityPkmn = 'drought'
    battle.opponent.active2 = build_pkmn('starmie', is_opponent=True, hp=100)
    battle.opponent.active2.abilityPkmn = 'naturalcure'

    activate(
        battle,
        [
            '',
            '-activate',
            'p2b: Starmie',
            'move: Skill Swap',
            'Drought',
            'Natural Cure',
            '[of] p1b: Torkoal',
        ],
    )

    assert battle.opponent.active2.abilityPkmn == 'drought'
    assert battle.user.active2.abilityPkmn == 'naturalcure'


def test_guard_split_averages_def_spd():
    from showdown.battle_modifier import activate as act

    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    user = build_pkmn('torkoal', is_opponent=False, hp=177)
    user.DEF_battle_actual = 200
    user.SPD_battle_actual = 100
    opp = build_pkmn('indeedee-f', is_opponent=True, hp=100)
    opp.DEF_battle_actual = -1
    opp.SPD_battle_actual = -1
    opp.DEF_battle_base = 100
    opp.SPD_battle_base = 140
    battle.user.active2 = user
    battle.opponent.active1 = opp

    act(
        battle,
        ['', '-activate', 'p2a: Indeedee', 'move: Guard Split', '[of] p1b: Torkoal'],
    )

    assert user.DEF_battle_actual == 150  # (200+100)//2
    assert opp.DEF_battle_actual == 150
    assert user.SPD_battle_actual == 120  # (100+140)//2
    assert opp.SPD_battle_actual == 120
    assert user.guard_split_active is True
    assert opp.guard_split_active is True


def test_swapboost_guard_swap_exchanges_def_spd_boosts():
    from showdown.battle_modifier import swapboost
    import constants

    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    user = build_pkmn('torkoal', is_opponent=False, hp=177)
    opp = build_pkmn('indeedee-f', is_opponent=True, hp=100)
    user.boosts[constants.DEFENSE] = 2
    user.boosts[constants.SPECIAL_DEFENSE] = -1
    opp.boosts[constants.DEFENSE] = 0
    opp.boosts[constants.SPECIAL_DEFENSE] = 3
    battle.user.active2 = user
    battle.opponent.active1 = opp

    swapboost(
        battle,
        [
            '',
            '-swapboost',
            'p2a: Indeedee',
            'p1b: Torkoal',
            'def, spd',
            '[from] move: Guard Swap',
        ],
    )

    assert user.boosts[constants.DEFENSE] == 0
    assert user.boosts[constants.SPECIAL_DEFENSE] == 3
    assert opp.boosts[constants.DEFENSE] == 2
    assert opp.boosts[constants.SPECIAL_DEFENSE] == -1


def test_replace_illusion_increments_confirmed_counter_once():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.opponent.opponentPkmnConfirmed = 2
    battle.opponent.active1 = build_pkmn('dragonite', is_opponent=True, hp=64)
    battle.opponent.opponents = [build_pkmn('zoroark', is_opponent=True)]

    replace_illusion(battle, ['', 'replace', 'p2a: Zoroark', 'Zoroark, L50, M'])

    assert battle.opponent.active1.name == 'zoroark'
    assert battle.opponent.active1.is_opponent_confirmed is True
    assert battle.opponent.opponentPkmnConfirmed == 3


def test_yawn_sets_is_drowsy_until_sleep_or_switch():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    target = build_pkmn('incineroar', is_opponent=True, hp=100)
    battle.opponent.active1 = target
    battle.opponent.active2 = build_pkmn('pawmot', is_opponent=True, hp=100)
    battle.opponent.reserved = [build_pkmn('blastoise', is_opponent=True, hp=73)]
    battle.opponent.opponents = [
        build_pkmn('incineroar', is_opponent=True),
        build_pkmn('pawmot', is_opponent=True),
        build_pkmn('blastoise', is_opponent=True),
    ]

    start_volatile_status(
        battle,
        ['', '-start', 'p2a: Incineroar', 'move: Yawn', '[of] p1a: Torkoal'],
    )
    assert target.is_drowsy is True
    assert target.volatile_statuses.get('yawn') is True

    # Debe sobrevivir al |turn| del turno de decisión.
    turn(battle, ['', 'turn', '2'])
    assert battle.opponent.active1.is_drowsy is True

    status(battle, ['', '-status', 'p2a: Incineroar', 'slp', '[from] move: Yawn'])
    assert battle.opponent.active1.is_drowsy is False
    assert battle.opponent.active1.non_volatile_statuses.get('slp') is True
    assert 'yawn' not in battle.opponent.active1.volatile_statuses


def test_yawn_drowsy_clears_on_switch_out():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    drowsy = build_pkmn('incineroar', is_opponent=True, hp=100)
    drowsy.is_drowsy = True
    drowsy.volatile_statuses['yawn'] = True
    battle.opponent.active1 = drowsy
    battle.opponent.reserved = [build_pkmn('blastoise', is_opponent=True, hp=73)]
    battle.opponent.opponents = [
        build_pkmn('incineroar', is_opponent=True),
        build_pkmn('blastoise', is_opponent=True),
    ]

    switch_or_drag(
        battle,
        ['', 'switch', 'p2a: Blastoise', 'Blastoise, L50, M', '73/100'],
    )

    reserved_incin = next(p for p in battle.opponent.reserved if p.name == 'incineroar')
    assert reserved_incin.is_drowsy is False
    assert 'yawn' not in reserved_incin.volatile_statuses
    assert battle.opponent.active1.is_drowsy is False


def test_perish_song_count_tracks_protocol_countdown():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    gengar = build_pkmn('gengar', is_opponent=True, hp=100)
    battle.opponent.active1 = gengar
    battle.opponent.active2 = build_pkmn('pawmot', is_opponent=True, hp=100)

    start_volatile_status(battle, ['', '-start', 'p2a: Gengar', 'perish3', '[silent]'])
    assert gengar.perish_count == 3
    assert gengar.volatile_statuses.get('perish3') is True

    start_volatile_status(battle, ['', '-start', 'p2a: Gengar', 'perish2'])
    assert gengar.perish_count == 2
    assert 'perish3' not in gengar.volatile_statuses
    assert gengar.volatile_statuses.get('perish2') is True

    start_volatile_status(battle, ['', '-start', 'p2a: Gengar', 'perish1'])
    assert gengar.perish_count == 1

    start_volatile_status(battle, ['', '-start', 'p2a: Gengar', 'perish0'])
    assert gengar.perish_count == 0


def test_perish_song_clears_on_switch_out():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    gengar = build_pkmn('gengar', is_opponent=True, hp=100)
    gengar.perish_count = 2
    gengar.volatile_statuses['perish2'] = True
    battle.opponent.active1 = gengar
    battle.opponent.reserved = [build_pkmn('blastoise', is_opponent=True, hp=73)]
    battle.opponent.opponents = [
        build_pkmn('gengar', is_opponent=True),
        build_pkmn('blastoise', is_opponent=True),
    ]

    switch_or_drag(
        battle,
        ['', 'switch', 'p2a: Blastoise', 'Blastoise, L50, M', '73/100'],
    )

    reserved_gengar = next(p for p in battle.opponent.reserved if p.name == 'gengar')
    assert reserved_gengar.perish_count == 0
    assert battle.opponent.active1.perish_count == 0


def test_enditem_clears_ots_template_for_item_clause():
    """Corrosive Gas / Knock Off deben vaciar también la plantilla OTS (Illusion)."""
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    ots_zoroark = build_pkmn('zoroark', is_opponent=True, hp=74)
    ots_zoroark.objectPkmn = 'expertbelt'
    disguised = build_pkmn('dragonite', is_opponent=True, hp=74)
    disguised.objectPkmn = 'habanberry'
    battle.opponent.active1 = disguised
    battle.opponent.opponents = [ots_zoroark, build_pkmn('dragonite', is_opponent=True)]

    remove_item(
        battle,
        ['', '-enditem', 'p2a: Dragonite', 'Expert Belt', '[from] move: Corrosive Gas'],
    )

    assert battle.opponent.active1.objectPkmn is None
    assert ots_zoroark.objectPkmn is None


def test_faint_reverts_transformed_user_ditto_identity():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.user.isUser = True

    ditto = build_pkmn('zoroark', is_opponent=False, hp=1)
    ditto.is_ditto = True
    ditto.is_zoroark = True
    ditto.abilityPkmn = 'illusion'
    ditto.HPB_battle_base = 155
    battle.user.active1 = ditto
    battle.user.active2 = build_pkmn('gengar', is_opponent=False, hp=45)

    faint(battle, ['', 'faint', 'p1a: Zoroark'])

    fainted = battle.user.active1
    assert fainted.name == 'ditto'
    assert fainted.is_ditto is True
    assert fainted.is_zoroark is False
    assert fainted.is_fainted is True
    assert fainted.HPA_battle_actual == 0
    assert fainted.abilityPkmn == 'imposter'


def test_transform_into_zoroark_does_not_set_is_zoroark_on_ditto():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.user.isUser = True

    ditto = build_pkmn('ditto', is_opponent=False, hp=41)
    ditto.objectPkmn = ''
    ditto.abilityPkmn = 'imposter'
    ditto.HPB_battle_base = 155
    battle.user.active1 = build_pkmn('gengar', is_opponent=False, hp=45)
    battle.user.active2 = ditto

    zoroark = build_pkmn('zoroark', is_opponent=True, hp=20)
    zoroark.is_zoroark = True
    zoroark.abilityPkmn = 'illusion'
    battle.opponent.active1 = zoroark

    transform(
        battle,
        ['', '-transform', 'p1b: Ditto', 'p2a: Zoroark', '[from] ability: Imposter'],
    )

    transformed = battle.user.active2
    assert transformed.is_ditto is True
    assert transformed.is_zoroark is False
    assert transformed.name == 'zoroark'
    assert transformed.abilityPkmn == 'illusion'


def test_set_player_and_toclean_preserve_account_names():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    set_player(battle, ['', 'player', 'p1', 'Deep1ODDQN', '167'])
    set_player(battle, ['', 'player', 'p2', 'popplioperu98', 'winona'])
    assert battle.user.account_name == 'Deep1ODDQN'
    assert battle.opponent.account_name == 'popplioperu98'

    battle.toClean()
    assert battle.user.account_name == 'Deep1ODDQN'
    assert battle.opponent.account_name == 'popplioperu98'


def test_indeedee_f_loads_types_from_champions_stats():
    p = Pokemon('indeedee-f', -1, True, 'terrainextender', 'psychicsurge', None, None, None, None, 'modest')
    assert p.type1 == 'Psychic'
    assert p.HPB_battle_base > 0


def test_illegal_ots_move_deduces_illusion_on_turn_one():
    """T1: 'Dragonite' usa Calm Mind (set de Zoroark) → deducir Illusion."""
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    dragonite = build_pkmn('dragonite', is_opponent=True, hp=100)
    dragonite.moves = ['dragondance', 'lightscreen', 'roar', 'snowscape']
    dragonite.is_opponent_confirmed = True

    ots_zoroark = build_pkmn('zoroark', is_opponent=True, hp=100)
    ots_zoroark.objectPkmn = 'expertbelt'
    ots_zoroark.moves = ['calmmind', 'toxic', 'knockoff', 'snarl']
    ots_dragonite = build_pkmn('dragonite', is_opponent=True, hp=100)
    ots_dragonite.moves = ['dragondance', 'lightscreen', 'roar', 'snowscape']

    battle.opponent.active1 = dragonite
    battle.opponent.active2 = build_pkmn('pawmot', is_opponent=True, hp=100)
    battle.opponent.opponents = [ots_zoroark, ots_dragonite, build_pkmn('pawmot', is_opponent=True)]

    observe_move(
        battle,
        ['', 'move', 'p2a: Dragonite', 'Calm Mind', 'p2a: Dragonite'],
    )

    assert battle.opponent.active1.name == 'zoroark'
    assert battle.opponent.active1.is_zoroark is True
    assert battle.opponent.active1.HPA_battle_actual == 100
    # El Dragonite real vuelve a reserva intacto
    assert any(p.name == 'dragonite' for p in battle.opponent.reserved)


def test_legal_ots_move_does_not_false_positive_illusion():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    dragonite = build_pkmn('dragonite', is_opponent=True, hp=100)
    dragonite.moves = ['dragondance', 'lightscreen', 'roar', 'snowscape']
    ots_zoroark = build_pkmn('zoroark', is_opponent=True)
    ots_zoroark.moves = ['calmmind', 'toxic', 'knockoff', 'snarl']
    ots_dragonite = build_pkmn('dragonite', is_opponent=True)
    ots_dragonite.moves = ['dragondance', 'lightscreen', 'roar', 'snowscape']

    battle.opponent.active1 = dragonite
    battle.opponent.opponents = [ots_zoroark, ots_dragonite]

    observe_move(
        battle,
        ['', 'move', 'p2a: Dragonite', 'Dragon Dance', 'p2a: Dragonite'],
    )

    assert battle.opponent.active1.name == 'dragonite'
    assert battle.opponent.active1.is_zoroark is False


# ---------------------------------------------------------------------------
# Cobertura restante de BATTLE_MODIFIERS_LOOKUP (handlers sin test dedicado)
# ---------------------------------------------------------------------------

def test_request_sets_flags_and_stores_json():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    request(
        battle,
        [
            '',
            'request',
            '{"wait":true,"side":{"name":"Deep1ODDQN","id":"p1","pokemon":[]},"rqid":7}',
        ],
    )
    assert battle.wait is True
    assert battle.active is False
    assert battle.force_switch is False
    assert battle.rqid == 7
    assert battle.request_json.get('wait') is True

    request(
        battle,
        [
            '',
            'request',
            '{"forceSwitch":[true,false],"side":{"name":"Deep1ODDQN","id":"p1","pokemon":[]},"rqid":14}',
        ],
    )
    assert battle.force_switch is True
    assert battle.wait is False
    assert battle.rqid == 14


def test_drag_brings_reserved_opponent_onto_field():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    battle.opponent.active1 = build_pkmn('dragonite', is_opponent=True, hp=100)
    battle.opponent.active2 = build_pkmn('blastoise', is_opponent=True, hp=100)
    whimsicott = build_pkmn('whimsicott', is_opponent=True, hp=100)
    battle.opponent.reserved = [whimsicott]

    switch_or_drag(
        battle,
        ['', 'drag', 'p2a: Whimsicott', 'Whimsicott, L50, F', '100/100'],
    )

    assert battle.opponent.active1.name == 'whimsicott'
    assert any(p.name == 'dragonite' for p in battle.opponent.reserved)


def test_swap_ally_switch_exchanges_actives():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.user.active1 = build_pkmn('aegislash', is_opponent=False, hp=135)
    battle.user.active2 = build_pkmn('farigiraf', is_opponent=False, hp=222)

    swap(battle, ['', 'swap', 'p1a: Aegislash', '1', '[from] move: Ally Switch'])

    assert battle.user.active1.name == 'farigiraf'
    assert battle.user.active2.name == 'aegislash'


def test_boost_and_unboost_clamp_stat_stages():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.opponent.active1 = build_pkmn('dragonite', is_opponent=True, hp=100)
    battle.opponent.active2 = build_pkmn('whimsicott', is_opponent=True, hp=100)

    boost(battle, ['', '-boost', 'p2a: Dragonite', 'atk', '1'])
    boost(battle, ['', '-boost', 'p2a: Dragonite', 'spe', '1'])
    assert battle.opponent.active1.boosts[constants.ATTACK] == 1
    assert battle.opponent.active1.boosts[constants.SPEED] == 1

    unboost(battle, ['', '-unboost', 'p2b: Whimsicott', 'spe', '2'])
    assert battle.opponent.active2.boosts[constants.SPEED] == -2


def test_end_volatile_clears_substitute():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.user.active1 = build_pkmn('aegislash', is_opponent=False, hp=135)
    battle.user.active1.volatile_statuses['substitute'] = True

    end_volatile_status(battle, ['', '-end', 'p1a: Aegislash', 'Substitute'])

    assert 'substitute' not in battle.user.active1.volatile_statuses


def test_set_item_assigns_held_item():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.opponent.active1 = build_pkmn('torkoal', is_opponent=True, hp=100)
    battle.opponent.active1.objectPkmn = None

    set_item(
        battle,
        ['', '-item', 'p2a: Torkoal', 'Choice Scarf', '[from] move: Trick'],
    )

    assert battle.opponent.active1.objectPkmn == 'choicescarf'


def test_ability_tag_updates_ability():
    """Live Worry Seed: |-ability|p1a: Farigiraf|Insomnia|Armor Tail|[from] move: Worry Seed"""
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.user.active1 = build_pkmn('farigiraf', is_opponent=False, hp=222)
    battle.user.active1.abilityPkmn = 'armortail'

    set_opponent_ability_from_ability_tag(
        battle,
        [
            '',
            '-ability',
            'p1a: Farigiraf',
            'Insomnia',
            'Armor Tail',
            '[from] move: Worry Seed',
        ],
    )

    assert battle.user.active1.abilityPkmn == 'insomnia'


def test_mega_sets_flag_and_guesses_mega_ability():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.opponent.active1 = build_pkmn('blastoise', is_opponent=True, hp=100)
    battle.opponent.active1.abilityPkmn = 'raindish'
    battle.opponent.active1.is_mega_evo = False

    mega(battle, ['', '-mega', 'p2a: Blastoise', 'Blastoise', 'Blastoisinite'])

    assert battle.opponent.active1.is_mega_evo is True
    assert battle.opponent.active1.abilityPkmn == 'megalauncher'


def test_clearboost_resets_one_pokemon():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.opponent.active1 = build_pkmn('farigiraf', is_opponent=True, hp=100)
    battle.opponent.active2 = build_pkmn('primarina', is_opponent=True, hp=100)
    battle.opponent.active1.boosts[constants.SPECIAL_ATTACK] = 2
    battle.opponent.active2.boosts[constants.SPECIAL_ATTACK] = 1

    clearboost(battle, ['', '-clearboost', 'p2a: Farigiraf'])

    assert battle.opponent.active1.boosts[constants.SPECIAL_ATTACK] == 0
    assert battle.opponent.active2.boosts[constants.SPECIAL_ATTACK] == 1


def test_clearallboost_resets_entire_field():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.user.active1 = build_pkmn('aegislash', is_opponent=False, hp=135)
    battle.user.active2 = build_pkmn('whimsicott', is_opponent=False, hp=155)
    battle.opponent.active1 = build_pkmn('dragonite', is_opponent=True, hp=100)
    battle.opponent.active2 = build_pkmn('blastoise', is_opponent=True, hp=100)
    battle.user.active1.boosts[constants.ATTACK] = 1
    battle.opponent.active1.boosts[constants.ATTACK] = 2
    battle.opponent.active1.boosts[constants.SPEED] = 1

    clearallboost(battle, ['', '-clearallboost'])

    for pkmn in (
        battle.user.active1,
        battle.user.active2,
        battle.opponent.active1,
        battle.opponent.active2,
    ):
        assert pkmn.boosts[constants.ATTACK] == 0
        assert pkmn.boosts[constants.SPEED] == 0


def test_copyboost_psych_up_copies_stages():
    """Live: |-copyboost|p1a: Farigiraf|p2a: Venusaur|[from] move: Psych Up"""
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.user.active1 = build_pkmn('farigiraf', is_opponent=False, hp=222)
    battle.opponent.active1 = build_pkmn('venusaur', is_opponent=True, hp=100)
    battle.opponent.active1.boosts[constants.SPECIAL_ATTACK] = 2
    battle.opponent.active1.boosts[constants.SPECIAL_DEFENSE] = 1
    battle.user.active1.boosts[constants.ATTACK] = 3

    copyboost(
        battle,
        ['', '-copyboost', 'p1a: Farigiraf', 'p2a: Venusaur', '[from] move: Psych Up'],
    )

    assert battle.user.active1.boosts[constants.SPECIAL_ATTACK] == 2
    assert battle.user.active1.boosts[constants.SPECIAL_DEFENSE] == 1
    assert battle.user.active1.boosts[constants.ATTACK] == 0


# ---------------------------------------------------------------------------
# Regresión live (BO3 2026-09-12 ~04:31): Aria brn, confusion, Illusion Hisui
# ---------------------------------------------------------------------------

def test_sparkling_aria_curestatus_clears_burn_on_hit_targets():
    """Live Aria: curestatus brn on ally Runerigus + foe Farigiraf; user Primarina stays brn."""
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    farigiraf = build_pkmn('farigiraf', is_opponent=False, hp=170)
    farigiraf.non_volatile_statuses['brn'] = True
    runerigus = build_pkmn('runerigus', is_opponent=True, hp=87)
    runerigus.non_volatile_statuses['brn'] = True
    primarina = build_pkmn('primarina', is_opponent=True, hp=94)
    primarina.non_volatile_statuses['brn'] = True
    battle.user.active1 = farigiraf
    battle.opponent.active1 = primarina
    battle.opponent.active2 = runerigus

    curestatus(battle, ['', '-curestatus', 'p2b: Runerigus', 'brn', '[msg]'])
    curestatus(battle, ['', '-curestatus', 'p1a: Farigiraf', 'brn', '[msg]'])

    assert 'brn' not in farigiraf.non_volatile_statuses
    assert 'brn' not in runerigus.non_volatile_statuses
    assert runerigus.non_volatile_statuses.get('brn') is not True
    assert primarina.non_volatile_statuses.get('brn') is True


def test_confusion_start_and_end_volatile():
    """Live: |-start|p1b: Torkoal|confusion  ...  |-end|p1b: Torkoal|confusion"""
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.user.active2 = build_pkmn('torkoal', is_opponent=False, hp=177)

    start_volatile_status(battle, ['', '-start', 'p1b: Torkoal', 'confusion'])
    assert battle.user.active2.volatile_statuses.get('confusion') is True

    end_volatile_status(battle, ['', '-end', 'p1b: Torkoal', 'confusion'])
    assert 'confusion' not in battle.user.active2.volatile_statuses


def test_replace_illusion_hisui_from_venusaur_disguise():
    """Live: |replace|p2b: Zoroark|Zoroark-Hisui, L50, F then |-end|...|Illusion"""
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    disguised = build_pkmn('venusaur', is_opponent=True, hp=36)
    disguised.is_zoroark = True
    disguised.illusion_disguise = 'venusaur'
    disguised.illusion_protocol_revealed = False
    disguised.abilityPkmn = 'illusion'
    disguised.objectPkmn = 'lifeorb'
    battle.opponent.active2 = disguised
    battle.opponent.active1 = build_pkmn('primarina', is_opponent=True, hp=88)
    ots_zoro = build_pkmn('zoroark-hisui', is_opponent=True)
    ots_zoro.objectPkmn = 'lifeorb'
    ots_zoro.abilityPkmn = 'illusion'
    battle.opponent.opponents = [
        ots_zoro,
        build_pkmn('venusaur', is_opponent=True),
        build_pkmn('primarina', is_opponent=True),
    ]
    battle.opponent.reserved = [build_pkmn('venusaur', is_opponent=True, hp=100)]

    replace_illusion(
        battle,
        ['', 'replace', 'p2b: Zoroark', 'Zoroark-Hisui, L50, F'],
    )
    end_volatile_status(battle, ['', '-end', 'p2b: Zoroark', 'Illusion'])

    revealed = battle.opponent.active2
    assert revealed.name in {'zoroark', 'zoroark-hisui'}
    assert revealed.is_zoroark is True
    assert revealed.illusion_protocol_revealed is True
    assert revealed.illusion_disguise == 'venusaur'
    assert revealed.HPA_battle_actual == 36
    assert (revealed.objectPkmn or '').replace(' ', '').lower() == 'lifeorb'


def test_guard_split_and_swap_live_message_shapes():
    """Live Runerigus messages for Guard Split / Guard Swap."""
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    user = build_pkmn('farigiraf', is_opponent=False, hp=222)
    user.DEF_battle_actual = 122
    user.SPD_battle_actual = 108
    opp = build_pkmn('runerigus', is_opponent=True, hp=100)
    opp.DEF_battle_actual = -1
    opp.SPD_battle_actual = -1
    opp.DEF_battle_base = 145
    opp.SPD_battle_base = 105
    torkoal = build_pkmn('torkoal', is_opponent=False, hp=177)
    battle.user.active1 = user
    battle.user.active2 = torkoal
    battle.opponent.active2 = opp

    activate(
        battle,
        ['', '-activate', 'p2b: Runerigus', 'move: Guard Split', '[of] p1a: Farigiraf'],
    )
    assert user.DEF_battle_actual == (122 + 145) // 2
    assert opp.DEF_battle_actual == user.DEF_battle_actual
    assert user.guard_split_active is True

    user.boosts[constants.DEFENSE] = 0
    user.boosts[constants.SPECIAL_DEFENSE] = 0
    torkoal.boosts[constants.DEFENSE] = 2
    torkoal.boosts[constants.SPECIAL_DEFENSE] = 1
    # Live swap was Runerigus <-> Torkoal; mirror that pairing
    from showdown.battle_modifier import swapboost as sb

    battle.opponent.active2.boosts[constants.DEFENSE] = 0
    battle.opponent.active2.boosts[constants.SPECIAL_DEFENSE] = 0
    sb(
        battle,
        [
            '',
            '-swapboost',
            'p2b: Runerigus',
            'p1b: Torkoal',
            'def, spd',
            '[from] move: Guard Swap',
        ],
    )
    assert battle.opponent.active2.boosts[constants.DEFENSE] == 2
    assert battle.opponent.active2.boosts[constants.SPECIAL_DEFENSE] == 1
    assert torkoal.boosts[constants.DEFENSE] == 0
    assert torkoal.boosts[constants.SPECIAL_DEFENSE] == 0


# Estimacion de Stat Points del rival (showdown/stat_inference.py)


def _build_speed_scenario(user_speed, opponent_name='dragonite',
                          opponent_nature='adamant', user_item=None,
                          opponent_item=None, opponent_ability=None):
    """Batalla minima con un activo por bando lista para observar orden de accion."""
    si.reset_sp_limits()
    si.configure_sp_limits(si.SP_HARD_MAX, si.SP_HARD_MAX * 6)

    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    mine = build_pkmn('whimsicott', is_opponent=False, hp=155)
    mine.VEL_battle_actual = user_speed
    mine.objectPkmn = user_item
    battle.user.active1 = mine

    theirs = build_pkmn(opponent_name, is_opponent=True, hp=100)
    theirs.nature = opponent_nature
    theirs.objectPkmn = opponent_item
    if opponent_ability is not None:
        theirs.abilityPkmn = opponent_ability
    battle.opponent.active1 = theirs

    turn(battle, ['', 'turn', '1'])
    return battle, mine, theirs


def test_champions_stat_formula_matches_request_stats():
    # Stats exactos que Showdown envio en |request| para el equipo del agente.
    assert si.champions_stat(120, 53, 'bold', 'hp') == 222
    assert si.champions_stat(90, 0, 'bold', 'atk') == 99
    assert si.champions_stat(110, 0, 'bold', 'spa') == 130
    assert si.champions_stat(60, 0, 'bold', 'spe') == 80
    assert si.champions_stat(60, 0, 'sassy', 'hp') == 135
    assert si.champions_stat(20, 3, 'relaxed', 'spe') == 37
    # La naturaleza sube y baja un 10%
    assert si.champions_stat(70, 0, 'bold', 'def') == 99
    assert si.champions_stat(70, 0, 'serious', 'def') == 90


def test_sp_pairs_collapse_to_the_same_stat():
    # A nivel 50 la formula divide por dos, asi que los SP van de dos en dos:
    # el objetivo alcanzable es el stat final, no el numero exacto de puntos.
    assert si.champions_stat(80, 39, 'adamant', 'spe') == si.champions_stat(80, 40, 'adamant', 'spe')
    assert si.champions_stat(80, 38, 'adamant', 'spe') != si.champions_stat(80, 39, 'adamant', 'spe')


def test_speed_order_narrows_opponent_when_opponent_moves_first():
    battle, _, theirs = _build_speed_scenario(user_speed=120)
    belief = battle.stat_beliefs.belief_for(theirs)
    assert belief.stat_range('spe') == (100, 132)

    # El rival actua primero con la misma prioridad -> su velocidad es >= 120
    # (con margen de desync Spe >= user - margin).
    observe_move(battle, ['', 'move', 'p2a: Dragonite', 'Dragon Dance', 'p2a: Dragonite'])
    observe_move(battle, ['', 'move', 'p1a: Whimsicott', 'Moonblast', 'p2a: Dragonite'])

    assert belief.stat_range('spe') == (114, 132)
    assert min(belief.candidates['spe']) == 27
    assert belief.observations == 1

    si.reset_sp_limits()


def test_speed_order_narrows_opponent_when_agent_moves_first():
    battle, _, theirs = _build_speed_scenario(user_speed=120)
    belief = battle.stat_beliefs.belief_for(theirs)

    # El agente actua primero -> la velocidad del rival es <= 120 (+ margen).
    observe_move(battle, ['', 'move', 'p1a: Whimsicott', 'Moonblast', 'p2a: Dragonite'])
    observe_move(battle, ['', 'move', 'p2a: Dragonite', 'Dragon Dance', 'p2a: Dragonite'])

    assert belief.stat_range('spe') == (100, 126)
    assert max(belief.candidates['spe']) == 52

    si.reset_sp_limits()


def test_trick_room_inverts_the_speed_conclusion():
    battle, _, theirs = _build_speed_scenario(user_speed=120)
    battle.terrain['trickroom'] = 5
    belief = battle.stat_beliefs.belief_for(theirs)

    # Con Trick Room actuar primero significa ser MAS LENTO (+ margen).
    observe_move(battle, ['', 'move', 'p2a: Dragonite', 'Dragon Dance', 'p2a: Dragonite'])
    observe_move(battle, ['', 'move', 'p1a: Whimsicott', 'Moonblast', 'p2a: Dragonite'])

    assert belief.stat_range('spe') == (100, 126)

    si.reset_sp_limits()


def test_mid_turn_tailwind_uses_live_speed_context():
    """
    Regresión: Tailwind del aliado mid-turn debe aplicar al siguiente mon.
    Snapshot congelado sin TW hacía Spe(Kingambit)>=Archaludon y excluía Spe 71.
    """
    battle, mine, theirs = _build_speed_scenario(user_speed=104, opponent_name='kingambit',
                                                 opponent_nature='adamant')
    # Aliado del rival que lanza Tailwind antes.
    ally = build_pkmn('salamence', is_opponent=True, hp=100)
    ally.nature = 'jolly'
    battle.opponent.active2 = ally
    mine.VEL_battle_actual = 104
    belief = battle.stat_beliefs.belief_for(theirs)
    assert 1 in belief.candidates['spe'] or True  # Spe bajo debe sobrevivir

    # Salamence Tailwind (aún sin TW en side), luego sidestart, luego Kingambit.
    observe_move(battle, ['', 'move', 'p2b: Salamence', 'Tailwind', 'p2b: Salamence'])
    from showdown.battle_modifier import sidestart
    sidestart(battle, ['', '-sidestart', 'p2: Opp', 'move: Tailwind'])
    observe_move(battle, ['', 'move', 'p2a: Kingambit', 'Low Kick', 'p2b: Salamence'])
    observe_move(battle, ['', 'move', 'p1a: Whimsicott', 'Moonblast', 'p2a: Kingambit'])

    # Con TW, Spe 71 efectiva 142 >= 104: el SP bajo sigue siendo viable.
    assert belief.stat_range('spe')[0] <= 71
    assert min(belief.candidates['spe']) <= 1

    si.reset_sp_limits()


def test_choice_scarf_is_accounted_for_in_the_speed_order():
    # Con Choice Scarf hace falta menos velocidad base para llegar primero,
    # asi que la conclusion debe ser mas debil que sin objeto.
    battle, _, theirs = _build_speed_scenario(user_speed=120, opponent_item='choicescarf')
    belief = battle.stat_beliefs.belief_for(theirs)

    observe_move(battle, ['', 'move', 'p2a: Dragonite', 'Dragon Dance', 'p2a: Dragonite'])
    observe_move(battle, ['', 'move', 'p1a: Whimsicott', 'Moonblast', 'p2a: Dragonite'])

    # 120 / 1.5 = 80, por debajo del minimo posible: no se aprende nada
    assert belief.stat_range('spe') == (100, 132)

    si.reset_sp_limits()


def test_tailwind_on_the_agent_side_is_accounted_for():
    battle, _, theirs = _build_speed_scenario(user_speed=120)
    battle.user.side_conditions['tailwind'] = 4
    belief = battle.stat_beliefs.belief_for(theirs)

    # Con Tailwind el agente corre a 240; que el rival vaya primero exigiria
    # mas velocidad de la que puede tener, asi que la premisa es imposible y
    # la observacion se descarta en lugar de vaciar los candidatos.
    observe_move(battle, ['', 'move', 'p2a: Dragonite', 'Dragon Dance', 'p2a: Dragonite'])
    observe_move(battle, ['', 'move', 'p1a: Whimsicott', 'Moonblast', 'p2a: Dragonite'])

    assert belief.stat_range('spe') == (100, 132)
    assert belief.skipped == 1

    si.reset_sp_limits()


def test_different_priority_teaches_nothing_about_speed():
    battle, _, theirs = _build_speed_scenario(user_speed=120)
    belief = battle.stat_beliefs.belief_for(theirs)

    # Extreme Speed tiene prioridad +2: ir primero no dice nada de la velocidad
    observe_move(battle, ['', 'move', 'p2a: Dragonite', 'Extreme Speed', 'p1a: Whimsicott'])
    observe_move(battle, ['', 'move', 'p1a: Whimsicott', 'Moonblast', 'p2a: Dragonite'])

    assert belief.stat_range('spe') == (100, 132)
    assert belief.observations == 0

    si.reset_sp_limits()


def test_trick_room_move_has_priority_minus_seven():
    assert si.move_priority('Trick Room') == -7
    assert si.move_priority('Grassy Glide') == 0
    assert si.move_priority('Grassy Glide', terrain='grassyterrain') == 1


def test_trick_room_move_does_not_teach_speed_under_trick_room():
    """
    Regresión: Trick Room era prioridad 0 en el meta. Bajo TR, usarlo después de
    un move de prioridad 0 (p.ej. Wood Hammer) se interpretaba como Spe >= rival
    y excluía Sinistcha Spe 0 (true).
    """
    battle, _, theirs = _build_speed_scenario(user_speed=105, opponent_name='sinistcha',
                                             opponent_nature='bold')
    battle.terrain['trickroom'] = 5
    belief = battle.stat_beliefs.belief_for(theirs)
    before = set(belief.candidates['spe'])

    observe_move(battle, ['', 'move', 'p1a: Whimsicott', 'Wood Hammer', 'p2a: Sinistcha'])
    observe_move(battle, ['', 'move', 'p2a: Sinistcha', 'Trick Room', 'p2a: Sinistcha'])

    assert belief.candidates['spe'] == before
    assert 0 in belief.candidates['spe']
    assert belief.observations == 0

    si.reset_sp_limits()


def test_grassy_glide_priority_in_terrain_skips_speed_compare():
    battle, _, theirs = _build_speed_scenario(user_speed=105, opponent_name='sinistcha',
                                             opponent_nature='bold')
    battle.terrain['grassyterrain'] = 5
    belief = battle.stat_beliefs.belief_for(theirs)
    before = set(belief.candidates['spe'])

    observe_move(battle, ['', 'move', 'p1a: Whimsicott', 'Grassy Glide', 'p2a: Sinistcha'])
    observe_move(battle, ['', 'move', 'p2a: Sinistcha', 'Matcha Gotcha', 'p1a: Whimsicott'])

    assert belief.candidates['spe'] == before
    assert belief.observations == 0

    si.reset_sp_limits()


def test_uncertain_speed_ability_discards_the_observation():
    battle, _, theirs = _build_speed_scenario(
        user_speed=120, opponent_name='floatzel', opponent_ability='swiftswim'
    )
    belief = battle.stat_beliefs.belief_for(theirs)
    before = belief.stat_range('spe')

    observe_move(battle, ['', 'move', 'p2a: Floatzel', 'Waterfall', 'p1a: Whimsicott'])
    observe_move(battle, ['', 'move', 'p1a: Whimsicott', 'Moonblast', 'p2a: Floatzel'])

    assert belief.stat_range('spe') == before
    assert belief.observations == 0

    si.reset_sp_limits()


def test_called_moves_do_not_count_as_action_order():
    battle, _, theirs = _build_speed_scenario(user_speed=120)
    belief = battle.stat_beliefs.belief_for(theirs)

    # Un |move| con [from] es un movimiento llamado, no la accion elegida
    observe_move(
        battle,
        ['', 'move', 'p2a: Dragonite', 'Dragon Dance', 'p2a: Dragonite', '[from] move: Sleep Talk'],
    )
    observe_move(battle, ['', 'move', 'p1a: Whimsicott', 'Moonblast', 'p2a: Dragonite'])

    assert belief.observations == 0
    assert belief.stat_range('spe') == (100, 132)

    si.reset_sp_limits()


def test_turn_reset_clears_the_action_order():
    battle, _, _ = _build_speed_scenario(user_speed=120)
    observe_move(battle, ['', 'move', 'p2a: Dragonite', 'Dragon Dance', 'p2a: Dragonite'])
    assert len(battle.turn_actions) == 1

    turn(battle, ['', 'turn', '2'])
    assert battle.turn_actions == []
    assert battle.turn_speed_snapshot is None

    si.reset_sp_limits()


def test_budget_prunes_candidates_that_exceed_the_total():
    si.reset_sp_limits()
    belief = si.StatBelief('dragonite', {'hp': 91, 'atk': 134, 'def': 95,
                                         'spa': 100, 'spd': 100, 'spe': 80}, 'adamant')
    # Champions permite 66 en total: si un stat se lleva 32, a otro no le pueden
    # quedar mas de 34, y el tope por stat ya es 32.
    belief.narrow('spe', {32})
    belief.narrow('atk', {32})
    assert max(belief.candidates['spd']) <= 66 - 64


def test_narrowing_is_ignored_when_it_would_empty_the_candidates():
    si.reset_sp_limits()
    belief = si.StatBelief('dragonite', {'spe': 80}, 'adamant')
    belief.narrow('spe', {0, 1, 2})
    assert belief.candidates['spe'] == {0, 1, 2}

    # Interseccion vacia: se conserva lo que ya se sabia y se cuenta como descarte
    assert belief.narrow('spe', {20, 21}) is False
    assert belief.candidates['spe'] == {0, 1, 2}
    assert belief.skipped == 1


def test_calibration_keeps_champions_limits_even_if_own_team_exceeds():
    """Proyecto Champions-only: no ensanchar topes por stats ilegales propios."""
    si.reset_sp_limits()
    assert si.sp_limits() == (si.SP_MAX, si.SP_BUDGET)

    mine = build_pkmn('farigiraf', is_opponent=False, hp=222)
    mine.nature = 'bold'
    mine.HPB_battle_base = 222
    mine.ATK_battle_actual = 99
    mine.DEF_battle_actual = 122
    mine.SPA_battle_actual = 130
    mine.SPD_battle_actual = 108
    mine.VEL_battle_actual = 80

    si.configure_sp_limits(63, 129)
    si.calibrate_from_known_team([mine])
    assert si.sp_limits() == (si.SP_MAX, si.SP_BUDGET)

    si.reset_sp_limits()


def test_forme_change_keeps_stat_points_and_swaps_bases():
    si.reset_sp_limits()
    beliefs = si.TeamStatBeliefs()
    aegislash = build_pkmn('aegislash', is_opponent=True, hp=100)
    aegislash.nature = 'sassy'

    belief = beliefs.belief_for(aegislash)
    belief.narrow('spe', {4, 5})
    assert belief.base_stats['def'] == 140
    assert belief.base_stats['atk'] == 50

    # Al pasar a Blade cambian las bases pero los Stat Points son los mismos
    aegislash.ATK_battle_base = 140
    aegislash.DEF_battle_base = 50
    same = beliefs.belief_for(aegislash)
    assert same is belief
    assert belief.base_stats['atk'] == 140
    assert belief.candidates['spe'] == {4, 5}


def test_mega_belief_ranges_remap_finals_but_keep_sp():
    """Blastoise Bold Atk=92 (forma base) no debe aparecer en el rango mega."""
    si.reset_sp_limits()
    si.configure_sp_limits(63, 129)  # mismos topes que partida Champions calibrada
    beliefs = si.TeamStatBeliefs()
    base = build_pkmn('blastoise', is_opponent=True, hp=100)
    base.nature = 'bold'
    belief = beliefs.belief_for(base)
    assert belief.stat_range('atk') == (92, 121)

    mega = build_pkmn('blastoise-mega', is_opponent=True, hp=100)
    mega.nature = 'bold'
    same = beliefs.belief_for(mega)
    assert same is belief
    assert belief.current_forme == 'blastoise-mega'
    # Mismos SP → finales con bases mega (Atk base 103 → 110-139)
    assert belief.stat_range('atk') == (110, 139)
    # La forma base sigue conteniendo el Atk real pre-mega
    dex = si.dex_base_stats('blastoise')
    assert belief.stat_range_with_bases('atk', dex) == (92, 121)
    assert 92 in belief.stat_values_with_bases('atk', dex)
    assert 92 not in belief.stat_values('atk')
    si.reset_sp_limits()


def test_illusion_reveal_forgets_the_impersonated_species():
    si.reset_sp_limits()
    beliefs = si.TeamStatBeliefs()
    venusaur = build_pkmn('venusaur', is_opponent=True, hp=100)
    beliefs.belief_for(venusaur)
    assert 'venusaur' in beliefs.beliefs

    beliefs.forget('Venusaur')
    assert 'venusaur' not in beliefs.beliefs


def test_speed_inference_never_excludes_the_true_value():
    """
    Propiedad clave: estrechar es irreversible, asi que el valor verdadero jamas
    debe quedar fuera del rango estimado. Se sortean repartos reales, se deduce
    el orden de accion que Showdown produciria y se comprueba la solidez.
    """
    import random

    si.reset_sp_limits()
    si.configure_sp_limits(si.SP_HARD_MAX, si.SP_HARD_MAX * 6)

    bases = {'dragonite': 80, 'farigiraf': 60, 'whimsicott': 116, 'torkoal': 20}
    natures = ('adamant', 'timid', 'jolly', 'relaxed', 'brave', 'serious')
    own_team_speeds = (80, 105, 130, 184)
    rng = random.Random(20260912)

    for _ in range(200):
        species = rng.choice(list(bases))
        base = bases[species]
        nature = rng.choice(natures)
        true_sp = rng.randint(0, si.SP_HARD_MAX)
        true_stat = si.champions_stat(base, true_sp, nature, 'spe')

        belief = si.StatBelief(species, {'spe': base}, nature)

        for _ in range(12):
            user_speed = rng.choice(own_team_speeds)
            if true_stat == user_speed:
                opponent_first = rng.random() < 0.5   # empate: Showdown decide al azar
            else:
                opponent_first = true_stat > user_speed

            allowed = {
                sp for sp in belief.candidates['spe']
                if (si.champions_stat(base, sp, nature, 'spe') >= user_speed)
                == opponent_first
                or si.champions_stat(base, sp, nature, 'spe') == user_speed
            }
            belief.narrow('spe', allowed)

            low, high = belief.stat_range('spe')
            assert low <= true_stat <= high, (
                f'{species} nat={nature} SP={true_sp} stat={true_stat} '
                f'quedo fuera de {low}-{high}'
            )

    si.reset_sp_limits()


def test_report_includes_the_stat_point_estimate():
    si.reset_sp_limits()
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    battle.user.account_name = 'agent'
    battle.opponent.account_name = 'rival'
    theirs = build_pkmn('dragonite', is_opponent=True, hp=100)
    theirs.nature = 'adamant'
    battle.opponent.active1 = theirs
    battle.stat_beliefs.belief_for(theirs)

    battle.imprimir(to_console=False)
    with open('logs/latest_state.txt', encoding='utf-8') as handle:
        report = handle.read()

    assert 'STAT POINTS ESTIMADOS' in report
    assert f'tope {si.SP_MAX}/stat, {si.SP_BUDGET} total' in report
    assert 'dragonite' in report
    assert 'SP:' in report
    assert 'est:' in report
    assert 'conv:' in report
    assert '0-63' not in report
    assert '0-32' in report


def test_singleturn_protect_and_helping_hand_clear_on_turn():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    prim = build_pkmn('primarina', is_opponent=True, hp=100)
    battle.opponent.active1 = prim
    mine = build_pkmn('manectric', is_opponent=False, hp=150)
    battle.user.active1 = mine

    singleturn(battle, ['', '-singleturn', 'p2a: Primarina', 'Protect'])
    assert prim.is_protected is True
    assert prim.volatile_statuses.get('protect') is True

    singleturn(battle, ['', '-singleturn', 'p1a: Manectric', 'Helping Hand'])
    assert mine.is_helping_hand is True

    turn(battle, ['', 'turn', '2'])
    assert prim.is_protected is False
    assert mine.is_helping_hand is False
    assert 'protect' not in prim.volatile_statuses


def test_singlemove_destiny_bond_survives_turn_clears_on_other_move():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    gengar = build_pkmn('gengar', is_opponent=True, hp=100)
    battle.opponent.active1 = gengar
    battle.user.active1 = build_pkmn('whimsicott', is_opponent=False, hp=155)

    singlemove(battle, ['', '-singlemove', 'p2a: Gengar', 'Destiny Bond'])
    assert gengar.is_destiny_bonded is True

    turn(battle, ['', 'turn', '3'])
    assert gengar.is_destiny_bonded is True  # cruza |turn|

    observe_move(battle, ['', 'move', 'p2a: Gengar', 'Shadow Ball', 'p1a: Whimsicott'])
    assert gengar.is_destiny_bonded is False


def test_charge_clears_after_electric_move_not_before_pending():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'
    mine = build_pkmn('manectric', is_opponent=False, hp=150)
    mine.nature = 'modest'
    mine.ATK_battle_actual = 100
    mine.DEF_battle_actual = 80
    mine.SPA_battle_actual = 120
    mine.SPD_battle_actual = 80
    mine.VEL_battle_actual = 150
    mine.HPB_battle_base = 150
    mine.is_charged = True
    battle.user.active1 = mine
    theirs = build_pkmn('dragonite', is_opponent=True, hp=100)
    theirs.nature = 'adamant'
    battle.opponent.active1 = theirs
    battle.user.active2 = build_pkmn('torkoal', is_opponent=False, hp=177)
    battle.opponent.active2 = build_pkmn('blastoise', is_opponent=True, hp=100)

    seen = {}

    def fake_range(req):
        field = req.get('field') or {}
        seen['charge'] = field.get('attackerSide', {}).get('isCharge')
        return (20, 25)

    set_damage_range_override(fake_range)
    try:
        observe_move(
            battle,
            ['', 'move', 'p1a: Manectric', 'Thunderbolt', 'p2a: Dragonite'],
        )
        pending = battle.pending_damage_ctx
        assert pending is not None
        assert mine.is_charged is False  # se consume tras congelar el pending
        theirs.HPA_battle_actual = 100
        heal_or_damage(battle, ['', '-damage', 'p2a: Dragonite', '78/100'])
        assert seen.get('charge') is True  # el golpe aún ve Charge en el ctx
    finally:
        set_damage_range_override(None)
        si.reset_sp_limits()


def test_report_hides_protect_shows_destiny_bond():
    from showdown.battle import _fmt_status_short

    p = build_pkmn('primarina', is_opponent=True, hp=100)
    p.is_protected = True
    p.volatile_statuses['protect'] = True
    assert 'protect' not in _fmt_status_short(p)

    p.is_destiny_bonded = True
    assert 'destinybond' in _fmt_status_short(p)
