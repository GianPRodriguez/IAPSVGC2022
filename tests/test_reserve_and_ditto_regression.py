import sys
sys.path.insert(0, '.')

from showdown.battle import Battle, Pokemon
from showdown.battle_modifier import (
    _resolve_duplicate_illusion_reserves,
    _revert_special_pokemon,
    activate,
    curestatus,
    end_volatile_status,
    fieldend,
    fieldstart,
    form_change,
    heal_or_damage,
    replace_illusion,
    sidestart,
    sideend,
    start_volatile_status,
    status,
    switch_or_drag,
    turn,
    transform,
    weather,
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
