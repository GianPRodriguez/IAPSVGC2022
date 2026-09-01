import sys
sys.path.insert(0, '.')

from showdown.battle import Battle, Pokemon
from showdown.battle_modifier import switch_or_drag, transform


def build_pkmn(name, is_opponent=True, hp=100):
    p = Pokemon(name, hp, is_opponent, '', '', None, None, None, None, None)
    p.HPA_battle_actual = hp
    return p


def test_starmie_remains_in_reserve_after_uncertain_zoroark_switch_sequence():
    battle = Battle()
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
    battle = Battle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    battle.user.active1 = build_pkmn('ditto', is_opponent=False)
    battle.opponent.active1 = build_pkmn('gengar', is_opponent=True)

    transform(battle, ['', '-transform', 'p1a: Ditto', 'p2a: Gengar', '[from] ability: Imposter'])

    transformed = battle.user.active1
    assert transformed.name == 'gengar'
    assert transformed.ATK_battle_base == battle.opponent.active1.ATK_battle_base
    assert transformed.ATK_battle_actual == -1


def test_opponent_ditto_transform_keeps_its_base_minus_one_and_copies_current_stats():
    battle = Battle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    battle.opponent.active1 = build_pkmn('ditto', is_opponent=True)
    battle.user.active1 = build_pkmn('starmie', is_opponent=False)

    transform(battle, ['', '-transform', 'p2a: Ditto', 'p1a: Starmie', '[from] ability: Imposter'])

    transformed = battle.opponent.active1
    assert transformed.name == 'starmie'
    assert transformed.ATK_battle_base == -1
    assert transformed.ATK_battle_actual == battle.user.active1.ATK_battle_actual
