import sys
sys.path.insert(0, '.')

from showdown.battle import Battle, Pokemon
from showdown.battle_modifier import replace_illusion


def build_pkmn(name):
    return Pokemon(name, 100, True, '', '', None, None, None, None, None)


def test_replace_illusion_treats_hisui_as_same_real_species():
    battle = Battle()
    battle.user.id = 'p1'

    battle.opponent.active1 = build_pkmn('dragonite')
    battle.opponent.active2 = build_pkmn('starmie')
    battle.opponent.reserved = [build_pkmn('starmie')]
    battle.opponent.opponents = [
        build_pkmn('zoroark-hisui'),
        build_pkmn('dragonite'),
        build_pkmn('starmie'),
    ]

    replace_illusion(battle, ['', '', 'p2a: Dragonite', 'Zoroark-Hisui, L50, M'])

    assert battle.opponent.active1.name in {'zoroark', 'zoroark-hisui'}
    assert battle.opponent.active2.name == 'dragonite'
    assert all(p.name != 'dragonite' for p in battle.opponent.reserved)
