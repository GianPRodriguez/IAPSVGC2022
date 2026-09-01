import sys
sys.path.insert(0, '.')

import showdown.battle as battle_module
from showdown.battle import Battle, Pokemon
from showdown.battle_modifier import replace_illusion

battle_module.getPkmnInfo = lambda name: (None, None, 100, 80, 80, 80, 80, 80)

def build_pkmn(name):
    return Pokemon(name, 100, True, '', '', None, None, None, None, None)

battle = Battle()
battle.user.id = 'p1'
battle.opponent.active1 = build_pkmn('dragonite')
battle.opponent.active2 = build_pkmn('starmie')
battle.opponent.reserved = [build_pkmn('starmie')]
battle.opponent.opponents = [
    build_pkmn('zoroark'),
    build_pkmn('dragonite'),
    build_pkmn('starmie'),
]

replace_illusion(battle, ['', '', 'p2a: Dragonite', 'Zoroark, L50, M'])

assert battle.opponent.active1.name == 'zoroark', battle.opponent.active1.name
assert battle.opponent.active2.name == 'dragonite', battle.opponent.active2.name
assert all(p.name != 'dragonite' for p in battle.opponent.reserved), [p.name for p in battle.opponent.reserved]
print('PASS')
