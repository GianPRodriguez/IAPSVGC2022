import sys
sys.path.insert(0, '.')

from showdown.battle import Battle, Pokemon
from showdown.battle_modifier import switch_or_drag



class TestBattle(Battle):
    async def action_team_preview(self, battle):
        pass

    async def action_battle(self, battle):
        pass

def build_pkmn(name):
    return Pokemon(name, 100, True, '', '', None, None, None, None, None)


def test_switch_keeps_starmie_in_reserve_when_disguised_dragons_switch_in():
    battle = TestBattle()
    battle.user.id = 'p1'
    battle.opponent.id = 'p2'

    battle.opponent.active1 = build_pkmn('dragonite')
    battle.opponent.active2 = build_pkmn('starmie')
    battle.opponent.reserved = []
    battle.opponent.opponents = [
        build_pkmn('zoroark'),
        build_pkmn('dragonite'),
        build_pkmn('starmie'),
    ]

    # Simula: Zoroark se disfraza como Dragonite y Starmie ya está activo
    switch_or_drag(battle, ['', 'switch', 'p2a: Dragonite', 'Dragonite, L50, F', '100/100'])
    switch_or_drag(battle, ['', 'switch', 'p2b: Starmie', 'Starmie, L50', '100/100'])
    switch_or_drag(battle, ['', 'switch', 'p2b: Dragonite', 'Dragonite, L50, F', '100/100'])

    assert battle.opponent.active1.name == 'dragonite'
    assert battle.opponent.active2.name == 'dragonite'
    assert any(p.name == 'starmie' for p in battle.opponent.reserved), [p.name for p in battle.opponent.reserved]
