import pytest
import showdown.battle as battle_module
from showdown.battle import Battle, Pokemon
from showdown.battle_modifier import replace_illusion


@pytest.fixture(autouse=True)
def _stub_pkmn_info(monkeypatch):
    # Antes se asignaba a nivel de modulo y contaminaba al resto de la suite.
    monkeypatch.setattr(
        battle_module, 'getPkmnInfo',
        lambda name: (None, None, 100, 80, 80, 80, 80, 80),
    )



class TestBattle(Battle):
    async def action_team_preview(self, battle):
        pass

    async def action_battle(self, battle):
        pass

def build_pkmn(name):
    return Pokemon(name, 100, True, '', '', None, None, None, None, None)


def test_replace_illusion_does_not_duplicate_known_real_species_in_reserve():
    battle = TestBattle()
    battle.user.id = 'p1'

    battle.opponent.active1 = build_pkmn('dragonite')
    battle.opponent.active2 = build_pkmn('dragonite')
    battle.opponent.reserved = [build_pkmn('starmie')]
    battle.opponent.opponents = [
        build_pkmn('zoroark'),
        build_pkmn('dragonite'),
        build_pkmn('starmie'),
    ]

    replace_illusion(battle, ['', '', 'p2a: Dragonite', 'Zoroark, L50, M'])

    assert battle.opponent.active1.name == 'zoroark'
    assert battle.opponent.active2.name == 'dragonite'
    assert all(p.name != 'dragonite' for p in battle.opponent.reserved)
