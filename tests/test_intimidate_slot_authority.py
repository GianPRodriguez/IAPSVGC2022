"""Intimidate / |-unboost|: autoridad del slot (anti pe atk-1 | iap=-)."""
from __future__ import annotations

import json

from showdown.battle import Battle, Pokemon
from showdown.battle_modifier import (
    _resolve_active_from_ident,
    request,
    set_opponent_ability_from_ability_tag,
    switch_or_drag,
    unboost,
)
from showdown.inference_sim.battle_state_check import _boosts_map


class _B(Battle):
    async def action_team_preview(self, battle):
        return None

    async def action_battle(self, battle):
        return None


def _mon(name: str, *, is_opponent: bool = False, hp: int = 200) -> Pokemon:
    p = Pokemon(name, hp, is_opponent, '', '', None, None, None, None, None)
    p.HPA_battle_actual = hp
    return p


def test_unboost_cross_slot_same_side_by_name():
    """a/b cruzados: |-unboost|p1a: Incineroar debe ir al Incineroar en b."""
    b = _B()
    b.user.id = 'p1'
    b.opponent.id = 'p2'
    b.user.active1 = _mon('fluttermane')
    b.user.active2 = _mon('incineroar')
    unboost(b, ['', '-unboost', 'p1a: Incineroar', 'atk', '1'])
    assert _boosts_map(b.user.active1) == {}
    assert _boosts_map(b.user.active2) == {'atk': -1}


def test_unboost_mirror_match_does_not_hit_foe_clone():
    """Ambos bandos con la misma especie: el slot p1a no debe ir al rival."""
    b = _B()
    b.user.id = 'p1'
    b.opponent.id = 'p2'
    b.user.active1 = _mon('incineroar')
    b.user.active2 = _mon('rillaboom')
    b.opponent.active1 = _mon('incineroar', is_opponent=True)
    b.opponent.active2 = _mon('rillaboom', is_opponent=True)
    unboost(b, ['', '-unboost', 'p1a: Incineroar', 'atk', '1'])
    unboost(b, ['', '-unboost', 'p1b: Rillaboom', 'atk', '1'])
    assert _boosts_map(b.user.active1) == {'atk': -1}
    assert _boosts_map(b.user.active2) == {'atk': -1}
    assert _boosts_map(b.opponent.active1) == {}
    assert _boosts_map(b.opponent.active2) == {}


def test_resolve_same_side_not_foe_when_slot_empty_species():
    """Sin mon propio de esa especie, no saltar al clon rival."""
    b = _B()
    b.user.id = 'p1'
    b.opponent.id = 'p2'
    b.user.active1 = _mon('sylveon')
    b.opponent.active1 = _mon('incineroar', is_opponent=True)
    got = _resolve_active_from_ident(b, 'p1a: Incineroar')
    # Slot a = sylveon (fallback); no el Incineroar rival.
    assert got is b.user.active1
    assert got is not b.opponent.active1


def test_intimidate_both_leads_survive_request():
    b = _B()
    b.user.id = 'p1'
    b.opponent.id = 'p2'
    b.user.reserved = [
        _mon('incineroar'),
        _mon('rillaboom'),
        _mon('fluttermane'),
        _mon('chienpao'),
    ]
    b.user.active1 = None
    b.user.active2 = None
    b.opponent.opponents = [
        _mon('landorus-therian', is_opponent=True, hp=100),
        _mon('amoonguss', is_opponent=True, hp=100),
    ]
    switch_or_drag(b, ['', 'switch', 'p1a: Incineroar', 'Incineroar, L50, M', '200/200'])
    switch_or_drag(b, ['', 'switch', 'p1b: Rillaboom', 'Rillaboom, L50, M', '200/200'])
    switch_or_drag(b, ['', 'switch', 'p2a: Landorus', 'Landorus-Therian, L50, M', '100/100'])
    set_opponent_ability_from_ability_tag(b, ['', '-ability', 'p2a: Landorus', 'Intimidate'])
    unboost(b, ['', '-unboost', 'p1a: Incineroar', 'atk', '1'])
    unboost(b, ['', '-unboost', 'p1b: Rillaboom', 'atk', '1'])
    assert _boosts_map(b.user.active1) == {'atk': -1}
    assert _boosts_map(b.user.active2) == {'atk': -1}

    payload = {
        'rqid': 1,
        'active': [{}, {}],
        'side': {
            'id': 'p1',
            'name': 'agent',
            'pokemon': [
                {
                    'ident': 'p1: Incineroar',
                    'details': 'Incineroar, L50, M',
                    'condition': '200/200',
                    'active': True,
                    'stats': {'atk': 100, 'def': 100, 'spa': 100, 'spd': 100, 'spe': 100},
                    'moves': ['flareblitz'],
                    'baseAbility': 'intimidate',
                    'item': '',
                    'pokeball': 'pokeball',
                    'ability': 'intimidate',
                },
                {
                    'ident': 'p1: Rillaboom',
                    'details': 'Rillaboom, L50, M',
                    'condition': '200/200',
                    'active': True,
                    'stats': {'atk': 100, 'def': 100, 'spa': 100, 'spd': 100, 'spe': 100},
                    'moves': ['grassyglide'],
                    'baseAbility': 'grassysurge',
                    'item': '',
                    'pokeball': 'pokeball',
                    'ability': 'grassysurge',
                },
                {
                    'ident': 'p1: Flutter Mane',
                    'details': 'Flutter Mane, L50',
                    'condition': '200/200',
                    'active': False,
                    'stats': {'atk': 100, 'def': 100, 'spa': 100, 'spd': 100, 'spe': 100},
                    'moves': ['moonblast'],
                    'baseAbility': 'protosynthesis',
                    'item': '',
                    'pokeball': 'pokeball',
                    'ability': 'protosynthesis',
                },
                {
                    'ident': 'p1: Chien-Pao',
                    'details': 'Chien-Pao, L50',
                    'condition': '200/200',
                    'active': False,
                    'stats': {'atk': 100, 'def': 100, 'spa': 100, 'spd': 100, 'spe': 100},
                    'moves': ['iciclecrash'],
                    'baseAbility': 'swordofruin',
                    'item': '',
                    'pokeball': 'pokeball',
                    'ability': 'swordofruin',
                },
            ],
        },
    }
    request(b, ['', 'request', json.dumps(payload)])
    assert _boosts_map(b.user.active1) == {'atk': -1}
    assert _boosts_map(b.user.active2) == {'atk': -1}
