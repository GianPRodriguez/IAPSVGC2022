"""Autoridad de clima/terreno (anti-péndulo sync).

- Clima: solo |-weather| (nunca -ability / OTS switch-in).
- Terreno: |-fieldstart| + fallback Surge en -ability/-activate (no OTS switch-in).
"""
from __future__ import annotations

from showdown.battle import Battle, Pokemon
from showdown.battle_modifier import (
    _apply_switch_in_field_abilities,
    _ensure_terrain_from_surge,
    _ensure_weather_from_ability,
    activate,
    fieldend,
    fieldstart,
    set_opponent_ability_from_ability_tag,
    weather,
)
from showdown.inference_sim.battle_state_check import _iap_fields, _iap_weather
from showdown.inference_sim.feeder import feed_line


class _B(Battle):
    async def action_team_preview(self, battle):
        return None

    async def action_battle(self, battle):
        return None


def _battle() -> _B:
    b = _B()
    b.user.id = 'p1'
    b.opponent.id = 'p2'
    return b


def _mon(name: str, *, ability: str = '') -> Pokemon:
    p = Pokemon(name, 100, True, '', ability, None, None, None, None, None)
    p.abilityPkmn = ability
    p.HPA_battle_actual = 100
    return p


def test_snow_warning_ability_alone_does_not_set_weather():
    b = _battle()
    b.opponent.active1 = _mon('abomasnow', ability='snowwarning')
    set_opponent_ability_from_ability_tag(
        b, ['', '-ability', 'p2a: Abomasnow', 'Snow Warning'],
    )
    assert _iap_weather(b) == ''
    _ensure_weather_from_ability(b, 'Snow Warning')
    assert _iap_weather(b) == ''


def test_weather_line_sets_snowscape():
    b = _battle()
    weather(
        b,
        ['', '-weather', 'Snowscape', '[from] ability: Snow Warning', '[of] p2a: Abomasnow'],
    )
    assert _iap_weather(b) == 'snowscape'


def test_weather_none_then_ability_stays_empty():
    b = _battle()
    weather(b, ['', '-weather', 'Snowscape'])
    weather(b, ['', '-weather', 'none'])
    set_opponent_ability_from_ability_tag(
        b, ['', '-ability', 'p2a: Abomasnow', 'Snow Warning'],
    )
    assert _iap_weather(b) == ''


def test_named_upkeep_can_reinstall_after_empty():
    b = _battle()
    weather(b, ['', '-weather', 'none'])
    weather(b, ['', '-weather', 'Snowscape', '[upkeep]'])
    assert _iap_weather(b) == 'snowscape'


def test_grassy_surge_ability_sets_terrain_without_fieldstart():
    """Fallback documentado: -ability Surge puede setear terreno si falta fieldstart."""
    b = _battle()
    b.opponent.active1 = _mon('rillaboom', ability='grassysurge')
    set_opponent_ability_from_ability_tag(
        b, ['', '-ability', 'p2a: Rillaboom', 'Grassy Surge'],
    )
    assert 'grassyterrain' in _iap_fields(b)


def test_activate_surge_sets_terrain():
    b = _battle()
    activate(b, ['', '-activate', 'p2a: Rillaboom', 'ability: Grassy Surge'])
    assert 'grassyterrain' in _iap_fields(b)


def test_switch_in_abilitypkmn_still_does_not_set_field():
    b = _battle()
    mon = _mon('ninetales', ability='snowwarning')
    mon.abilityPkmn = 'snowwarning'
    _apply_switch_in_field_abilities(b, mon)
    assert _iap_weather(b) == ''
    mon2 = _mon('tapukoko', ability='electricsurge')
    mon2.abilityPkmn = 'electricsurge'
    _apply_switch_in_field_abilities(b, mon2)
    assert _iap_fields(b) == set()


def test_feed_line_protocol_path():
    b = _battle()
    feed_line(b, '|-ability|p2a: Abomasnow|Snow Warning')
    assert _iap_weather(b) == ''
    feed_line(b, '|-weather|Snowscape|[from] ability: Snow Warning')
    assert _iap_weather(b) == 'snowscape'
    feed_line(b, '|-ability|p2b: Rillaboom|Grassy Surge')
    assert 'grassyterrain' in _iap_fields(b)
    feed_line(b, '|-fieldstart|move: Grassy Terrain')
    assert 'grassyterrain' in _iap_fields(b)


def test_fieldstart_and_fieldend():
    b = _battle()
    fieldstart(b, ['', '-fieldstart', 'move: Psychic Terrain'])
    assert 'psychicterrain' in _iap_fields(b)
    fieldend(b, ['', '-fieldend', 'move: Psychic Terrain'])
    assert _iap_fields(b) == set()
