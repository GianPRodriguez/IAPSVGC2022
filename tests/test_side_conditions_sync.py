"""Side conditions sync: protocol sidestart + PE harness mirror (anti-desync)."""
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

from showdown.battle import Battle, Pokemon
from showdown.battle_modifier import sidestart, sideend
from showdown.inference_sim.battle_state_check import (
    contrast_battle,
    mirror_pe_field_and_boosts_into_iap,
)
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
    b.user.side_conditions = defaultdict(lambda: 0)
    b.opponent.side_conditions = defaultdict(lambda: 0)
    return b


def _mon(name: str) -> Pokemon:
    p = Pokemon(name, 100, False, '', '', None, None, None, None, None)
    p.HPA_battle_actual = 100
    return p


def test_sidestart_toxicspikes_stacks_and_sideend():
    b = _battle()
    sidestart(b, ['', '-sidestart', 'p2: Opp', 'move: Toxic Spikes'])
    assert b.opponent.side_conditions['toxicspikes'] == 1
    sidestart(b, ['', '-sidestart', 'p2: Opp', 'move: Toxic Spikes'])
    assert b.opponent.side_conditions['toxicspikes'] == 2
    sidestart(b, ['', '-sidestart', 'p2: Opp', 'move: Toxic Spikes'])
    assert b.opponent.side_conditions['toxicspikes'] == 2  # cap
    sideend(b, ['', '-sideend', 'p2: Opp', 'move: Toxic Spikes'])
    assert 'toxicspikes' not in b.opponent.side_conditions


def test_sidestart_toxicspikes_without_move_prefix():
    b = _battle()
    assert feed_line(b, '|-sidestart|p1: Us|Toxic Spikes') == '-sidestart'
    assert b.user.side_conditions['toxicspikes'] == 1


def test_sidestart_stealthrock_and_stickyweb():
    b = _battle()
    sidestart(b, ['', '-sidestart', 'p2: Opp', 'move: Stealth Rock'])
    sidestart(b, ['', '-sidestart', 'p2: Opp', 'move: Sticky Web'])
    assert b.opponent.side_conditions['stealthrock'] == 1
    assert b.opponent.side_conditions['stickyweb'] == 1


def test_mirror_fills_missing_tailwind_and_toxicspikes():
    """Regresión fixplan: pe=['toxicspikes'|'tailwind'] | iap=-"""
    iap = _battle()
    iap.user.active1 = _mon('incineroar')
    iap.user.active2 = _mon('rillaboom')
    iap.opponent.active1 = _mon('landorus')
    iap.opponent.active2 = _mon('amoonguss')

    # Fake PE battle with SC present and matching actives
    pe = SimpleNamespace(
        weather={},
        fields={},
        side_conditions={'TAILWIND': 3},
        opponent_side_conditions={'TOXIC_SPIKES': 1},
        active_pokemon=[SimpleNamespace(species='Incineroar', fainted=False, boosts={}, status=None, current_hp_fraction=1.0),
                        SimpleNamespace(species='Rillaboom', fainted=False, boosts={}, status=None, current_hp_fraction=1.0)],
        opponent_active_pokemon=[SimpleNamespace(species='Landorus', fainted=False, boosts={}, status=None, current_hp_fraction=1.0),
                                 SimpleNamespace(species='Amoonguss', fainted=False, boosts={}, status=None, current_hp_fraction=1.0)],
        turn=4,
    )

    # Before mirror: mismatch
    mm_before = contrast_battle(iap, pe, battle_tag='t')
    fields_before = {m.field for m in mm_before}
    assert 'our_side_conditions' in fields_before or 'opp_side_conditions' in fields_before

    mirror_pe_field_and_boosts_into_iap(iap, pe)
    assert int(iap.user.side_conditions.get('tailwind', 0) or 0) > 0
    assert int(iap.opponent.side_conditions.get('toxicspikes', 0) or 0) > 0

    mm_after = contrast_battle(iap, pe, battle_tag='t')
    sc_fields = {m.field for m in mm_after if 'side_conditions' in m.field}
    assert not sc_fields, sc_fields


def test_mirror_fills_missing_trickroom_and_opp_tailwind():
    """Regresión fixplan: pe trickroom + opp tailwind | iap=-"""
    from poke_env.battle.field import Field
    from poke_env.battle.side_condition import SideCondition

    iap = _battle()
    iap.turn = 17
    pe = SimpleNamespace(
        weather={},
        fields={Field.TRICK_ROOM: 10},
        side_conditions={},
        opponent_side_conditions={SideCondition.TAILWIND: 12},
        active_pokemon=[None, None],
        opponent_active_pokemon=[None, None],
        turn=17,
    )
    mirror_pe_field_and_boosts_into_iap(iap, pe)
    mm = contrast_battle(iap, pe, battle_tag='t')
    assert not any(m.field == 'fields' for m in mm)
    assert not any(m.field == 'opp_side_conditions' for m in mm)
    assert iap.terrain.get('trickroom', 0) > 0
    assert iap.opponent.side_conditions.get('tailwind', 0) > 0


def test_mirror_fills_missing_grassyterrain():
    """Regresión fixplan: pe=['grassyterrain'] | iap=-"""
    from poke_env.battle.field import Field

    iap = _battle()
    iap.turn = 7
    pe = SimpleNamespace(
        weather={},
        fields={Field.GRASSY_TERRAIN: 2},
        side_conditions={},
        opponent_side_conditions={},
        active_pokemon=[None, None],
        opponent_active_pokemon=[None, None],
        turn=7,
    )
    assert contrast_battle(iap, pe, battle_tag='t')
    mirror_pe_field_and_boosts_into_iap(iap, pe)
    mm = contrast_battle(iap, pe, battle_tag='t')
    assert not any(m.field == 'fields' for m in mm)
    assert iap.terrain.get('grassyterrain', 0) > 0


def test_mirror_clears_iap_side_condition_absent_in_pe():
    iap = _battle()
    iap.user.side_conditions['tailwind'] = 4
    pe = SimpleNamespace(
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
        active_pokemon=[None, None],
        opponent_active_pokemon=[None, None],
        turn=2,
    )
    mirror_pe_field_and_boosts_into_iap(iap, pe)
    assert 'tailwind' not in iap.user.side_conditions
