"""Offline unit tests for poke-env ↔ Spec adapter (no PS required)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from showdown.rl.action_space import PASS_ID, SLOT_ACTION_N, encode_slot_action
from showdown.rl.constants import OBS_DIM
from showdown.rl.observation import build_observation
from showdown.rl.pe_adapter import (
    pe_autoregressive_masks,
    pe_slot_mask,
    snapshot_from_battle,
)
from showdown.rl.snapshot import SlotAction


def _fake_mon(species: str, *, hp_frac: float = 1.0, fainted: bool = False):
    mv = SimpleNamespace(id='tackle', name='Tackle', current_pp=32, max_pp=32)
    return SimpleNamespace(
        species=species,
        name=species,
        base_species=species,
        types=('Normal',),
        current_hp_fraction=hp_frac,
        current_hp=int(100 * hp_frac),
        max_hp=100,
        fainted=fainted,
        boosts={'atk': 0},
        status=None,
        effects={},
        moves={'tackle': mv},
        ability='Intimidate',
        item='safetygoggles',
        tera_type=None,
        first_turn=False,
        active=True,
        is_dynamaxed=False,
    )


def _fake_battle():
    a0 = _fake_mon('Incineroar')
    a1 = _fake_mon('Rillaboom')
    b0 = _fake_mon('Landorus')
    b1 = _fake_mon('Amoonguss', hp_frac=0.5)
    battle = SimpleNamespace(
        turn=3,
        team={'p1a': a0, 'p1b': a1, 'p1c': _fake_mon('Flutter Mane')},
        opponent_team={'p2a': b0, 'p2b': b1},
        active_pokemon=[a0, a1],
        opponent_active_pokemon=[b0, b1],
        can_mega_evolve=[False, False],
        can_tera=[False, True],
        finished=False,
        won=None,
        force_switch=[False, False],
        player_role='p1',
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
        available_moves=[
            [SimpleNamespace(id='flareblitz')],
            [SimpleNamespace(id='grassyglide')],
        ],
        available_switches=[[], [_fake_mon('Flutter Mane')]],
        valid_orders=None,
    )
    return battle


def test_snapshot_from_battle_shapes():
    snap = snapshot_from_battle(_fake_battle())
    assert snap.turn == 3
    assert snap.our.active[0] is not None
    assert snap.our.active[0].species == 'Incineroar'
    assert abs(snap.foe.active[1].hp_pct - 50.0) < 1e-6
    obs = build_observation(snap)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_pe_masks_pass_and_moves():
    battle = _fake_battle()
    m0 = pe_slot_mask(battle, 0)
    assert m0.shape == (SLOT_ACTION_N,)
    assert m0[PASS_ID] == 1.0
    assert m0[0:4].sum() == 4.0
    _m1, m2 = pe_autoregressive_masks(battle)
    assert m2[16] == 1.0


def test_joint_order_fallback_without_join(monkeypatch):
    from poke_env.player.player import Player
    from showdown.rl import pe_adapter

    battle = _fake_battle()
    called = {'n': 0}

    def _rand(_b):
        called['n'] += 1
        return MagicMock(name='rand_order')

    monkeypatch.setattr(Player, 'choose_random_doubles_move', staticmethod(_rand))
    monkeypatch.setattr(
        'poke_env.player.battle_order.DoubleBattleOrder.join_orders',
        staticmethod(lambda *_a, **_k: []),
    )

    _order, fb = pe_adapter.joint_order_from_spec_actions(
        battle,
        (encode_slot_action(kind='move', move_index=0, target=0), PASS_ID),
    )
    assert fb is True
    assert called['n'] == 1


def test_slot_action_to_pe_pass():
    from poke_env.player.battle_order import PassBattleOrder
    from showdown.rl.pe_adapter import slot_action_to_pe_order

    battle = _fake_battle()
    order = slot_action_to_pe_order(battle, 0, SlotAction(kind='pass', raw_id=PASS_ID))
    assert isinstance(order, PassBattleOrder)
