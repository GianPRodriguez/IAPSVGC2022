"""Spec C2 IAP reward → UI scale (terminal + material)."""
from __future__ import annotations

from types import SimpleNamespace

from showdown.inference_sim.reward_telemetry import (
    finalize_iap_game_reward,
    material_from_iap_side,
    observe_iap_spec_reward,
    push_iap_reward_to_ui,
)
class _Mon:
    def __init__(self, hpa, hpb, fainted=False):
        self.HPA_battle_actual = hpa
        self.HPB_battle_base = hpb
        self.is_fainted = fainted


def test_material_reads_hp_and_faint():
    user = SimpleNamespace(
        isUser=True,
        active1=_Mon(50, 100),
        active2=_Mon(100, 100),
        reserved=[_Mon(0, 100, fainted=True), _Mon(100, 100)],
    )
    mat = material_from_iap_side(user, party_size=4)
    assert mat.faint_count == 1
    assert mat.hp_pct_total == 50.0 + 100.0 + 100.0  # fainted contributes 0 HP


def test_observe_and_terminal_win_updates_ui_scale():
    our_a = _Mon(100, 100)
    our_b = _Mon(100, 100)
    foe_a = _Mon(100, 100)
    foe_b = _Mon(100, 100)
    battle = SimpleNamespace(
        user=SimpleNamespace(
            isUser=True, account_name='Bot',
            active1=our_a, active2=our_b, reserved=[],
        ),
        opponent=SimpleNamespace(
            isUser=False, account_name='Human',
            active1=foe_a, active2=foe_b, opponents=[],
        ),
        finished=False,
        won=None,
        last_c2_reward=0.0,
        spec_c2_return=0.0,
        last_c1_reward=0.0,
        spec_c1_return=0.0,
        _spec_prev_our=None,
        _spec_prev_foe=None,
        lastReward=0.0,
        reward=0.0,
    )

    def calculateRewards():
        battle.lastReward = float(battle.last_c2_reward) * 100.0
        battle.reward = float(battle.spec_c2_return) * 100.0
        return battle.lastReward, battle.reward

    battle.calculateRewards = calculateRewards

    # Baseline
    d0, ep0 = observe_iap_spec_reward(battle)
    assert d0 == 0.0 and ep0 == 0.0

    # Chip damage on foe
    foe_a.HPA_battle_actual = 50
    d1, ep1 = observe_iap_spec_reward(battle)
    assert d1 > 0.0
    assert ep1 == d1

    class _Q:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    q = _Q()
    finalize_iap_game_reward(battle, 'Bot', q)
    assert battle.spec_c2_return > ep1  # +win
    assert any(item[0] == 2 for item in q.items)
    # UI scale: win contributes ~100
    _delta, total = battle.calculateRewards()
    assert total >= 100.0 * 0.9  # win 1.0 * 100 (+ prior chip)


def test_push_ui_payload_shape():
    battle = SimpleNamespace(
        last_c2_reward=0.15,
        spec_c2_return=0.3,
        lastReward=0.0,
        reward=0.0,
    )

    def calculateRewards():
        battle.lastReward = battle.last_c2_reward * 100.0
        battle.reward = battle.spec_c2_return * 100.0
        return battle.lastReward, battle.reward

    battle.calculateRewards = calculateRewards

    class _Q:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    q = _Q()
    push_iap_reward_to_ui(battle, q, label='Fin juego')
    assert len(q.items) == 1
    assert q.items[0][0] == 2
    assert q.items[0][1] == 'Fin juego'
    assert q.items[0][9] == 15.0
    assert q.items[0][11] == 30.0
