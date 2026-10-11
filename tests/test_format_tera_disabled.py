"""Reg M-C: Terastallization must stay off (format kill-switch)."""
from __future__ import annotations

import numpy as np

from showdown.format_rules import TERASTALLIZATION_ENABLED, allow_tera
from showdown.rl.constants import TERA_ID
from showdown.rl.snapshot import BattleSnapshot, FieldSnapshot, MonSnapshot, SideSnapshot


def test_kill_switch_default_off():
    assert TERASTALLIZATION_ENABLED is False
    assert allow_tera(True) is False
    assert allow_tera(False) is False


def test_slot_mask_never_enables_tera():
    from showdown.rl.action_space import build_slot_mask

    mon = MonSnapshot(
        species='incineroar',
        can_mega=True,
        can_tera=True,  # even if snapshot lies, mask must refuse
        tera_type='Fire',
        moves=['fakout', 'flareblitz', 'partingshot', 'u-turn'],
        move_pps=[5, 5, 5, 5],
        is_active=True,
        revealed=True,
        hp_pct=100.0,
    )
    side = SideSnapshot(active=[mon, None], bench=[])
    snap = BattleSnapshot(
        turn=1,
        our=side,
        foe=SideSnapshot(active=[None, None], bench=[]),
        field=FieldSnapshot(),
    )
    mask = build_slot_mask(snap, 0)
    assert mask[TERA_ID] == 0.0
    assert mask.shape == (24,)


def test_helpers_format_ignores_mega_flag_2():
    from showdown.battle_bots.helpers import _helper_format_action

    s = _helper_format_action('move', 1, 2, 1)
    assert 'terastallize' not in s
