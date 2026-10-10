"""greedy_joint_snapshot — portable joint damage greedy on BattleSnapshot."""
from __future__ import annotations

from typing import Optional

import numpy as np

from showdown.rl.action_space import decode_joint_action, decode_slot_action
from showdown.rl.damage_heuristic import estimate_damage_pct
from showdown.rl.mcts_puct.simulator import legal_joints
from showdown.rl.snapshot import BattleSnapshot, MonSnapshot


def _slot_damage(
    snap: BattleSnapshot,
    slot: int,
    action_id: int,
) -> float:
    act = decode_slot_action(int(action_id))
    if act.kind not in ('move', 'mega', 'tera'):
        return 0.0
    atk: Optional[MonSnapshot] = (
        snap.our.active[slot] if slot < len(snap.our.active) else None
    )
    if atk is None or atk.fainted:
        return 0.0
    mi = int(act.move_index)
    move_name = atk.moves[mi] if mi < len(atk.moves) else ''
    ti = int(min(max(act.target, 0), 1))
    dfn = snap.foe.active[ti] if ti < len(snap.foe.active) else None
    if dfn is None or dfn.fainted:
        return 0.0
    lo, hi = estimate_damage_pct(
        move_name,
        attacker_types=atk.types,
        defender_types=dfn.types,
        weather=str(getattr(snap.field, 'weather', '') or ''),
    )
    return 0.5 * (float(lo) + float(hi))


def score_joint(snap: BattleSnapshot, joint: int) -> float:
    """Heuristic: expected damage to foe from both slots (+ faint bias)."""
    a1, a2 = decode_joint_action(int(joint))
    dmg = _slot_damage(snap, 0, a1) + _slot_damage(snap, 1, a2)
    # Prefer actions that threaten remaining HP
    foe_hp = snap.foe.team_hp_pct_total()
    return float(dmg) + 0.01 * max(0.0, 400.0 - foe_hp)


def greedy_joint_snapshot(snap: BattleSnapshot) -> int:
    """
    Argmax joint damage heuristic over legal joints.

    No poke-env / network. Used for MCTS rollouts (both sides).
    """
    legal = legal_joints(snap)
    if legal.size == 0:
        from showdown.rl.action_space import encode_joint_action
        from showdown.rl.constants import PASS_ID
        return encode_joint_action(PASS_ID, PASS_ID)
    best_j = int(legal[0])
    best_s = score_joint(snap, best_j)
    for j in legal[1:]:
        s = score_joint(snap, int(j))
        if s > best_s:
            best_s = s
            best_j = int(j)
    return best_j
