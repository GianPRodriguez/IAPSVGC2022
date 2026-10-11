"""ActionSpec — auto-regressive double slot + logit masking (Spec v1.2)."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from showdown.format_rules import allow_tera
from showdown.rl.constants import (
    ILLEGAL_LOGIT,
    JOINT_ACTION_N,
    MEGA_ID,
    N_MOVE_SLOTS,
    N_SWITCH_SLOTS,
    N_TARGETS,
    PASS_ID,
    RESERVED_ID,
    SLOT_ACTION_N,
    TERA_ID,
)
from showdown.rl.snapshot import BattleSnapshot, SlotAction

# Re-export for callers
__all_slot__ = SLOT_ACTION_N


def encode_joint_action(a1: int, a2: int) -> int:
    """a = 24*a1 + a2 ∈ {0..575}."""
    return int(a1) * SLOT_ACTION_N + int(a2)


def decode_joint_action(joint_id: int) -> Tuple[int, int]:
    j = int(joint_id)
    if j < 0 or j >= JOINT_ACTION_N:
        return PASS_ID, PASS_ID
    return j // SLOT_ACTION_N, j % SLOT_ACTION_N


def encode_slot_action(
    *,
    kind: str,
    move_index: int = 0,
    target: int = 0,
    switch_index: int = 0,
) -> int:
    kind = (kind or 'pass').lower()
    if kind == 'pass':
        return PASS_ID
    if kind == 'mega':
        return MEGA_ID
    if kind == 'tera':
        return TERA_ID
    if kind == 'switch':
        return 16 + int(max(0, min(N_SWITCH_SLOTS - 1, switch_index)))
    # move
    mi = int(max(0, min(N_MOVE_SLOTS - 1, move_index)))
    ti = int(max(0, min(N_TARGETS - 1, target)))
    return mi * N_TARGETS + ti


def decode_slot_action(action_id: int) -> SlotAction:
    aid = int(action_id)
    if aid < 0 or aid >= SLOT_ACTION_N:
        return SlotAction(kind='pass', raw_id=PASS_ID)
    if aid == PASS_ID or aid == RESERVED_ID:
        return SlotAction(kind='pass', raw_id=aid)
    if aid == MEGA_ID:
        return SlotAction(kind='mega', move_index=0, target=0, raw_id=aid)
    if aid == TERA_ID:
        return SlotAction(kind='tera', move_index=0, target=0, raw_id=aid)
    if 16 <= aid <= 19:
        return SlotAction(kind='switch', switch_index=aid - 16, raw_id=aid)
    mi = aid // N_TARGETS
    ti = aid % N_TARGETS
    return SlotAction(kind='move', move_index=mi, target=ti, raw_id=aid)


def apply_logit_mask(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax(z + (1-M)*(-1e9)). Returns masked logits (not probs)."""
    z = np.asarray(logits, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if z.shape != m.shape:
        raise ValueError(f'logits/mask shape mismatch {z.shape} vs {m.shape}')
    return z + (1.0 - m) * ILLEGAL_LOGIT


def softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = apply_logit_mask(logits, mask)
    # Numerical stability
    x = masked - np.max(masked)
    e = np.exp(x)
    # Zero-out illegal mass completely
    e = e * (np.asarray(mask, dtype=np.float64) > 0.5)
    s = e.sum()
    if s <= 0:
        # Fallback uniform over legal — or all-pass
        legal = np.asarray(mask, dtype=np.float64) > 0.5
        if not legal.any():
            out = np.zeros_like(e)
            out[PASS_ID] = 1.0
            return out
        out = legal.astype(np.float64)
        return out / out.sum()
    return e / s


def build_slot_mask(
    battle: BattleSnapshot,
    slot: int,
    *,
    other_slot_action: Optional[SlotAction] = None,
) -> np.ndarray:
    """
    Build M ∈ {0,1}^24 for one active slot from snapshot legality.

    Source of truth in live env is poke-env valid_orders; this snapshot mask
    is a Spec-compatible approximation for offline / tests / bootstrap.
    """
    mask = np.zeros(SLOT_ACTION_N, dtype=np.float32)
    mask[PASS_ID] = 1.0
    mask[RESERVED_ID] = 0.0

    force = battle.force_switch[slot] if slot < len(battle.force_switch) else False
    active = None
    if slot < len(battle.our.active):
        active = battle.our.active[slot]

    # Switches
    taken_bench: set = set()
    if other_slot_action is not None and other_slot_action.kind == 'switch':
        taken_bench.add(int(other_slot_action.switch_index))

    for bi, mon in enumerate(battle.our.bench):
        if mon is None or mon.fainted or mon.hp_pct <= 0:
            continue
        if bi in taken_bench:
            continue
        mask[16 + bi] = 1.0

    if force:
        # Only switches (+ pass if no switch — PS may require switch)
        if not any(mask[16:20] > 0):
            mask[PASS_ID] = 1.0
        return mask

    if active is None or active.fainted:
        return mask

    n_moves = len(active.moves)
    for mi in range(N_MOVE_SLOTS):
        if mi >= n_moves or not active.moves[mi]:
            continue
        for ti in range(N_TARGETS):
            mask[mi * N_TARGETS + ti] = 1.0

    if active.can_mega:
        mask[MEGA_ID] = 1.0
    # Reg M-C: Terastallization disabled (id 22 stays illegal).
    if allow_tera(active.can_tera):
        mask[TERA_ID] = 1.0
    return mask


def build_autoregressive_masks(
    battle: BattleSnapshot,
    slot1_action_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mask for slot1; mask for slot2 conditioned on slot1 choice."""
    m1 = build_slot_mask(battle, 0)
    s1 = decode_slot_action(slot1_action_id) if slot1_action_id is not None else None
    if s1 is None:
        # Unconditional slot2 mask (union) for BDQN-style; PPO should pass slot1 first.
        m2 = build_slot_mask(battle, 1, other_slot_action=None)
    else:
        m2 = build_slot_mask(battle, 1, other_slot_action=s1)
    return m1, m2


def resolve_action_or_fallback(
    action_id: int,
    mask: np.ndarray,
) -> Tuple[int, bool]:
    """
    L2 fallback: if illegal, pick first legal (prefer PASS then lowest index).
    Returns (resolved_id, was_fallback).
    """
    aid = int(action_id)
    m = np.asarray(mask)
    if 0 <= aid < len(m) and m[aid] > 0.5:
        return aid, False
    # Prefer PASS if legal
    if PASS_ID < len(m) and m[PASS_ID] > 0.5:
        return PASS_ID, True
    legal = np.flatnonzero(m > 0.5)
    if len(legal) == 0:
        return PASS_ID, True
    return int(legal[0]), True


def legal_joint_mask_from_slot_masks(
    m1: np.ndarray,
    m2_for_a1,
) -> np.ndarray:
    """
    Build M_joint ∈ {0,1}^576.

    m2_for_a1: callable(a1: int) -> mask(24,) conditioned on slot1 choice.
    """
    out = np.zeros(JOINT_ACTION_N, dtype=np.float32)
    m1a = np.asarray(m1, dtype=np.float32).reshape(-1)
    for a1 in range(min(SLOT_ACTION_N, m1a.shape[0])):
        if m1a[a1] <= 0.5:
            continue
        m2 = np.asarray(m2_for_a1(a1), dtype=np.float32).reshape(-1)
        base = a1 * SLOT_ACTION_N
        for a2 in range(min(SLOT_ACTION_N, m2.shape[0])):
            if m2[a2] > 0.5:
                out[base + a2] = 1.0
    if not np.any(out > 0.5):
        # Safe fallback: both pass
        out[encode_joint_action(PASS_ID, PASS_ID)] = 1.0
    return out


def legal_joint_mask(battle: BattleSnapshot) -> np.ndarray:
    """Canonical joint mask from BattleSnapshot (offline / Mock)."""
    m1 = build_slot_mask(battle, 0)

    def _m2(a1: int) -> np.ndarray:
        return build_slot_mask(battle, 1, other_slot_action=decode_slot_action(a1))

    return legal_joint_mask_from_slot_masks(m1, _m2)


def apply_q_mask(q_values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Q + (1-M)*(-1e9) for masked argmax."""
    return apply_logit_mask(q_values, mask)
