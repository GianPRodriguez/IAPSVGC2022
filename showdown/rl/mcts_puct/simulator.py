"""S2 in-process BattleSnapshot transitions (docs/mcts_puct_spec.md)."""
from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np

from showdown.rl.action_space import (
    decode_joint_action,
    decode_slot_action,
    legal_joint_mask,
)
from showdown.rl.damage_heuristic import estimate_damage_pct
from showdown.rl.snapshot import BattleSnapshot, MonSnapshot, SideSnapshot


def clone_state(snap: BattleSnapshot) -> BattleSnapshot:
    return copy.deepcopy(snap)


def flip_perspective(snap: BattleSnapshot) -> BattleSnapshot:
    """Swap our/foe so opp legal masks / greedy can reuse 'our' helpers."""
    out = clone_state(snap)
    out.our, out.foe = out.foe, out.our
    f = out.field
    f.our_tailwind, f.opp_tailwind = f.opp_tailwind, f.our_tailwind
    f.our_tailwind_turns, f.opp_tailwind_turns = f.opp_tailwind_turns, f.our_tailwind_turns
    f.our_reflect, f.opp_reflect = f.opp_reflect, f.our_reflect
    f.our_lightscreen, f.opp_lightscreen = f.opp_lightscreen, f.our_lightscreen
    f.our_auroraveil, f.opp_auroraveil = f.opp_auroraveil, f.our_auroraveil
    if out.won is not None:
        out.won = not out.won
    return out


def legal_joints(snap: BattleSnapshot) -> np.ndarray:
    """Indices of legal joint actions."""
    m = legal_joint_mask(snap)
    return np.flatnonzero(m > 0.5).astype(np.int64)


def cand1_return(snap: BattleSnapshot) -> float:
    """Candidate 1 sparse from root (our) perspective. Turn-cap → 0."""
    if not snap.finished:
        return 0.0
    if snap.won is True:
        return 1.0
    if snap.won is False:
        return -1.0
    return 0.0


def _apply_damage(
    attacker: Optional[MonSnapshot],
    defender: Optional[MonSnapshot],
    move_name: str,
    weather: str,
    rng: np.random.Generator,
) -> None:
    if attacker is None or defender is None or defender.fainted:
        return
    lo, hi = estimate_damage_pct(
        move_name,
        attacker_types=attacker.types,
        defender_types=defender.types,
        weather=weather,
        base_atk_spa=100.0 * (0.5 + float(getattr(attacker, 'atk_belief', 0.5))),
        base_def_spd=100.0,
    )
    if hi <= 0:
        # Fallback crude chip so rollouts resolve
        dmg = float(rng.uniform(8.0, 18.0))
    else:
        dmg = float(rng.uniform(max(0.0, lo), max(lo, hi)))
    # Accuracy sample: miss ~10% if move unknown accuracy — keep simple 95%
    if rng.random() > 0.95:
        return
    defender.hp_pct = max(0.0, float(defender.hp_pct) - dmg)
    if defender.hp_pct <= 0:
        defender.fainted = True
        defender.hp_pct = 0.0


def _apply_actor(
    snap: BattleSnapshot,
    joint: int,
    rng: np.random.Generator,
    *,
    actor_is_our: bool,
) -> None:
    a1, a2 = decode_joint_action(int(joint))
    side: SideSnapshot = snap.our if actor_is_our else snap.foe
    opp: SideSnapshot = snap.foe if actor_is_our else snap.our
    weather = str(getattr(snap.field, 'weather', '') or '')

    for slot, aid in ((0, a1), (1, a2)):
        act = decode_slot_action(int(aid))
        if act.kind == 'switch':
            bi = int(act.switch_index)
            if bi < len(side.bench) and side.bench[bi] is not None:
                outgoing = side.active[slot] if slot < len(side.active) else None
                incoming = side.bench[bi]
                if slot < len(side.active):
                    side.active[slot] = incoming
                side.bench[bi] = outgoing
                if incoming is not None:
                    incoming.is_active = True
                if outgoing is not None:
                    outgoing.is_active = False
            continue
        if act.kind not in ('move', 'mega', 'tera'):
            continue
        atk = side.active[slot] if slot < len(side.active) else None
        if atk is None or atk.fainted:
            continue
        mi = int(act.move_index)
        move_name = atk.moves[mi] if mi < len(atk.moves) else ''
        ti = int(min(max(act.target, 0), 1))
        dfn = opp.active[ti] if ti < len(opp.active) else None
        _apply_damage(atk, dfn, move_name, weather, rng)


def step_sim(
    snap: BattleSnapshot,
    joint_self: int,
    joint_opp: int,
    rng: np.random.Generator,
    *,
    max_turns: int = 40,
) -> BattleSnapshot:
    """
    Apply simultaneous crude transitions for both sides (True MC).

    Order: self actions then opp actions (approximation; simultaneous in spirit).
    """
    s = clone_state(snap)
    if s.finished:
        return s
    try:
        _apply_actor(s, int(joint_self), rng, actor_is_our=True)
        _apply_actor(s, int(joint_opp), rng, actor_is_our=False)
    except Exception:
        # Leave state as-is; caller may treat as failure
        pass
    s.turn = int(s.turn) + 1
    foe_alive = any(m is not None and not m.fainted for m in s.foe.all_mons())
    our_alive = any(m is not None and not m.fainted for m in s.our.all_mons())
    if not foe_alive:
        s.finished = True
        s.won = True
        s.tied = False
    elif not our_alive:
        s.finished = True
        s.won = False
        s.tied = False
    elif s.turn > int(max_turns):
        s.finished = True
        s.won = None
        s.tied = False
    return s


def sample_uniform_joint(snap: BattleSnapshot, rng: np.random.Generator) -> int:
    legal = legal_joints(snap)
    if legal.size == 0:
        from showdown.rl.constants import PASS_ID
        from showdown.rl.action_space import encode_joint_action
        return encode_joint_action(PASS_ID, PASS_ID)
    return int(rng.choice(legal))
