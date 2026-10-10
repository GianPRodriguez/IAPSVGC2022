"""Minimal IAP Battle → BattleSnapshot for DDQN observation / masks."""
from __future__ import annotations

from typing import List, Optional, Tuple

from showdown.rl.action_space import legal_joint_mask
from showdown.rl.snapshot import (
    BattleSnapshot,
    FieldSnapshot,
    MonSnapshot,
    SideSnapshot,
)


def _mon_from_iap(pkmn, *, is_active: bool) -> Optional[MonSnapshot]:
    if pkmn is None:
        return None
    fainted = bool(getattr(pkmn, 'is_fainted', False))
    max_hp = int(getattr(pkmn, 'HPB_battle_base', 100) or 100)
    cur = int(getattr(pkmn, 'HPA_battle_actual', max_hp) or 0)
    hp_pct = 0.0 if fainted or max_hp <= 0 else 100.0 * max(0, cur) / max_hp
    moves = []
    pps = []
    for mv in (getattr(pkmn, 'moves', None) or [])[:4]:
        if mv is None:
            continue
        name = str(getattr(mv, 'id', None) or getattr(mv, 'name', None) or '')
        if name:
            moves.append(name)
            pps.append(int(getattr(mv, 'pp', 1) or 1))
    while len(moves) < 4:
        moves.append('')
        pps.append(0)
    types = tuple(t for t in (getattr(pkmn, 'type1', None), getattr(pkmn, 'type2', None)) if t)
    return MonSnapshot(
        species=str(getattr(pkmn, 'name', '') or ''),
        ots_species=str(getattr(pkmn, 'name', '') or ''),
        hp_pct=float(hp_pct),
        types=types,
        moves=tuple(moves[:4]),
        move_pps=tuple(pps[:4]),
        is_active=is_active,
        revealed=True,
        fainted=fainted,
        ability=str(getattr(pkmn, 'abilityPkmn', '') or ''),
        ability_known='confirmed' if getattr(pkmn, 'abilityPkmn', None) else 'unknown',
        item=str(getattr(pkmn, 'objectPkmn', '') or ''),
        item_known='confirmed' if getattr(pkmn, 'objectPkmn', None) else 'unknown',
        can_mega=bool(getattr(pkmn, 'can_mega', False)),
        can_tera=bool(getattr(pkmn, 'can_tera', False)),
    )


def snapshot_from_iap(battle) -> BattleSnapshot:
    user = battle.user
    opp = battle.opponent
    our_active = [
        _mon_from_iap(getattr(user, 'active1', None), is_active=True),
        _mon_from_iap(getattr(user, 'active2', None), is_active=True),
    ]
    our_bench = [
        _mon_from_iap(r, is_active=False)
        for r in (getattr(user, 'reserved', None) or [])[:2]
    ]
    while len(our_bench) < 2:
        our_bench.append(None)
    foe_active = [
        _mon_from_iap(getattr(opp, 'active1', None), is_active=True),
        _mon_from_iap(getattr(opp, 'active2', None), is_active=True),
    ]
    foe_bench = [
        _mon_from_iap(r, is_active=False)
        for r in (getattr(opp, 'reserved', None) or [])[:2]
    ]
    while len(foe_bench) < 2:
        foe_bench.append(None)

    force = (False, False)
    if getattr(battle, 'force_switch', False):
        fs = (battle.request_json or {}).get('forceSwitch') or [False, False]
        force = (bool(fs[0]) if len(fs) > 0 else False, bool(fs[1]) if len(fs) > 1 else False)

    # Enrich can_mega from request
    active_req = (battle.request_json or {}).get('active') or []
    for i, mon in enumerate(our_active):
        if mon is None or i >= len(active_req):
            continue
        info = active_req[i] or {}
        mon.can_mega = bool(info.get('canMegaEvo', False))
        mon.can_tera = bool(info.get('canTerastallize', False) or info.get('canTera', False))

    return BattleSnapshot(
        turn=int(getattr(battle, 'turn', 0) or 0),
        our=SideSnapshot(active=our_active, bench=our_bench[:2]),
        foe=SideSnapshot(active=foe_active, bench=foe_bench[:2]),
        field=FieldSnapshot(),
        finished=False,
        won=None,
        tied=False,
        force_switch=force,
        player_id='p1',
    )


def mask_and_obs_from_iap(battle):
    from showdown.rl.observation import build_observation_ddqn

    snap = snapshot_from_iap(battle)
    obs = build_observation_ddqn(snap)
    mask = legal_joint_mask(snap)
    return obs, mask, snap


def joint_to_iap_action(joint_id: int, battle) -> tuple:
    """Map joint id → IAP action_battle return fields (partial)."""
    from showdown.rl.action_space import decode_joint_action, decode_slot_action

    a1, a2 = decode_joint_action(int(joint_id))

    def _one(slot_id: int, active_index: int):
        act = decode_slot_action(slot_id)
        if act.kind == 'pass':
            return 'pass', 0, 0, 0
        if act.kind == 'switch':
            # bench 0→slot3, 1→slot4
            return 'switch', 3 + int(act.switch_index), 0, 0
        mega = 1 if act.kind == 'mega' else 0
        # Spec target 0/1 foe → Showdown 1/2; 2 ally → -1/-2; 3 → 0
        ti = int(act.target)
        if ti == 0:
            to_move = 1
        elif ti == 1:
            to_move = 2
        elif ti == 2:
            to_move = -2 if active_index == 0 else -1
        else:
            to_move = 0
        move_slot = int(act.move_index) + 1
        if act.kind in ('mega', 'tera'):
            move_slot = 1
            if act.kind == 'tera':
                mega = 0  # IAP uses separate mega flag; tera via mega=2 in greedy
                # Keep mega field 2 for tera if convention matches greedy_solo
                mega = 2
        return 'move', move_slot, mega, to_move

    k1, n1, m1, t1 = _one(a1, 0)
    k2, n2, m2, t2 = _one(a2, 1)
    # One mega only
    if m1 and m2 and m1 == 1 and m2 == 1:
        m2 = 0
    return k1, n1, m1, t1, k2, n2, m2, t2
