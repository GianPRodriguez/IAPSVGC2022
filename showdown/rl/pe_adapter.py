"""Convert poke-env DoubleBattle ↔ Spec snapshots / orders (docs/rl_env_spec.md v1.1)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from showdown.format_rules import allow_tera
from showdown.rl.action_space import (
    PASS_ID,
    SLOT_ACTION_N,
    decode_slot_action,
    legal_joint_mask_from_slot_masks,
)
from showdown.rl.constants import JOINT_ACTION_N, N_MOVE_SLOTS, N_SWITCH_SLOTS
from showdown.rl.snapshot import (
    BattleSnapshot,
    FieldSnapshot,
    MonSnapshot,
    SideSnapshot,
    SlotAction,
)

# Spec target index → poke-env move_target
# Spec: 0 foe_a, 1 foe_b, 2 ally, 3 self/field
_SPEC_TARGET_TO_PE = {0: 1, 1: 2, 2: -1, 3: 0}


def _norm(s: str) -> str:
    return str(s or '').strip().lower().replace(' ', '').replace('-', '').replace('_', '')


def _type_tuple(mon) -> Tuple[str, ...]:
    types = getattr(mon, 'types', None) or ()
    out: List[str] = []
    for t in types:
        if t is None:
            continue
        name = getattr(t, 'name', None) or str(t)
        out.append(str(name).replace('_', ' ').title())
    return tuple(out)


def _species(mon) -> str:
    if mon is None:
        return ''
    for attr in ('species', 'name', 'base_species'):
        v = getattr(mon, attr, None)
        if v:
            return str(v)
    return ''


def _hp_pct(mon) -> float:
    if mon is None or getattr(mon, 'fainted', False):
        return 0.0
    frac = getattr(mon, 'current_hp_fraction', None)
    if frac is not None:
        try:
            return max(0.0, min(100.0, 100.0 * float(frac)))
        except Exception:
            pass
    cur = getattr(mon, 'current_hp', None)
    mx = getattr(mon, 'max_hp', None)
    if cur is not None and mx:
        try:
            return max(0.0, min(100.0, 100.0 * float(cur) / float(mx)))
        except Exception:
            pass
    return 100.0


def _boosts(mon) -> Dict[str, int]:
    raw = getattr(mon, 'boosts', None) or {}
    out: Dict[str, int] = {}
    for k, v in dict(raw).items():
        try:
            out[str(k).lower()] = int(v)
        except Exception:
            continue
    return out


def _status(mon) -> str:
    st = getattr(mon, 'status', None)
    if st is None:
        return ''
    return str(getattr(st, 'name', st) or '').lower()


def _volatiles(mon) -> Dict[str, float]:
    out: Dict[str, float] = {}
    effects = getattr(mon, 'effects', None) or {}
    for k, v in dict(effects).items():
        key = _norm(getattr(k, 'name', k))
        try:
            out[key] = float(v) if v is not None else 1.0
        except Exception:
            out[key] = 1.0
    if getattr(mon, 'is_dynamaxed', False):
        out['dynamax'] = 1.0
    return out


def _moves_and_pp(mon) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    moves_attr = getattr(mon, 'moves', None) or {}
    names: List[str] = []
    pps: List[int] = []
    if isinstance(moves_attr, dict):
        for mv in list(moves_attr.values())[:4]:
            if mv is None:
                continue
            mid = getattr(mv, 'id', None) or getattr(mv, 'name', None) or ''
            names.append(str(mid))
            pps.append(int(getattr(mv, 'current_pp', getattr(mv, 'max_pp', 0)) or 0))
    return tuple(names), tuple(pps)


def _ability_item(mon) -> Tuple[str, str, str, str]:
    ab = getattr(mon, 'ability', None)
    ab_s = str(getattr(ab, 'name', ab) or '') if ab else ''
    ab_known = 'confirmed' if ab_s else 'unknown'
    it = getattr(mon, 'item', None)
    it_s = str(it or '') if it else ''
    it_known = 'confirmed' if it_s else 'unknown'
    return ab_s, ab_known, it_s, it_known


def mon_from_pe(mon, *, is_active: bool, ots_species: str = '') -> Optional[MonSnapshot]:
    if mon is None:
        return None
    moves, pps = _moves_and_pp(mon)
    ab, ab_k, it, it_k = _ability_item(mon)
    sp = _species(mon)
    return MonSnapshot(
        species=sp,
        ots_species=ots_species or sp,
        hp_pct=_hp_pct(mon),
        max_hp=float(getattr(mon, 'max_hp', 1) or 1),
        fainted=bool(getattr(mon, 'fainted', False)),
        types=_type_tuple(mon),
        ability=ab,
        ability_known=ab_k,
        item=it,
        item_known=it_k,
        boosts=_boosts(mon),
        status=_status(mon),
        volatiles=_volatiles(mon),
        moves=moves,
        move_pps=pps,
        revealed=True,
        is_active=is_active,
        just_switched=bool(getattr(mon, 'first_turn', False) or getattr(mon, 'active', False) and getattr(mon, 'just_switched', False)),
        can_mega=False,
        # Reg M-C: never expose Tera as available (even if mon has tera_type metadata).
        can_tera=False,
        tera_type='',
    )


def _field_key(key) -> str:
    return _norm(getattr(key, 'name', key))


def field_from_pe(battle) -> FieldSnapshot:
    weather = getattr(battle, 'weather', None)
    if isinstance(weather, dict):
        wname = next(iter(weather.keys()), '') if weather else ''
    else:
        wname = weather or ''
    fields = getattr(battle, 'fields', None) or {}
    field_ids = {_field_key(k) for k in fields.keys()}
    our_sc = getattr(battle, 'side_conditions', None) or {}
    opp_sc = getattr(battle, 'opponent_side_conditions', None) or {}

    def _sc(sc, name: str) -> bool:
        for k in sc.keys():
            if name in _field_key(k):
                return True
        return False

    terrain = ''
    for name in ('electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain'):
        if name in field_ids:
            terrain = name
            break
    return FieldSnapshot(
        weather=_norm(getattr(wname, 'name', wname)),
        terrain=terrain,
        trick_room='trickroom' in field_ids,
        our_tailwind=_sc(our_sc, 'tailwind'),
        opp_tailwind=_sc(opp_sc, 'tailwind'),
        our_reflect=_sc(our_sc, 'reflect'),
        our_lightscreen=_sc(our_sc, 'lightscreen'),
        our_auroraveil=_sc(our_sc, 'auroraveil'),
        opp_reflect=_sc(opp_sc, 'reflect'),
        opp_lightscreen=_sc(opp_sc, 'lightscreen'),
        opp_auroraveil=_sc(opp_sc, 'auroraveil'),
        gravity='gravity' in field_ids,
    )


def _side_from_pe(
    actives: Sequence,
    team_values: Sequence,
    *,
    can_mega: Sequence[bool] = (),
    can_tera: Sequence[bool] = (),
    ots_map: Optional[Dict[str, str]] = None,
) -> SideSnapshot:
    ots_map = ots_map or {}
    active_snaps: List[Optional[MonSnapshot]] = [None, None]
    for i in range(2):
        mon = actives[i] if i < len(actives) else None
        snap = mon_from_pe(
            mon,
            is_active=True,
            ots_species=ots_map.get(_norm(_species(mon)), ''),
        )
        if snap is not None:
            if i < len(can_mega):
                snap.can_mega = bool(can_mega[i])
            if i < len(can_tera):
                snap.can_tera = allow_tera(bool(can_tera[i]))
            else:
                snap.can_tera = False
        active_snaps[i] = snap

    active_ids = {_norm(_species(m)) for m in actives if m is not None}
    bench: List[Optional[MonSnapshot]] = []
    for mon in team_values:
        if mon is None:
            continue
        if _norm(_species(mon)) in active_ids and not getattr(mon, 'fainted', False):
            # Same species may appear twice; prefer identity
            if mon in actives:
                continue
        if mon in list(actives):
            continue
        bench.append(
            mon_from_pe(
                mon,
                is_active=False,
                ots_species=ots_map.get(_norm(_species(mon)), ''),
            )
        )
    while len(bench) < 2:
        bench.append(None)
    return SideSnapshot(active=active_snaps, bench=bench[:2])


def snapshot_from_battle(
    battle,
    *,
    ots_our: Optional[Dict[str, str]] = None,
    ots_foe: Optional[Dict[str, str]] = None,
) -> BattleSnapshot:
    """Build Spec BattleSnapshot from poke-env DoubleBattle."""
    our_team = list((getattr(battle, 'team', None) or {}).values())
    foe_team = list((getattr(battle, 'opponent_team', None) or {}).values())
    actives = list(getattr(battle, 'active_pokemon', None) or [None, None])
    foes = list(getattr(battle, 'opponent_active_pokemon', None) or [None, None])
    can_mega = list(getattr(battle, 'can_mega_evolve', None) or [False, False])
    # Reg M-C: strip engine Tera flags before snapshot.
    can_tera = [allow_tera(bool(x)) for x in (getattr(battle, 'can_tera', None) or [False, False])]

    finished = bool(getattr(battle, 'finished', False))
    won = getattr(battle, 'won', None)
    tied = False
    if finished and won is None:
        tied = True

    force = tuple(bool(x) for x in (getattr(battle, 'force_switch', None) or [False, False]))[:2]
    if len(force) < 2:
        force = (force[0] if force else False, False)

    return BattleSnapshot(
        turn=int(getattr(battle, 'turn', 0) or 0),
        our=_side_from_pe(actives, our_team, can_mega=can_mega, can_tera=can_tera, ots_map=ots_our),
        foe=_side_from_pe(foes, foe_team, ots_map=ots_foe),
        field=field_from_pe(battle),
        finished=finished,
        won=bool(won) if won is not None else None,
        tied=tied,
        force_switch=(force[0], force[1]),
        player_id=str(getattr(battle, 'player_role', None) or 'p1'),
    )


def _create_order(obj, **kwargs):
    from poke_env.player.player import Player
    return Player.create_order(obj, **kwargs)


def slot_action_to_pe_order(battle, slot: int, act: SlotAction):
    """Convert Spec SlotAction → poke-env SingleBattleOrder (best effort)."""
    from poke_env.player.battle_order import PassBattleOrder

    force = False
    fs = getattr(battle, 'force_switch', None) or []
    if slot < len(fs):
        force = bool(fs[slot])

    if act.kind == 'pass' or force and act.kind not in ('switch',):
        if force:
            switches = list(battle.available_switches[slot]) if slot < len(battle.available_switches) else []
            if switches:
                return _create_order(switches[0])
        return PassBattleOrder()

    if act.kind == 'switch':
        switches = list(battle.available_switches[slot]) if slot < len(battle.available_switches) else []
        if not switches:
            return PassBattleOrder()
        idx = int(max(0, min(len(switches) - 1, act.switch_index)))
        return _create_order(switches[idx])

    moves = list(battle.available_moves[slot]) if slot < len(battle.available_moves) else []
    if not moves:
        return PassBattleOrder()
    mi = int(max(0, min(len(moves) - 1, act.move_index)))
    move = moves[mi]
    target = _SPEC_TARGET_TO_PE.get(int(act.target), 1)
    return _create_order(
        move,
        move_target=target,
        mega=(act.kind == 'mega'),
        terastallize=allow_tera(act.kind == 'tera'),
    )


def joint_order_from_spec_actions(
    battle,
    action: Tuple[int, int],
) -> Tuple[object, bool]:
    """
    Build DoubleBattleOrder from Spec MultiDiscrete action.
    Returns (order, used_fallback).
    """
    from poke_env.player.battle_order import DoubleBattleOrder
    from poke_env.player.player import Player

    a0 = decode_slot_action(int(action[0]))
    a1 = decode_slot_action(int(action[1]))
    o1 = slot_action_to_pe_order(battle, 0, a0)
    o2 = slot_action_to_pe_order(battle, 1, a1)
    joined = DoubleBattleOrder.join_orders([o1], [o2])
    if joined:
        order = joined[0]
        # Validate against valid_orders when present
        try:
            vo = getattr(battle, 'valid_orders', None)
            if vo and len(vo) >= 2:
                legal1 = {str(x) for x in (vo[0] or [])}
                legal2 = {str(x) for x in (vo[1] or [])}
                if legal1 and str(o1) not in legal1:
                    raise ValueError('slot0 illegal')
                if legal2 and str(o2) not in legal2:
                    raise ValueError('slot1 illegal')
        except ValueError:
            return Player.choose_random_doubles_move(battle), True
        return order, False
    return Player.choose_random_doubles_move(battle), True


def pe_slot_mask(battle, slot: int, *, other: Optional[SlotAction] = None) -> np.ndarray:
    """Legal mask from poke-env available moves/switches (+ Spec PASS)."""
    mask = np.zeros(SLOT_ACTION_N, dtype=np.float32)
    mask[PASS_ID] = 1.0
    force = False
    fs = getattr(battle, 'force_switch', None) or []
    if slot < len(fs):
        force = bool(fs[slot])

    switches = list(battle.available_switches[slot]) if slot < len(getattr(battle, 'available_switches', [])) else []
    taken = set()
    if other is not None and other.kind == 'switch':
        taken.add(int(other.switch_index))
    for bi in range(min(N_SWITCH_SLOTS, len(switches))):
        if bi in taken:
            continue
        mask[16 + bi] = 1.0

    if force:
        return mask

    moves = list(battle.available_moves[slot]) if slot < len(getattr(battle, 'available_moves', [])) else []
    for mi in range(min(N_MOVE_SLOTS, len(moves))):
        for ti in range(4):
            mask[mi * 4 + ti] = 1.0

    can_mega = list(getattr(battle, 'can_mega_evolve', None) or [])
    can_tera = list(getattr(battle, 'can_tera', None) or [])
    if slot < len(can_mega) and can_mega[slot]:
        mask[21] = 1.0
    if slot < len(can_tera) and allow_tera(bool(can_tera[slot])):
        mask[22] = 1.0
    return mask


def pe_autoregressive_masks(battle, slot1_action_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    m1 = pe_slot_mask(battle, 0)
    other = decode_slot_action(slot1_action_id) if slot1_action_id is not None else None
    m2 = pe_slot_mask(battle, 1, other=other)
    return m1, m2


def legal_joint_mask(battle) -> np.ndarray:
    """
    M_joint ∈ {0,1}^576 from poke-env available moves/switches.

    When `valid_orders` is present, further restrict pairs whose decoded orders
    are not in the legal order sets (best-effort string match).
    """
    m1 = pe_slot_mask(battle, 0)

    def _m2(a1: int) -> np.ndarray:
        return pe_slot_mask(battle, 1, other=decode_slot_action(a1))

    joint = legal_joint_mask_from_slot_masks(m1, _m2)

    vo = getattr(battle, 'valid_orders', None)
    if not vo or len(vo) < 2:
        return joint

    try:
        legal1 = {str(x) for x in (vo[0] or [])}
        legal2 = {str(x) for x in (vo[1] or [])}
    except Exception:
        return joint
    if not legal1 and not legal2:
        return joint

    # Filter joint ids whose slot orders are both representable & legal.
    from showdown.rl.action_space import decode_joint_action
    filtered = np.zeros(JOINT_ACTION_N, dtype=np.float32)
    for j in np.flatnonzero(joint > 0.5):
        a1, a2 = decode_joint_action(int(j))
        o1 = slot_action_to_pe_order(battle, 0, decode_slot_action(a1))
        o2 = slot_action_to_pe_order(battle, 1, decode_slot_action(a2))
        ok1 = (not legal1) or (str(o1) in legal1)
        ok2 = (not legal2) or (str(o2) in legal2)
        if ok1 and ok2:
            filtered[int(j)] = 1.0
    if np.any(filtered > 0.5):
        return filtered
    return joint
