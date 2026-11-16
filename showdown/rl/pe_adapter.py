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
from showdown.rl.constants import (
    JOINT_ACTION_N,
    MEGA_BASE,
    N_MOVE_ACTIONS,
    N_MOVE_SLOTS,
    N_SWITCH_SLOTS,
    SWITCH_BASE,
    TERA_ID,
)
from showdown.rl.snapshot import (
    BattleSnapshot,
    FieldSnapshot,
    MonSnapshot,
    SideSnapshot,
    SlotAction,
)

# Spec target index → poke-env move_target
# Spec: 0 foe_a, 1 foe_b, 2 ally, 3 self/field
_SPEC_TARGET_TO_PE = {0: 1, 1: 2, 2: -1, 3: 0}  # legado; usar spec_target_to_pe


def spec_target_to_pe(slot: int, target: int, move=None) -> int:
    """Objetivo Spec → posición Showdown, **relativa al slot**.

    0/1 = rival a/b (1/2). 2 = aliado: −2 para el slot 0 y −1 para el slot 1
    (antes era −1 fijo: el slot 0 "apuntaba al aliado" con su propia posición,
    Showdown lo rechazaba y el slot 0 nunca podía apuntar a su aliado). 3 = sin
    objetivo (0), salvo movimientos ``adjacentAllyOrSelf`` → la propia posición.
    """
    t = int(target)
    if t == 0:
        return 1
    if t == 1:
        return 2
    self_pos, ally_pos = (-1, -2) if int(slot) == 0 else (-2, -1)
    if t == 2:
        return ally_pos
    tgt = getattr(getattr(move, 'target', None), 'name', '') if move is not None else ''
    if str(tgt).upper() == 'ADJACENT_ALLY_OR_SELF':
        return self_pos
    return 0


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


def _exact_stats(mon) -> Optional[Dict[str, int]]:
    """Stats finales del |request| (solo equipo propio)."""
    st = dict(getattr(mon, 'stats', None) or {})
    hp = getattr(mon, 'max_hp', None)
    keys = ('atk', 'def', 'spa', 'spd', 'spe')
    if not hp or not all(st.get(k) for k in keys):
        return None
    return {'hp': int(hp), **{k: int(st[k]) for k in keys}}


def mon_from_pe(mon, *, is_active: bool, ots_species: str = '', own: bool = False) -> Optional[MonSnapshot]:
    if mon is None:
        return None
    moves, pps = _moves_and_pp(mon)
    ab, ab_k, it, it_k = _ability_item(mon)
    if it == 'unknown_item':
        it, it_k = '', 'unknown'
    sp = _species(mon)
    return MonSnapshot(
        stats=_exact_stats(mon) if own else None,
        nature='' if own else str(getattr(mon, 'nature', '') or ''),
        seen=True if own else bool(getattr(mon, 'revealed', True)),
        mega_evolved='mega' in _norm(sp),
        status_counter=int(getattr(mon, 'status_counter', 0) or 0),
        protect_counter=int(getattr(mon, 'protect_counter', 0) or 0),
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
    own: bool = False,
    switch_order: Sequence = (),
    mega_used: bool = False,
    keys: Optional[Dict[int, str]] = None,
    hist=None,
) -> SideSnapshot:
    ots_map = ots_map or {}
    active_snaps: List[Optional[MonSnapshot]] = [None, None]
    for i in range(2):
        mon = actives[i] if i < len(actives) else None
        snap = mon_from_pe(
            mon,
            is_active=True,
            ots_species=ots_map.get(_norm(_species(mon)), ''),
            own=own,
        )
        if snap is not None:
            _apply_history(snap, (keys or {}).get(id(mon)), hist, active=True)
            if i < len(can_mega):
                snap.can_mega = bool(can_mega[i])
            if i < len(can_tera):
                snap.can_tera = allow_tera(bool(can_tera[i]))
            else:
                snap.can_tera = False
        active_snaps[i] = snap

    non_active = [m for m in team_values if m is not None and m not in list(actives)]
    if own and switch_order:
        # Banca propia en el orden de available_switches (= índice de la acción
        # de cambio), luego el resto (debilitados).
        first = [m for m in switch_order if m in non_active]
        non_active = first + [m for m in non_active if m not in first]
    elif not own:
        # Rival: primero los ya vistos en combate (la hoja muestra 6, trae 4).
        non_active = ([m for m in non_active if getattr(m, 'revealed', True)]
                      + [m for m in non_active if not getattr(m, 'revealed', True)])
    reserve: List[MonSnapshot] = []
    for mon in non_active:
        snap = mon_from_pe(
            mon,
            is_active=False,
            ots_species=ots_map.get(_norm(_species(mon)), ''),
            own=own,
        )
        if snap is not None:
            _apply_history(snap, (keys or {}).get(id(mon)), hist, active=False)
            reserve.append(snap)
    bench: List[Optional[MonSnapshot]] = list(reserve[:2])
    while len(bench) < 2:
        bench.append(None)
    return SideSnapshot(active=active_snaps, bench=bench[:2], reserve=reserve,
                        mega_used=bool(mega_used))


def _apply_history(snap: MonSnapshot, key: Optional[str], hist, *, active: bool) -> None:
    """Historial del protocolo → campos del snapshot (último movimiento, bloqueo…)."""
    if hist is None or key is None:
        return
    h = hist.mon(key)
    if h is None:
        return
    if h.last_move_turn >= 0 and h.last_move:
        snap.last_move = h.last_move
        snap.last_move_ago = max(0, int(hist.turn) - int(h.last_move_turn))
    if active and h.switch_in_turn >= 0:
        snap.turns_active = max(0, int(hist.turn) - int(h.switch_in_turn))
    snap.moves_since_switch = int(h.moves_since_switch) if active else 0
    snap.times_switched_in = int(h.times_switched_in)
    snap.hp_lost_last_turn = float(hist.hp_lost_last_turn(key))
    item = _norm(snap.item)
    vol = {_norm(k) for k in (snap.volatiles or {})}
    if active and snap.last_move and snap.moves_since_switch > 0 and (
            item.startswith('choice') or 'encore' in vol):
        snap.locked_move = snap.last_move


def _apply_brought_belief(side: SideSnapshot) -> None:
    """Rival: la hoja muestra 6 y trae 4. Con k vistos, cada no visto tiene
    P(traído) = (4−k)/(6−k); con k ≥ 4 los no vistos quedan descartados."""
    mons = [m for m in list(side.active) + list(side.reserve) if m is not None]
    k = sum(1 for m in mons if m.seen)
    unseen = [m for m in mons if not m.seen]
    for m in unseen:
        if k >= 4:
            m.p_brought, m.excluded = 0.0, True
        else:
            m.p_brought = (4.0 - k) / max(1.0, 6.0 - k)


def _last_turn_summary(hist, our_role: str) -> Dict[str, float]:
    if hist is None:
        return {}
    foe_role = 'p2' if our_role == 'p1' else 'p1'
    out: Dict[str, float] = {}
    for side, role in (('our', our_role), ('foe', foe_role)):
        counts = hist.last_turn.get(role, {}) if hist.last_turn else {}
        for k in ('protect', 'switch', 'faint', 'fakeout', 'spread'):
            out[f'{side}_{k}'] = float(counts.get(k, 0.0))
    return out


def stat_inference_enabled() -> bool:
    """``OBS_STAT_INFERENCE=1`` activa los rangos de SP inferidos en la obs.

    Apagado por defecto: debe activarse en entrenamiento E inferencia a la vez
    (si la red entrena con rangos completos y en inferencia ve rangos
    angostados, las entradas cambian de distribución).
    """
    import os
    return os.environ.get('OBS_STAT_INFERENCE', '').strip().lower() in ('1', 'true', 'yes')


def player_stat_beliefs(player) -> Optional[object]:
    """``TeamStatBeliefs`` del espejo IAP del jugador (ValidatedBattleMixin), si existe."""
    if not stat_inference_enabled():
        return None
    iap = getattr(player, 'iap_battle', None)
    return getattr(iap, 'stat_beliefs', None) if iap is not None else None


def sp_bounds_from_beliefs(stat_beliefs, species: str) -> Optional[Dict[str, Tuple[int, int]]]:
    """SP mínimo/máximo aún posibles por stat para un rival (inferencia por daño/velocidad)."""
    if stat_beliefs is None or not species:
        return None
    try:
        from showdown.stat_inference import normalize_species
        table = getattr(stat_beliefs, 'beliefs', None) or {}
        belief = table.get(normalize_species(species))
        if belief is None:
            return None
        out: Dict[str, Tuple[int, int]] = {}
        for stat, cands in (belief.candidates or {}).items():
            if cands:
                out[str(stat)] = (int(min(cands)), int(max(cands)))
        return out or None
    except Exception:
        return None


def _apply_sp_bounds(side: SideSnapshot, stat_beliefs) -> None:
    for mon in list(side.active) + list(side.reserve):
        if mon is not None and mon.stats is None:
            mon.sp_bounds = sp_bounds_from_beliefs(stat_beliefs, mon.display_species())


def snapshot_from_battle(
    battle,
    *,
    ots_our: Optional[Dict[str, str]] = None,
    ots_foe: Optional[Dict[str, str]] = None,
    stat_beliefs=None,
) -> BattleSnapshot:
    """Build Spec BattleSnapshot from poke-env DoubleBattle."""
    team_map = dict(getattr(battle, 'team', None) or {})
    our_team = list(team_map.values())
    # poke-env conserva en ``team`` los 6 del request de preview; el |request| del
    # combate solo trae los 4 elegidos. Filtrar por esos idents evita contar como
    # banca Pokémon que no se trajeron.
    try:
        req_side = (getattr(battle, 'last_request', None) or {}).get('side', {}) or {}
        idents = {str(p.get('ident')) for p in (req_side.get('pokemon') or []) if p.get('ident')}
    except Exception:
        idents = set()
    if idents and not getattr(battle, 'teampreview', False):
        kept = [mon for key, mon in team_map.items() if key in idents]
        if kept:
            our_team = kept
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

    avail_moves_raw = list(getattr(battle, 'available_moves', None) or [[], []])
    avail_sw_raw = list(getattr(battle, 'available_switches', None) or [[], []])
    avail_moves = tuple(
        tuple(str(getattr(mv, 'id', '') or '') for mv in (avail_moves_raw[i] if i < len(avail_moves_raw) else []))
        for i in range(2)
    )
    avail_switches = tuple(
        tuple(_species(m) for m in (avail_sw_raw[i] if i < len(avail_sw_raw) else []))
        for i in range(2)
    )
    switch_order = list(avail_sw_raw[0]) if avail_sw_raw and avail_sw_raw[0] else (
        list(avail_sw_raw[1]) if len(avail_sw_raw) > 1 else []
    )

    from showdown.rl.battle_history import history_from_battle

    hist = history_from_battle(battle)
    our_role = str(getattr(battle, 'player_role', None) or 'p1')
    keys = {id(m): k for k, m in team_map.items()}
    keys.update({id(m): k for k, m in (getattr(battle, 'opponent_team', None) or {}).items()})
    foe_side = _side_from_pe(foes, foe_team, ots_map=ots_foe,
                             mega_used=bool(getattr(battle, 'opponent_used_mega_evolve', False)),
                             keys=keys, hist=hist)
    _apply_brought_belief(foe_side)
    if stat_beliefs is not None:
        _apply_sp_bounds(foe_side, stat_beliefs)

    return BattleSnapshot(
        turn=int(getattr(battle, 'turn', 0) or 0),
        our=_side_from_pe(actives, our_team, can_mega=can_mega, can_tera=can_tera, ots_map=ots_our,
                          own=True, switch_order=switch_order,
                          mega_used=bool(getattr(battle, 'used_mega_evolve', False)),
                          keys=keys, hist=hist),
        foe=foe_side,
        field=field_from_pe(battle),
        finished=finished,
        won=bool(won) if won is not None else None,
        tied=tied,
        force_switch=(force[0], force[1]),
        player_id=str(getattr(battle, 'player_role', None) or 'p1'),
        avail_moves=avail_moves,
        avail_switches=avail_switches,
        last_turn=_last_turn_summary(hist, our_role),
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

    if act.kind == 'pass':
        # Un `pass` en un hueco forzado es legal cuando el otro hueco toma el único
        # reemplazo (la máscara solo lo permite entonces). Antes se convertía en un cambio
        # al mismo Pokémon que el otro hueco: la orden doble se descartaba y se jugaba al azar.
        return PassBattleOrder()
    if force and act.kind not in ('switch',):
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
    target = spec_target_to_pe(slot, int(act.target), move)
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
    from showdown.battle_bots.helpers import enforce_single_mega
    o1, o2 = enforce_single_mega(o1, o2)
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

    # Cambio forzado de OTRO hueco: este hueco solo admite `pass` (Showdown rechaza
    # cualquier otra orden y poke-env entra en un bucle de reintentos).
    if any(bool(x) for x in fs) and not force:
        return mask

    switches = list(battle.available_switches[slot]) if slot < len(getattr(battle, 'available_switches', [])) else []
    taken = set()
    if other is not None and other.kind == 'switch':
        taken.add(int(other.switch_index))
    for bi in range(min(N_SWITCH_SLOTS, len(switches))):
        if bi in taken:
            continue
        mask[SWITCH_BASE + bi] = 1.0

    if force:
        # ¿Puede pasar este hueco forzado? Showdown exige que entre un reemplazo por cada
        # hueco forzado mientras haya reemplazos. Con los dos huecos forzados y un solo
        # Pokémon en la banca, uno cambia y el otro pasa; nunca los dos pasan. Antes la
        # máscara permitía (pass, pass) y el adaptador caía a una jugada al azar.
        has_switch = bool(mask[SWITCH_BASE:SWITCH_BASE + N_SWITCH_SLOTS].any())
        if other is None:
            n_forced = sum(1 for x in fs[:2] if bool(x))
            pool = {id(m) for side in (getattr(battle, 'available_switches', None) or [])[:2]
                    for m in (side or [])}
            can_pass = (not has_switch) or n_forced > len(pool)
        else:
            # Segundo hueco, ya conocida la acción del primero: pasa solo si no le queda
            # ningún reemplazo (el primero tomó el único).
            can_pass = not has_switch
        if not can_pass:
            mask[PASS_ID] = 0.0
        return mask

    moves = list(battle.available_moves[slot]) if slot < len(getattr(battle, 'available_moves', [])) else []
    for mi in range(min(N_MOVE_SLOTS, len(moves))):
        for ti in range(4):
            mask[mi * 4 + ti] = 1.0

    can_mega = list(getattr(battle, 'can_mega_evolve', None) or [])
    can_tera = list(getattr(battle, 'can_tera', None) or [])
    # Mega = flag sobre un movimiento (ids 16–31, mismo movimiento×objetivo);
    # nunca con un cambio. Una sola por combate: si el otro hueco ya la pidió, no.
    other_mega = other is not None and other.kind == 'mega'
    if slot < len(can_mega) and can_mega[slot] and not other_mega:
        mask[MEGA_BASE:MEGA_BASE + N_MOVE_ACTIONS] = mask[0:N_MOVE_ACTIONS]
    if slot < len(can_tera) and allow_tera(bool(can_tera[slot])):
        mask[TERA_ID] = 1.0
    return _filter_ally_damage(battle, slot, _filter_by_valid_orders(battle, slot, mask))


# Habilidad del aliado → tipos que absorbe o que lo benefician al recibirlos: apuntarle un
# ataque de ese tipo puede ser una jugada real (p. ej. Lightning Rod, Justified).
_ALLY_ABSORB = {
    'lightningrod': {'electric'}, 'voltabsorb': {'electric'}, 'motordrive': {'electric'},
    'stormdrain': {'water'}, 'waterabsorb': {'water'}, 'dryskin': {'water'},
    'flashfire': {'fire'}, 'wellbakedbody': {'fire'}, 'sapsipper': {'grass'},
    'eartheater': {'ground'}, 'justified': {'dark'},
}
# Movimientos con poder base que sobre el aliado ayudan en vez de dañar.
_ALLY_OK_MOVES = {'pollenpuff'}


def _filter_ally_damage(battle, slot: int, mask: np.ndarray) -> np.ndarray:
    """``RL_MASK_ALLY_DAMAGE=1``: quita los ATAQUES DE DAÑO dirigidos al propio aliado.

    Se siguen permitiendo los movimientos de estado sobre el aliado (Helping Hand, Coaching,
    Decorate…), los que tienen poder base pero ayudan al aliado (Pollen Puff lo cura) y los
    ataques de un tipo que la habilidad del aliado absorbe o aprovecha (Lightning Rod, Storm
    Drain, Justified…), para que con otros equipos el agente pueda aprender esas sinergias.
    Apagado por defecto.
    """
    import os
    if os.environ.get('RL_MASK_ALLY_DAMAGE', '').strip().lower() not in ('1', 'true', 'yes', 'on'):
        return mask
    moves = list(battle.available_moves[slot]) if slot < len(getattr(battle, 'available_moves', [])) else []
    actives = list(getattr(battle, 'active_pokemon', None) or [])
    ally = actives[1 - slot] if len(actives) == 2 else None
    ally_ab = _norm(getattr(ally, 'ability', '') or '') if ally is not None else ''
    absorbs = _ALLY_ABSORB.get(ally_ab, set())
    out = mask.copy()
    for mi, move in enumerate(moves[:N_MOVE_SLOTS]):
        damaging = float(getattr(move, 'base_power', 0) or 0) > 0
        if not damaging or _norm(getattr(move, 'id', '')) in _ALLY_OK_MOVES:
            continue
        mtype = _norm(getattr(getattr(move, 'type', None), 'name', '') or '')
        if '*' in absorbs or mtype in absorbs:
            continue
        for base in (0, MEGA_BASE):
            out[base + mi * 4 + 2] = 0.0  # objetivo 2 = aliado
    return out if np.any(out > 0.5) else mask


def _valid_order_strings(battle, slot: int) -> Optional[set]:
    try:
        vo = getattr(battle, 'valid_orders', None)
    except Exception:
        return None
    if not vo or slot >= len(vo):
        return None
    legal = {str(x) for x in (vo[slot] or [])}
    return legal or None


def _filter_by_valid_orders(battle, slot: int, mask: np.ndarray) -> np.ndarray:
    """Deja solo los ids cuya orden poke-env está en ``battle.valid_orders[slot]``.

    Sin este filtro la máscara de ENTRENAMIENTO era más permisiva que la de
    inferencia: ``pass`` siempre, los 4 objetivos para cualquier movimiento y
    cambios aunque el Pokémon esté atrapado. Showdown rechazaba esas órdenes, el
    backend las sustituía por una aleatoria (``used_fallback``), la transición se
    descartaba y el bootstrap de DDQN maximizaba sobre Q de ids nunca entrenados.
    Si el filtro dejara la máscara vacía (desfase de strings), se conserva la
    original como defensa.
    """
    legal = _valid_order_strings(battle, slot)
    if legal is None:
        return mask
    out = np.zeros_like(mask)
    for aid in np.flatnonzero(mask > 0.5):
        try:
            order = slot_action_to_pe_order(battle, slot, decode_slot_action(int(aid)))
        except Exception:
            continue
        if str(order) in legal:
            out[int(aid)] = 1.0
    return out if np.any(out > 0.5) else mask


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
