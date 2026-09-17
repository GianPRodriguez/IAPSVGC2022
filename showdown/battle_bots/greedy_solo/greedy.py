"""Greedy-Solo core: max-damage por lado (+ KO / Tera / switches forzados).

Política factored (cada activo elige solo). Micro-opts:
  - Un solo batch de calc para ambos lados activos
  - Status no van al calc (score 0)
  - Tera solo si convierte un hit en KO garantizado
  - Mega por defecto si canMega
  - Side/field flags + alliesFainted
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from showdown.battle_bots.greedy_solo.calc_async import (
    calc_damage_batch,
    damage_pct,
    guaranteed_ko,
)
from showdown.battle_bots.greedy_solo.type_chart import matchup_vs_actives, pokemon_types
from showdown.stat_inference import _poke_payload, _terrain_for_calc, _weather_for_calc


KO_BONUS = 1000.0


def _norm_abi(raw: Optional[str]) -> str:
    return str(raw or '').strip().lower().replace(' ', '').replace('-', '')


@dataclass
class CalcPoke:
    species: str
    nature: str = 'Serious'
    item: Optional[str] = None
    ability: Optional[str] = None
    status: str = ''
    boosts: Optional[Dict[str, int]] = None
    stats: Optional[Dict[str, int]] = None
    cur_hp: Optional[int] = None
    types: Tuple[Optional[str], ...] = ()
    ability_on: bool = False
    just_switched: bool = False
    tera_type: Optional[str] = None
    is_terastallized: bool = False
    is_helping_hand: bool = False
    is_charged: bool = False
    is_protected: bool = False
    has_substitute: bool = False
    is_redirecting: bool = False
    allies_fainted: int = 0
    # Ya en turno 2 de un move con flags.charge (twoturnmove / preparing).
    is_two_turn_charging: bool = False
    # Obligado a recargar este turno (Hyper Beam, etc.).
    must_recharge: bool = False


@dataclass
class FieldContext:
    """Flags de campo + pantallas del bando defensor (rival si atacamos)."""

    weather: Optional[str] = None
    terrain: Optional[str] = None
    wonder_room: bool = False
    magic_room: bool = False
    gravity: bool = False
    reflect: bool = False
    light_screen: bool = False
    aurora_veil: bool = False
    is_beads_of_ruin: bool = False
    is_sword_of_ruin: bool = False
    is_tablets_of_ruin: bool = False
    is_vessel_of_ruin: bool = False


@dataclass
class MoveOption:
    slot: int  # 1-4
    name: str
    target_type: str = 'normal'
    category: str = 'Physical'


@dataclass
class ScoredAction:
    score: float
    move_slot: int
    target: int  # Showdown target: 1/2 foe, -1/-2 ally, 0 none
    mega: int = 0  # 0 none, 1 mega, 2 tera
    damage_pct: float = 0.0
    ko: bool = False
    kind: str = 'move'  # 'move' | 'switch' (move_slot = bench slot si switch)


def _stats_or_default(poke: CalcPoke) -> Dict[str, int]:
    if poke.stats:
        return dict(poke.stats)
    return {'hp': 100, 'atk': 100, 'def': 100, 'spa': 100, 'spd': 100, 'spe': 100}


def apply_belief_stats(poke: CalcPoke, belief) -> CalcPoke:
    """
    Sobrescribe stats/cur_hp del rival con el belief SP.

    - Si el belief está estrecho (pocos SP o resuelto): usa el point-estimate.
    - Si sigue abierto en bulk (HP/Def/SpD): usa el extremo ALTO (más duro de
      tumbar) para no inflar KOs falsos.
    - Si sigue abierto en ofensiva (Atk/SpA/Spe): usa el extremo ALTO (amenaza).
    """
    if belief is None:
        return poke
    try:
        from showdown.stat_inference import champions_stat
    except Exception:
        champions_stat = None

    stats = _stats_or_default(poke)
    for key in ('hp', 'atk', 'def', 'spa', 'spd', 'spe'):
        cands = sorted(getattr(belief, 'candidates', {}).get(key) or [])
        base = int((getattr(belief, 'base_stats', {}) or {}).get(key) or 0)
        if not cands or base <= 0:
            continue
        nature = getattr(belief, 'nature', None)
        try:
            narrow = bool(belief.is_narrow(key))
        except Exception:
            narrow = len(cands) <= 8
        if narrow or champions_stat is None:
            if champions_stat is not None:
                use_sp = None
                try:
                    use_sp = belief.preferred_sp(key)
                except Exception:
                    use_sp = None
                if use_sp is None or use_sp not in set(cands):
                    use_sp = cands[len(cands) // 2]
                stats[key] = champions_stat(base, int(use_sp), nature, key)
            else:
                est = belief.point_estimate_stats()
                if key in est:
                    stats[key] = int(est[key])
            continue
        # Belief ancho: extremos pesimistas para decisión greedy.
        if champions_stat is None:
            continue
        if key in ('hp', 'def', 'spd', 'atk', 'spa', 'spe'):
            use_sp = cands[-1]  # max SP → mayor stat final (bulk/amenaza)
            stats[key] = champions_stat(base, use_sp, nature, key)

    cur = poke.cur_hp
    max_hp = int(stats.get('hp') or 100)
    if cur is not None and max_hp > 0:
        if int(cur) <= 100 and max_hp > 100:
            cur = max(1, int(round(max_hp * (int(cur) / 100.0))))
        else:
            cur = min(int(cur), max_hp)
    else:
        cur = max_hp
    poke.stats = stats
    poke.cur_hp = cur
    if getattr(belief, 'nature', None):
        poke.nature = belief.nature
    return poke


def _is_spread_move(move: MoveOption) -> bool:
    t = (move.target_type or '').lower().replace('_', '').replace('-', '')
    return t in (
        'alladjacentfoes', 'alladjacent', 'all', 'foes', 'allyside', 'allies',
    )


def _ally_has(ally: Optional[CalcPoke], *abilities: str) -> bool:
    if ally is None:
        return False
    return _norm_abi(ally.ability) in {_norm_abi(a) for a in abilities}


def _ability_on_for(poke: CalcPoke, *, ally: Optional[CalcPoke], foe: Optional[CalcPoke]) -> bool:
    abi = _norm_abi(poke.ability)
    if abi in ('flashfire', 'slowstart', 'electromorphosis'):
        return bool(poke.ability_on)
    if abi in ('plus', 'minus'):
        return _ally_has(ally, 'plus', 'minus')
    if abi == 'stakeout' and foe is not None:
        return bool(foe.just_switched)
    return bool(poke.ability_on)


def _mon_payload(
    poke: CalcPoke,
    *,
    ally: Optional[CalcPoke] = None,
    foe: Optional[CalcPoke] = None,
) -> Dict[str, Any]:
    return _poke_payload(
        poke.species,
        poke.nature,
        _stats_or_default(poke),
        item=poke.item,
        ability=poke.ability,
        status=poke.status,
        boosts=poke.boosts,
        cur_hp=poke.cur_hp,
        ability_on=_ability_on_for(poke, ally=ally, foe=foe),
        types=[t for t in poke.types if t] or None,
        tera_type=poke.tera_type if poke.is_terastallized else None,
        allies_fainted=int(poke.allies_fainted or 0),
    )


def _invert_sp(base: int, final: int, nature: Optional[str], stat: str) -> int:
    """SP Champions que mejor reproduce `final` dado base/naturaleza."""
    from showdown.stat_inference import champions_stat
    best_sp, best_err = 0, 10**9
    for sp in range(0, 64):
        got = champions_stat(base, sp, nature, stat)
        err = abs(got - final)
        if err < best_err:
            best_sp, best_err = sp, err
            if err == 0:
                break
    return best_sp


def _resolve_mega_forme(species: str, item: Optional[str]) -> Optional[str]:
    key = str(species or '').strip().lower().replace(' ', '-')
    if not key or 'mega' in key:
        return None
    item_id = _norm_abi(item)
    # Charizardite Y → charizarditey (no endsWith 'ite'); eviolite excluido.
    if not item_id or item_id == 'eviolite' or 'ite' not in item_id:
        return None
    from API.PokemonData import champions_species_key
    # Preferir X/Y si el item lo indica.
    if item_id.endswith('x') or item_id.endswith('itex'):
        cands = (f'{key}-mega-x', f'{key}-mega', f'{key}-mega-y')
    elif item_id.endswith('y') or item_id.endswith('itey'):
        cands = (f'{key}-mega-y', f'{key}-mega', f'{key}-mega-x')
    else:
        cands = (f'{key}-mega', f'{key}-mega-x', f'{key}-mega-y')
    for cand in cands:
        resolved = champions_species_key(cand)
        if resolved:
            return resolved
    return None


def apply_pending_mega(poke: CalcPoke) -> CalcPoke:
    """
    Remapea stats a la forme mega (mismos SP) para calcs del turno en que
    vamos a megaevolucionar. HP no cambia de base en casi todas las megas.
    """
    mega = _resolve_mega_forme(poke.species, poke.item)
    if not mega:
        return poke
    from showdown.stat_inference import champions_stat, dex_base_stats
    from showdown.battle_modifier import _mega_forme_ability

    base = dex_base_stats(poke.species)
    mega_b = dex_base_stats(mega)
    if not base or not mega_b:
        return poke
    cur = _stats_or_default(poke)
    new_stats: Dict[str, int] = {}
    for stat in ('hp', 'atk', 'def', 'spa', 'spd', 'spe'):
        b0 = int(base.get(stat) or 0)
        b1 = int(mega_b.get(stat) or 0)
        if b0 <= 0 or b1 <= 0:
            new_stats[stat] = int(cur.get(stat) or 1)
            continue
        sp = _invert_sp(b0, int(cur.get(stat) or 1), poke.nature, stat)
        new_stats[stat] = champions_stat(b1, sp, poke.nature, stat)
    mega_abi = _mega_forme_ability(mega) or poke.ability
    types = types_from_species(mega) or poke.types
    return CalcPoke(
        species=mega,
        nature=poke.nature,
        item=poke.item,
        ability=mega_abi or poke.ability,
        status=poke.status,
        boosts=dict(poke.boosts or {}),
        stats=new_stats,
        cur_hp=min(int(poke.cur_hp or new_stats['hp']), new_stats['hp']),
        types=types,
        ability_on=poke.ability_on,
        just_switched=poke.just_switched,
        tera_type=poke.tera_type,
        is_terastallized=poke.is_terastallized,
        is_helping_hand=poke.is_helping_hand,
        is_charged=poke.is_charged,
        is_protected=poke.is_protected,
        has_substitute=poke.has_substitute,
        is_redirecting=poke.is_redirecting,
        allies_fainted=poke.allies_fainted,
        is_two_turn_charging=poke.is_two_turn_charging,
        must_recharge=poke.must_recharge,
    )


def _attacker_side(
    attacker: CalcPoke,
    ally: Optional[CalcPoke],
    *,
    weather: Optional[str],
) -> Dict[str, bool]:
    side: Dict[str, bool] = {}
    if attacker.is_helping_hand:
        side['isHelpingHand'] = True
    if attacker.is_charged:
        side['isCharge'] = True
    if _ally_has(ally, 'battery'):
        side['isBattery'] = True
    if _ally_has(ally, 'powerspot'):
        side['isPowerSpot'] = True
    if _ally_has(ally, 'steelyspirit'):
        side['isSteelySpirit'] = True
    sun = _norm_abi(weather) in ('sunnyday', 'desolateland')
    if sun and (
        _norm_abi(attacker.ability) == 'flowergift' or _ally_has(ally, 'flowergift')
    ):
        side['isFlowerGift'] = True
    return side


def _defender_side(
    defender: CalcPoke,
    def_ally: Optional[CalcPoke],
    ctx: FieldContext,
) -> Dict[str, bool]:
    side: Dict[str, bool] = {}
    if ctx.aurora_veil:
        side['isAuroraVeil'] = True
    else:
        if ctx.reflect:
            side['isReflect'] = True
        if ctx.light_screen:
            side['isLightScreen'] = True
    if _ally_has(def_ally, 'friendguard'):
        side['isFriendGuard'] = True
    return side


def _field_payload(
    ctx: FieldContext,
    attacker: CalcPoke,
    ally: Optional[CalcPoke],
    defender: CalcPoke,
    def_ally: Optional[CalcPoke],
    *,
    is_spread: bool = False,
) -> Dict[str, Any]:
    return {
        'gameType': 'Doubles',
        'weather': _weather_for_calc(ctx.weather),
        'terrain': _terrain_for_calc(ctx.terrain),
        'isWonderRoom': bool(ctx.wonder_room),
        'isMagicRoom': bool(ctx.magic_room),
        'isGravity': bool(ctx.gravity),
        'isSpread': bool(is_spread),
        'isBeadsOfRuin': bool(ctx.is_beads_of_ruin),
        'isSwordOfRuin': bool(ctx.is_sword_of_ruin),
        'isTabletsOfRuin': bool(ctx.is_tablets_of_ruin),
        'isVesselOfRuin': bool(ctx.is_vessel_of_ruin),
        'attackerSide': _attacker_side(attacker, ally, weather=ctx.weather),
        'defenderSide': _defender_side(defender, def_ally, ctx),
    }


def build_calc_request(
    attacker: CalcPoke,
    defender: CalcPoke,
    move_name: str,
    *,
    weather: Optional[str] = None,
    terrain: Optional[str] = None,
    is_spread: bool = False,
    tera_type: Optional[str] = None,
    field_ctx: Optional[FieldContext] = None,
    ally: Optional[CalcPoke] = None,
    def_ally: Optional[CalcPoke] = None,
) -> Dict[str, Any]:
    ctx = field_ctx or FieldContext(weather=weather, terrain=terrain)
    atk_payload = _mon_payload(attacker, ally=ally, foe=defender)
    def_payload = _mon_payload(defender, ally=def_ally, foe=attacker)
    if tera_type:
        atk_payload['teraType'] = tera_type
    return {
        'attacker': atk_payload,
        'defender': def_payload,
        'move': move_name,
        'field': _field_payload(ctx, attacker, ally, defender, def_ally, is_spread=is_spread),
    }


def _is_damaging(move: MoveOption) -> bool:
    return bool(move.name) and move.category.lower() != 'status'


def _other_foe(
    foes: Sequence[Tuple[int, CalcPoke]],
    tid: int,
) -> Optional[CalcPoke]:
    for oid, foe in foes:
        if oid != tid:
            return foe
    return None


def _collect_side_requests(
    side: str,
    attacker: CalcPoke,
    ally: Optional[CalcPoke],
    foes: Sequence[Tuple[int, CalcPoke]],
    moves: Sequence[MoveOption],
    *,
    field_ctx: FieldContext,
    can_tera: bool,
    tera_type: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, MoveOption, int, CalcPoke, int]]]:
    """Arma requests/meta para un lado. meta: (side, move, tid, foe, special)."""
    requests: List[Dict[str, Any]] = []
    meta: List[Tuple[str, MoveOption, int, CalcPoke, int]] = []
    atk_base = _mon_payload(attacker, ally=ally, foe=None)

    for move in moves:
        if not _is_damaging(move):
            continue
        spread = _is_spread_move(move)
        redirectors = [(tid, f) for tid, f in foes if f.is_redirecting]
        targets = foes
        if redirectors and not spread:
            targets = redirectors

        for tid, foe in targets:
            if foe.is_protected:
                continue
            def_ally = _other_foe(foes, tid)
            def_payload = _mon_payload(foe, ally=def_ally, foe=attacker)
            atk_hit = (
                _mon_payload(attacker, ally=ally, foe=foe)
                if _norm_abi(attacker.ability) == 'stakeout'
                else atk_base
            )
            field = _field_payload(
                field_ctx, attacker, ally, foe, def_ally, is_spread=spread,
            )
            requests.append({
                'attacker': atk_hit,
                'defender': def_payload,
                'move': move.name,
                'field': field,
            })
            meta.append((side, move, tid, foe, 0))
            if can_tera and tera_type:
                atk_tera = dict(atk_hit)
                atk_tera['teraType'] = tera_type
                requests.append({
                    'attacker': atk_tera,
                    'defender': def_payload,
                    'move': move.name,
                    'field': field,
                })
                meta.append((side, move, tid, foe, 2))
    return requests, meta


def _collect_ally_hit_requests(
    side: str,
    attacker: CalcPoke,
    ally: Optional[CalcPoke],
    moves: Sequence[MoveOption],
    *,
    field_ctx: FieldContext,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, int]]]:
    """Calc attacker→ally para moves allAdjacent/all. meta: (side, move_slot)."""
    from showdown.battle_bots.move_mechanics import hits_ally

    requests: List[Dict[str, Any]] = []
    meta: List[Tuple[str, int]] = []
    if ally is None:
        return requests, meta
    atk_base = _mon_payload(attacker, ally=None, foe=ally)
    for move in moves:
        if not _is_damaging(move):
            continue
        if not hits_ally(move.target_type, move_name=move.name):
            continue
        def_payload = _mon_payload(ally, ally=None, foe=attacker)
        field = _field_payload(
            field_ctx, attacker, None, ally, None, is_spread=True,
        )
        requests.append({
            'attacker': atk_base,
            'defender': def_payload,
            'move': move.name,
            'field': field,
        })
        meta.append((side, int(move.slot)))
    return requests, meta


def _actions_from_ranges(
    moves: Sequence[MoveOption],
    meta: Sequence[Tuple[str, MoveOption, int, CalcPoke, int]],
    ranges: Sequence[Optional[Tuple[int, int]]],
    *,
    can_mega: bool,
    attacker: Optional[CalcPoke] = None,
    weather: Optional[str] = None,
    terrain: Optional[str] = None,
    gravity: bool = False,
    foes_for_priority: Optional[Sequence[CalcPoke]] = None,
    foes: Optional[Sequence[Tuple[int, CalcPoke]]] = None,
) -> List[ScoredAction]:
    """Convierte ranges→ScoredAction. Tera solo si convierte en KO (base no KO)."""
    from showdown.battle_bots.move_mechanics import (
        accuracy_scale_damage,
        accuracy_scale_ko_bonus,
        accuracy_tiebreak,
        charge_boost_score,
        expected_effect_score,
        gate_damage_range,
        power_herb_cost,
        priority_move_fails,
        recharge_lock_penalty,
        resolve_charge,
        resolve_recharge,
    )

    base_ko: Dict[Tuple[int, int], bool] = {}
    staged: List[Tuple[ScoredAction, int]] = []  # action, special
    item = attacker.item if attacker is not None else None
    already = bool(attacker.is_two_turn_charging) if attacker is not None else False
    recharging = bool(attacker.must_recharge) if attacker is not None else False
    side_foes = list(foes_for_priority or [])

    for (_side, move, tid, foe, special), rng in zip(meta, ranges):
        if str(move.name or '').strip().lower().replace(' ', '').replace('-', '') == 'fakeout':
            from showdown.battle_bots.greedy_solo.type_chart import fake_out_can_affect
            abi = attacker.ability if attacker is not None else None
            if not fake_out_can_affect(abi, foe):
                continue
        if priority_move_fails(
            move.name,
            attacker=attacker,
            defender=foe,
            defender_side=side_foes or [foe],
            terrain=terrain,
            gravity=gravity,
        ):
            continue
        res = resolve_charge(
            move.name, weather=weather, item=item, already_charging=already,
        )
        rech = resolve_recharge(move.name, currently_recharging=recharging)
        gated = gate_damage_range(
            rng, move.name, weather=weather, item=item,
            already_charging=already, must_recharge=recharging,
        )
        max_hp = int((_stats_or_default(foe).get('hp') or 1))
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else max_hp)
        if recharging or (res.is_charge_move and not res.fires_this_turn):
            pct = 0.0
            ko = False
            score = 0.0 if recharging else charge_boost_score(res)
        else:
            raw_pct = damage_pct(gated, max_hp)
            pct = accuracy_scale_damage(raw_pct, move.name)
            raw_ko = guaranteed_ko(gated, cur_hp)
            # Substitute: el golpe rompe/daña el sub, no KO al mon detrás.
            if foe.has_substitute:
                raw_ko = False
            ko = raw_ko
            score = pct + (accuracy_scale_ko_bonus(KO_BONUS, move.name) if raw_ko else 0.0)
            score += expected_effect_score(
                move.name, attacker=attacker, defender=foe, terrain=terrain,
            ).score
            score += accuracy_tiebreak(move.name)
            if res.is_charge_move and res.fires_this_turn:
                # Weather/Herb: el boost de carga ocurre el mismo turno.
                score += charge_boost_score(res)
            score -= power_herb_cost(res)
            score -= recharge_lock_penalty(rech, ko=ko)
        key = (move.slot, tid)
        if special == 0:
            base_ko[key] = ko
            staged.append((ScoredAction(
                score=score,
                move_slot=move.slot,
                target=tid,
                # Mega por defecto cuando está disponible (no solo en KO).
                mega=1 if can_mega else 0,
                damage_pct=pct,
                ko=ko,
            ), 0))
        else:
            if not ko:
                continue
            staged.append((ScoredAction(
                score=score,
                move_slot=move.slot,
                target=tid,
                mega=2,
                damage_pct=pct,
                ko=True,
            ), 2))

    actions: List[ScoredAction] = []
    for action, special in staged:
        if special == 2 and base_ko.get((action.move_slot, action.target), False):
            continue
        actions.append(action)

    for move in moves:
        if not _is_damaging(move):
            res = resolve_charge(
                move.name, weather=weather, item=item, already_charging=already,
            )
            if recharging:
                actions.append(ScoredAction(
                    score=0.0, move_slot=move.slot, target=0,
                    mega=1 if can_mega else 0, damage_pct=0.0, ko=False,
                ))
                continue
            if res.is_charge_move and not res.fires_this_turn:
                actions.append(ScoredAction(
                    score=charge_boost_score(res), move_slot=move.slot, target=0,
                    mega=1 if can_mega else 0, damage_pct=0.0, ko=False,
                ))
                continue
            # Status puro: mejor objetivo entre foes del meta (o `foes`).
            foe_by_tid: Dict[int, Any] = {}
            for _s, _m, ft, foe, _sp in meta:
                foe_by_tid[int(ft)] = foe
            if not foe_by_tid and foes:
                for ft, foe in foes:
                    foe_by_tid[int(ft)] = foe
            if not foe_by_tid and side_foes:
                for i, foe in enumerate(side_foes, start=1):
                    foe_by_tid[i] = foe
            best_tid, best_sc = 0, 0.0
            for ft, foe in foe_by_tid.items():
                sc = expected_effect_score(
                    move.name, attacker=attacker, defender=foe, terrain=terrain,
                ).score
                if sc > best_sc:
                    best_sc = sc
                    best_tid = int(ft)
            score = best_sc - power_herb_cost(res) + accuracy_tiebreak(move.name)
            actions.append(ScoredAction(
                score=score, move_slot=move.slot, target=best_tid,
                mega=1 if can_mega else 0, damage_pct=0.0, ko=False,
            ))
    return actions


async def score_single_hits(
    attacker: CalcPoke,
    foes: Sequence[Tuple[int, CalcPoke]],
    moves: Sequence[MoveOption],
    *,
    weather: Optional[str] = None,
    terrain: Optional[str] = None,
    can_mega: bool = False,
    can_tera: bool = False,
    tera_type: Optional[str] = None,
    field_ctx: Optional[FieldContext] = None,
    ally: Optional[CalcPoke] = None,
) -> List[ScoredAction]:
    """API legacy: un solo lado (usa el batch unificado internamente)."""
    ctx = field_ctx or FieldContext(weather=weather, terrain=terrain)
    requests, meta = _collect_side_requests(
        'L', attacker, ally, foes, moves,
        field_ctx=ctx, can_tera=can_tera, tera_type=tera_type,
    )
    ranges = await calc_damage_batch(requests)
    foe_mons = [f for _, f in foes]
    return _actions_from_ranges(
        moves, meta, ranges, can_mega=can_mega, attacker=attacker,
        weather=ctx.weather, terrain=ctx.terrain, gravity=ctx.gravity,
        foes_for_priority=foe_mons, foes=foes,
    )


async def best_double_action(
    left_attacker: Optional[CalcPoke],
    right_attacker: Optional[CalcPoke],
    left_moves: Sequence[MoveOption],
    right_moves: Sequence[MoveOption],
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    weather: Optional[str] = None,
    terrain: Optional[str] = None,
    can_mega_left: bool = False,
    can_mega_right: bool = False,
    can_tera_left: bool = False,
    can_tera_right: bool = False,
    tera_left: Optional[str] = None,
    tera_right: Optional[str] = None,
    field_ctx: Optional[FieldContext] = None,
) -> Tuple[Optional[ScoredAction], Optional[ScoredAction], float]:
    """
    Un solo round-trip al calc para ambos Pokémon activos.
    Misma política greedy: max por lado + KO bonus + split de foco.
    """
    ctx = field_ctx or FieldContext(weather=weather, terrain=terrain)

    # Si este turno megaevolucionamos, el calc debe usar stats/ability mega.
    left_atk = apply_pending_mega(left_attacker) if (left_attacker and can_mega_left) else left_attacker
    right_atk = apply_pending_mega(right_attacker) if (right_attacker and can_mega_right) else right_attacker

    requests: List[Dict[str, Any]] = []
    meta: List[Tuple[str, MoveOption, int, CalcPoke, int]] = []

    if left_atk and left_moves:
        r, m = _collect_side_requests(
            'L', left_atk, right_atk, foes, left_moves,
            field_ctx=ctx, can_tera=can_tera_left, tera_type=tera_left,
        )
        requests.extend(r)
        meta.extend(m)
    if right_atk and right_moves:
        r, m = _collect_side_requests(
            'R', right_atk, left_atk, foes, right_moves,
            field_ctx=ctx, can_tera=can_tera_right, tera_type=tera_right,
        )
        requests.extend(r)
        meta.extend(m)

    ranges = await calc_damage_batch(requests)

    left_meta = [m for m in meta if m[0] == 'L']
    right_meta = [m for m in meta if m[0] == 'R']
    left_ranges = [rng for m, rng in zip(meta, ranges) if m[0] == 'L']
    right_ranges = [rng for m, rng in zip(meta, ranges) if m[0] == 'R']

    left_actions = _actions_from_ranges(
        left_moves, left_meta, left_ranges, can_mega=can_mega_left,
        attacker=left_atk, weather=ctx.weather, terrain=ctx.terrain,
        gravity=ctx.gravity, foes_for_priority=[f for _, f in foes], foes=foes,
    ) if left_atk and left_moves else []
    right_actions = _actions_from_ranges(
        right_moves, right_meta, right_ranges, can_mega=can_mega_right,
        attacker=right_atk, weather=ctx.weather, terrain=ctx.terrain,
        gravity=ctx.gravity, foes_for_priority=[f for _, f in foes], foes=foes,
    ) if right_atk and right_moves else []

    if not left_actions and not right_actions:
        return None, None, 0.0

    best_l: Optional[ScoredAction] = max(left_actions, key=lambda a: a.score) if left_actions else None
    best_r: Optional[ScoredAction] = max(right_actions, key=lambda a: a.score) if right_actions else None

    if best_l and best_r and best_l.mega and best_r.mega:
        if best_l.score >= best_r.score:
            best_r = ScoredAction(
                score=best_r.score, move_slot=best_r.move_slot, target=best_r.target,
                mega=0, damage_pct=best_r.damage_pct, ko=best_r.ko,
            )
        else:
            best_l = ScoredAction(
                score=best_l.score, move_slot=best_l.move_slot, target=best_l.target,
                mega=0, damage_pct=best_l.damage_pct, ko=best_l.ko,
            )

    if best_l and best_r and best_l.target == best_r.target and len(foes) > 1:
        alt_r = [a for a in right_actions if a.target != best_l.target]
        if alt_r:
            alt = max(alt_r, key=lambda a: a.score)
            if alt.damage_pct >= 1.0 and (alt.score + best_l.score) >= (best_l.score + best_r.score) * 0.85:
                best_r = alt

    total = (best_l.score if best_l else 0.0) + (best_r.score if best_r else 0.0)
    return best_l, best_r, total


def any_positive_damage(actions: Sequence[Optional[ScoredAction]]) -> bool:
    return any(a is not None and a.damage_pct > 0 for a in actions)


def best_forced_switch_slot(
    reserved: Sequence[Tuple[int, Tuple[Optional[str], ...]]],
    foe_types: Sequence[Tuple[Optional[str], ...]],
) -> int:
    if not reserved:
        return 3
    best_slot, best_score = reserved[0][0], float('-inf')
    for slot, types in reserved:
        score = matchup_vs_actives(types, foe_types)
        if score > best_score:
            best_score = score
            best_slot = slot
    return int(best_slot)


@lru_cache(maxsize=512)
def types_from_species(species: str) -> Tuple[Optional[str], ...]:
    from API.PokemonData import getPkmnInfo
    from showdown.battle_bots.greedy_solo.species_names import resolve_species_name
    info = getPkmnInfo(resolve_species_name(species))
    if not info:
        return tuple()
    return pokemon_types(info[0], info[1])


def field_context_from_iap(battle) -> FieldContext:
    """Construye FieldContext desde Battle IAP (bot principal)."""
    weather = getattr(battle, 'weather', None)
    if weather and str(weather).lower() in ('none', 'null', ''):
        weather = None
    terrain = None
    wonder = magic = gravity = False
    for key, val in dict(getattr(battle, 'terrain', None) or {}).items():
        if not val:
            continue
        k = _norm_abi(key)
        if k in ('electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain'):
            terrain = k
        elif k == 'wonderroom':
            wonder = True
        elif k == 'magicroom':
            magic = True
        elif k == 'gravity':
            gravity = True

    opp = getattr(battle, 'opponent', None)
    sc = dict(getattr(opp, 'side_conditions', None) or {}) if opp else {}

    def _sc(name: str) -> bool:
        return int(sc.get(name, 0) or 0) > 0

    actives = []
    for side in (getattr(battle, 'user', None), opp):
        if side is None:
            continue
        for attr in ('active1', 'active2'):
            mon = getattr(side, attr, None)
            if mon is not None and not getattr(mon, 'is_fainted', False):
                actives.append(mon)

    def _has(ab: str) -> bool:
        return any(_norm_abi(getattr(m, 'abilityPkmn', '')) == ab for m in actives)

    return FieldContext(
        weather=str(weather) if weather else None,
        terrain=terrain,
        wonder_room=wonder,
        magic_room=magic,
        gravity=gravity,
        reflect=_sc('reflect'),
        light_screen=_sc('lightscreen'),
        aurora_veil=_sc('auroraveil'),
        is_beads_of_ruin=_has('beadsofruin'),
        is_sword_of_ruin=_has('swordofruin'),
        is_tablets_of_ruin=_has('tabletsofruin'),
        is_vessel_of_ruin=_has('vesselofruin'),
    )
