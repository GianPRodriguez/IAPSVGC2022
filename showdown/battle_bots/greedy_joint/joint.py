"""Greedy-Joint: 1-ply search over A1 x A2 (turn synergy, T=1).

Scope:
  - Cartesian product of valid moves (attack + support)
  - Combined damage per foe + KO_BONUS
  - Redirect (Follow Me / Rage Powder): bonus if partner does high damage
    or guaranteed KO this turn
  - Protect / Fake Out / HH / speed control as 1-ply utility
  - No voluntary switches (force_switch only in main/pokeenv)
  - One mega per pair; horizon T=1 (heuristic, not MCTS)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from showdown.battle_bots.greedy_solo.calc_async import calc_damage_batch, damage_pct, guaranteed_ko
from showdown.battle_bots.greedy_solo.greedy import (
    KO_BONUS,
    CalcPoke,
    FieldContext,
    MoveOption,
    ScoredAction,
    _collect_ally_hit_requests,
    _collect_side_requests,
    _is_damaging,
    _stats_or_default,
    apply_pending_mega,
)

_PROTECT_BONUS = 18.0
_FAKE_OUT_BONUS = 25.0
_HELPING_HAND_BONUS = 20.0
_SPEED_CONTROL_BONUS = 12.0
_STATUS_BASE = 4.0
_FOCUS_SPLIT_BONUS = 8.0
_CHARGE_BASE = 2.0

_REDIRECT_BASE = 8.0
_REDIRECT_WITH_ATTACK = 40.0
_REDIRECT_WITH_KO = 70.0
_HIGH_DAMAGE_PCT = 45.0


def _norm_move(name: str) -> str:
    return str(name or '').strip().lower().replace(' ', '').replace('-', '')


def fake_out_can_affect(
    attacker: Optional[CalcPoke],
    defender: CalcPoke,
) -> bool:
    """Fake Out (Normal) no afecta a Ghost salvo Scrappy / Mind's Eye."""
    from showdown.battle_bots.greedy_solo.type_chart import fake_out_can_affect as _fo

    abi = attacker.ability if attacker is not None else None
    return _fo(abi, defender)


def _utility_kind(
    move: MoveOption,
    *,
    weather: Optional[str] = None,
    item: Optional[str] = None,
    already_charging: bool = False,
) -> str:
    from showdown.battle_bots.move_mechanics import resolve_charge

    mid = _norm_move(move.name)
    res = resolve_charge(
        mid, weather=weather, item=item, already_charging=already_charging,
    )
    if res.is_charge_move and not res.fires_this_turn:
        return 'charge'
    if mid in (
        'protect', 'detect', 'spikyshield', 'banefulbunker', 'silktrap',
        'burningbulwark', 'obstruct', 'kingsshield',
    ):
        return 'protect'
    if mid in ('followme', 'ragepowder', 'spotlight'):
        return 'redirect'
    if mid == 'fakeout':
        return 'fakeout'
    if mid == 'helpinghand':
        return 'helpinghand'
    if mid in ('tailwind', 'trickroom', 'icywind', 'electroweb', 'bulldoze', 'glaciate'):
        return 'speed'
    if not _is_damaging(move):
        return 'status'
    return 'damage'


@dataclass
class JointAtom:
    move_slot: int
    target: int
    mega: int = 0
    damage_pct: Dict[int, float] = field(default_factory=dict)
    max_roll: Dict[int, int] = field(default_factory=dict)
    ko_tids: Set[int] = field(default_factory=set)
    utility: str = 'damage'
    move_name: str = ''
    # Power Herb consume + recharge lock (restado en _score_pair).
    timing_penalty: float = 0.0
    # E[status/secundarios] ya escalado por accuracy.
    effect_bonus: float = 0.0
    # Friendly fire (allAdjacent / all): daño potencial al compañero.
    hits_ally: bool = False
    ally_damage_pct: float = 0.0
    ally_max_roll: int = 0
    ally_ko: bool = False

    @property
    def total_damage_pct(self) -> float:
        return float(sum(self.damage_pct.values()))

    @property
    def has_guaranteed_ko(self) -> bool:
        return bool(self.ko_tids)


def _atoms_from_side(
    moves: Sequence[MoveOption],
    meta: Sequence[Tuple[str, MoveOption, int, CalcPoke, int]],
    ranges: Sequence[Optional[Tuple[int, int]]],
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    can_mega: bool,
    attacker: Optional[CalcPoke] = None,
    weather: Optional[str] = None,
    terrain: Optional[str] = None,
    gravity: bool = False,
    ally: Optional[CalcPoke] = None,
    ally_hit_by_slot: Optional[Dict[int, Optional[Tuple[int, int]]]] = None,
) -> List[JointAtom]:
    from showdown.battle_bots.move_mechanics import (
        accuracy_scale_damage,
        accuracy_tiebreak,
        expected_move_effect_score,
        gate_damage_range,
        hits_ally,
        move_accuracy,
        parse_move_effects,
        power_herb_cost,
        priority_move_fails,
        recharge_lock_penalty,
        resolve_charge,
        resolve_recharge,
    )

    atoms: List[JointAtom] = []
    base_ko: Dict[Tuple[int, int], bool] = {}
    item = attacker.item if attacker is not None else None
    already = bool(attacker.is_two_turn_charging) if attacker is not None else False
    recharging = bool(attacker.must_recharge) if attacker is not None else False
    charge_slots: Set[int] = set()
    foe_mons = [f for _, f in foes]
    ally_hits = ally_hit_by_slot or {}

    def _penalty(move_name: str, *, ko: bool, firing: bool) -> float:
        if not firing:
            return 0.0
        res = resolve_charge(
            move_name, weather=weather, item=item, already_charging=already,
        )
        rech = resolve_recharge(move_name)
        return power_herb_cost(res) + recharge_lock_penalty(rech, ko=ko)

    def _prio_fails(move_name: str, foe: CalcPoke) -> bool:
        return priority_move_fails(
            move_name,
            attacker=attacker,
            defender=foe,
            defender_side=foe_mons,
            terrain=terrain,
            gravity=gravity,
        )

    def _ally_ff(move: MoveOption) -> Tuple[bool, float, int, bool]:
        """hits_ally, damage_pct, max_roll, guaranteed_ko vs partner."""
        does = hits_ally(move.target_type, move_name=move.name)
        if not does or ally is None:
            return does, 0.0, 0, False
        rng = ally_hits.get(int(move.slot))
        gated = gate_damage_range(
            rng, move.name, weather=weather, item=item,
            already_charging=already, must_recharge=recharging,
        )
        if gated is None:
            return True, 0.0, 0, False
        max_hp = int((_stats_or_default(ally).get('hp') or 1))
        cur_hp = int(ally.cur_hp if ally.cur_hp is not None else max_hp)
        raw_pct = damage_pct(gated, max_hp)
        # Guardamos raw; accuracy se aplica en la pena del par.
        max_r = int(gated[1] if gated else 0)
        ko = bool(guaranteed_ko(gated, cur_hp) and not ally.has_substitute)
        return True, float(raw_pct), max_r, ko

    for (_side, move, tid, foe, special), rng in zip(meta, ranges):
        is_fakeout = _norm_move(move.name) == 'fakeout'
        if is_fakeout and not fake_out_can_affect(attacker, foe):
            continue
        if _prio_fails(move.name, foe):
            # Extreme Speed / Fake Out / etc. fallan (Psychic Terrain o Armor Tail…).
            continue
        res = resolve_charge(
            move.name, weather=weather, item=item, already_charging=already,
        )
        gated = gate_damage_range(
            rng, move.name, weather=weather, item=item,
            already_charging=already, must_recharge=recharging,
        )
        max_hp = int((_stats_or_default(foe).get('hp') or 1))
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else max_hp)
        eff = expected_move_effect_score(
            move.name, attacker=attacker, defender=foe, terrain=terrain,
        )
        ff_hit, ff_pct, ff_roll, ff_ko = _ally_ff(move)
        if recharging or (res.is_charge_move and not res.fires_this_turn):
            pct = 0.0
            ko = False
            max_r = 0
            utility = 'charge' if (res.is_charge_move and not res.fires_this_turn and not recharging) else 'status'
            if utility == 'charge':
                charge_slots.add(move.slot)
            pen = 0.0
            effect_b = 0.0
            ff_hit, ff_pct, ff_roll, ff_ko = False, 0.0, 0, False
        else:
            raw_pct = damage_pct(gated, max_hp)
            pct = accuracy_scale_damage(raw_pct, move.name)
            # KO duro solo con acc≈100%; si no, E[KO]=KO_BONUS*acc vía effect_bonus.
            raw_ko = guaranteed_ko(gated, cur_hp) and not foe.has_substitute
            acc = move_accuracy(move.name)
            ko = bool(raw_ko and acc >= 0.99)
            soft_ko = float(KO_BONUS) * acc if (raw_ko and not ko) else 0.0
            max_r = int(round((gated[1] if gated else 0) * acc))
            utility = 'damage'
            pen = _penalty(move.name, ko=raw_ko, firing=True)
            effect_b = float(eff.score) + soft_ko + accuracy_tiebreak(move.name)
        key = (move.slot, tid)
        if special == 0:
            base_ko[key] = ko
            atoms.append(JointAtom(
                move_slot=move.slot,
                target=tid,
                mega=1 if can_mega else 0,
                damage_pct={tid: pct} if pct else {},
                max_roll={tid: max_r} if max_r else {},
                ko_tids={tid} if ko else set(),
                utility=utility,
                move_name=move.name,
                timing_penalty=pen,
                effect_bonus=effect_b,
                hits_ally=ff_hit,
                ally_damage_pct=ff_pct,
                ally_max_roll=ff_roll,
                ally_ko=ff_ko,
            ))
        elif special == 2 and ko and not base_ko.get(key, False):
            atoms.append(JointAtom(
                move_slot=move.slot,
                target=tid,
                mega=2,
                damage_pct={tid: pct},
                max_roll={tid: max_r},
                ko_tids={tid},
                utility='damage',
                move_name=move.name,
                timing_penalty=_penalty(move.name, ko=True, firing=True),
                effect_bonus=effect_b,
                hits_ally=ff_hit,
                ally_damage_pct=ff_pct,
                ally_max_roll=ff_roll,
                ally_ko=ff_ko,
            ))

    hittable_fakeout = [
        tid for tid, foe in foes
        if fake_out_can_affect(attacker, foe) and not _prio_fails('fakeout', foe)
    ]

    for move in moves:
        kind = _utility_kind(
            move, weather=weather, item=item, already_charging=already,
        )
        if kind == 'damage':
            continue
        if kind == 'charge' and move.slot in charge_slots:
            continue
        if kind == 'fakeout' and not hittable_fakeout:
            continue
        if recharging:
            kind = 'status'
        res = resolve_charge(
            move.name, weather=weather, item=item, already_charging=already,
        )
        pen = power_herb_cost(res) if res.fires_this_turn else 0.0

        if kind == 'status':
            # Un átomo por foe vulnerable (Sleep Powder al Grass → score 0, se omite).
            added = False
            for tid, foe in foes:
                if foe.is_protected:
                    continue
                sc = expected_move_effect_score(
                    move.name, attacker=attacker, defender=foe, terrain=terrain,
                ).score
                if sc <= 0.0:
                    continue
                atoms.append(JointAtom(
                    move_slot=move.slot,
                    target=tid,
                    mega=1 if can_mega else 0,
                    damage_pct={},
                    max_roll={},
                    ko_tids=set(),
                    utility='status',
                    move_name=move.name,
                    timing_penalty=pen,
                    effect_bonus=sc + accuracy_tiebreak(move.name),
                ))
                added = True
            if not added:
                # Setup / heal / pivot / campo sin target foe, o score 0 en foes.
                feats = parse_move_effects(move.name)
                sc_self = expected_move_effect_score(
                    move.name, attacker=attacker, defender=None, terrain=terrain,
                ).score
                if sc_self <= _STATUS_BASE * 0.5 and not (
                    feats.self_boosts or feats.heal_frac or feats.self_switch
                    or feats.weather or feats.terrain or feats.side_condition
                    or feats.volatile_status or feats.hp_cost_frac
                    or feats.foe_boosts
                ):
                    sc_self = _STATUS_BASE * move_accuracy(move.name)
                atoms.append(JointAtom(
                    move_slot=move.slot,
                    target=0,
                    mega=1 if can_mega else 0,
                    utility='status',
                    move_name=move.name,
                    timing_penalty=pen,
                    effect_bonus=sc_self + accuracy_tiebreak(move.name),
                ))
            continue

        target = hittable_fakeout[0] if kind == 'fakeout' else 0
        atoms.append(JointAtom(
            move_slot=move.slot,
            target=target,
            mega=1 if can_mega else 0,
            damage_pct={},
            max_roll={},
            ko_tids=set(),
            utility=kind,
            move_name=move.name,
            timing_penalty=pen,
        ))
    return atoms


def _charge_utility(atom: JointAtom) -> float:
    from showdown.battle_bots.move_mechanics import charge_boost_score, resolve_charge

    if atom.utility != 'charge':
        return 0.0
    res = resolve_charge(atom.move_name)
    return _CHARGE_BASE + charge_boost_score(res)


def _redirect_synergy(
    redirect: JointAtom,
    partner: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
) -> float:
    """Follow Me / Rage Powder: 1-ply support if partner hits hard or KOs."""
    if redirect.utility != 'redirect':
        return 0.0
    if partner.has_guaranteed_ko:
        return _REDIRECT_WITH_KO
    for tid, foe in foes:
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else 1)
        if cur_hp > 0 and partner.max_roll.get(tid, 0) >= cur_hp and not foe.has_substitute:
            return _REDIRECT_WITH_KO
    if partner.total_damage_pct >= _HIGH_DAMAGE_PCT:
        return _REDIRECT_WITH_ATTACK
    if partner.total_damage_pct > 0:
        return _REDIRECT_BASE + min(partner.total_damage_pct * 0.25, 20.0)
    return _REDIRECT_BASE


def _score_pair(
    left: JointAtom,
    right: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    left_poke: Optional[CalcPoke] = None,
    right_poke: Optional[CalcPoke] = None,
    include_friendly_fire: bool = True,
) -> float:
    from showdown.battle_bots.move_mechanics import friendly_fire_pair_penalty

    if not foes:
        return _utility_score(left, right, foes=foes, has_damage=False)

    combined_pct: Dict[int, float] = {tid: 0.0 for tid, _ in foes}
    combined_roll: Dict[int, int] = {tid: 0 for tid, _ in foes}
    for atom in (left, right):
        for tid, pct in atom.damage_pct.items():
            if tid in combined_pct:
                combined_pct[tid] += pct
                combined_roll[tid] += int(atom.max_roll.get(tid, 0))

    score = 0.0
    targets_hit = 0
    for tid, foe in foes:
        max_hp = int((_stats_or_default(foe).get('hp') or 1))
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else max_hp)
        pct = combined_pct.get(tid, 0.0)
        remain_pct = max(0.0, 100.0 * cur_hp / max(max_hp, 1))
        effective = min(pct, remain_pct) if remain_pct > 0 else 0.0
        score += effective
        if effective >= 1.0:
            targets_hit += 1
        if combined_roll.get(tid, 0) >= cur_hp and cur_hp > 0 and not foe.has_substitute:
            score += KO_BONUS

    if targets_hit >= 2:
        score += _FOCUS_SPLIT_BONUS

    has_damage = (left.total_damage_pct + right.total_damage_pct) > 0
    score += _utility_score(left, right, foes=foes, has_damage=has_damage)
    score += float(left.effect_bonus) + float(right.effect_bonus)
    score -= float(left.timing_penalty) + float(right.timing_penalty)
    if include_friendly_fire:
        score -= friendly_fire_pair_penalty(
            left, right, left_poke, right_poke,
        )
    return score


def _utility_score(
    left: JointAtom,
    right: JointAtom,
    *,
    foes: Sequence[Tuple[int, CalcPoke]],
    has_damage: bool,
) -> float:
    s = 0.0
    for atom in (left, right):
        if atom.utility == 'protect':
            s += _PROTECT_BONUS if has_damage else _STATUS_BASE
        elif atom.utility == 'redirect':
            s += _REDIRECT_BASE if not has_damage else 0.0
        elif atom.utility == 'fakeout':
            # Átomos 'fakeout' solo se crean si hay target no inmune (ver _atoms_from_side).
            s += _FAKE_OUT_BONUS
        elif atom.utility == 'helpinghand':
            s += _HELPING_HAND_BONUS if has_damage else _STATUS_BASE
        elif atom.utility == 'speed':
            s += _SPEED_CONTROL_BONUS
        elif atom.utility == 'charge':
            s += _charge_utility(atom)
        elif atom.utility == 'status':
            # effect_bonus ya trae E[status]*acc; fallback mínimo si vino vacío.
            if atom.effect_bonus <= 0:
                s += _STATUS_BASE

    if left.utility == 'protect' and right.total_damage_pct > 0:
        s += 6.0
    if right.utility == 'protect' and left.total_damage_pct > 0:
        s += 6.0
    if left.utility == 'helpinghand' and right.total_damage_pct > 0:
        s += 8.0
    if right.utility == 'helpinghand' and left.total_damage_pct > 0:
        s += 8.0

    if left.utility == 'redirect':
        s += _redirect_synergy(left, right, foes)
    if right.utility == 'redirect':
        s += _redirect_synergy(right, left, foes)
    return s


def _atom_to_scored(atom: JointAtom) -> ScoredAction:
    tid = atom.target
    pct = float(atom.damage_pct.get(tid, atom.total_damage_pct))
    return ScoredAction(
        score=pct,
        move_slot=atom.move_slot,
        target=atom.target,
        mega=atom.mega,
        damage_pct=pct,
        ko=atom.has_guaranteed_ko,
    )


def _strip_mega(atom: JointAtom) -> JointAtom:
    return JointAtom(
        move_slot=atom.move_slot,
        target=atom.target,
        mega=0,
        damage_pct=dict(atom.damage_pct),
        max_roll=dict(atom.max_roll),
        ko_tids=set(atom.ko_tids),
        utility=atom.utility,
        move_name=atom.move_name,
        timing_penalty=atom.timing_penalty,
        effect_bonus=atom.effect_bonus,
        hits_ally=atom.hits_ally,
        ally_damage_pct=atom.ally_damage_pct,
        ally_max_roll=atom.ally_max_roll,
        ally_ko=atom.ally_ko,
    )


async def best_joint_double_action(
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
    """1-ply joint argmax over A1 x A2 (moves only; no voluntary switches)."""
    ctx = field_ctx or FieldContext(weather=weather, terrain=terrain)
    left_atk = apply_pending_mega(left_attacker) if (left_attacker and can_mega_left) else left_attacker
    right_atk = apply_pending_mega(right_attacker) if (right_attacker and can_mega_right) else right_attacker

    requests: List[dict] = []
    meta: List[Tuple[str, MoveOption, int, CalcPoke, int]] = []
    ally_meta: List[Tuple[str, int]] = []

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
    # Ally-hit calcs al final para que ranges[:len(meta)] = foes.
    if left_atk and left_moves:
        ar, am = _collect_ally_hit_requests(
            'L', left_atk, right_atk, left_moves, field_ctx=ctx,
        )
        requests.extend(ar)
        ally_meta.extend(am)
    if right_atk and right_moves:
        ar, am = _collect_ally_hit_requests(
            'R', right_atk, left_atk, right_moves, field_ctx=ctx,
        )
        requests.extend(ar)
        ally_meta.extend(am)

    ranges = await calc_damage_batch(requests) if requests else []

    foe_count = len(meta)
    foe_ranges = ranges[:foe_count]
    ally_ranges = ranges[foe_count:]
    ally_hit_l: Dict[int, Optional[Tuple[int, int]]] = {}
    ally_hit_r: Dict[int, Optional[Tuple[int, int]]] = {}
    for (side, slot), rng in zip(ally_meta, ally_ranges):
        if side == 'L':
            ally_hit_l[int(slot)] = rng
        else:
            ally_hit_r[int(slot)] = rng

    left_meta = [m for m in meta if m[0] == 'L']
    right_meta = [m for m in meta if m[0] == 'R']
    left_ranges = [rng for m, rng in zip(meta, foe_ranges) if m[0] == 'L']
    right_ranges = [rng for m, rng in zip(meta, foe_ranges) if m[0] == 'R']

    left_atoms = (
        _atoms_from_side(
            left_moves, left_meta, left_ranges, foes,
            can_mega=can_mega_left, attacker=left_atk,
            weather=ctx.weather, terrain=ctx.terrain, gravity=ctx.gravity,
            ally=right_atk, ally_hit_by_slot=ally_hit_l,
        )
        if left_atk and left_moves else []
    )
    right_atoms = (
        _atoms_from_side(
            right_moves, right_meta, right_ranges, foes,
            can_mega=can_mega_right, attacker=right_atk,
            weather=ctx.weather, terrain=ctx.terrain, gravity=ctx.gravity,
            ally=left_atk, ally_hit_by_slot=ally_hit_r,
        )
        if right_atk and right_moves else []
    )

    if not left_atoms and not right_atoms:
        return None, None, 0.0
    if not left_atoms:
        best_r = max(
            right_atoms,
            key=lambda a: a.total_damage_pct
            + (_FAKE_OUT_BONUS if a.utility == 'fakeout' else 0)
            + (_REDIRECT_BASE if a.utility == 'redirect' else 0),
        )
        return None, _atom_to_scored(best_r), best_r.total_damage_pct
    if not right_atoms:
        best_l = max(
            left_atoms,
            key=lambda a: a.total_damage_pct
            + (_FAKE_OUT_BONUS if a.utility == 'fakeout' else 0)
            + (_REDIRECT_BASE if a.utility == 'redirect' else 0),
        )
        return _atom_to_scored(best_l), None, best_l.total_damage_pct

    best_score = float('-inf')
    best_pair: Optional[Tuple[JointAtom, JointAtom]] = None
    for a1 in left_atoms:
        for a2 in right_atoms:
            pair_l, pair_r = a1, a2
            if a1.mega and a2.mega:
                if a1.total_damage_pct >= a2.total_damage_pct:
                    pair_r = _strip_mega(a2)
                else:
                    pair_l = _strip_mega(a1)
            s = _score_pair(
                pair_l, pair_r, foes,
                left_poke=left_atk, right_poke=right_atk,
            )
            if s > best_score:
                best_score = s
                best_pair = (pair_l, pair_r)

    assert best_pair is not None
    bl, br = best_pair
    return _atom_to_scored(bl), _atom_to_scored(br), best_score
