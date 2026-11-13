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
from showdown.battle_bots.greedy_joint.diag import classify_pair, note_turn_kind

# =============================================================================
# Greedy-Joint — A + A2 + A3 + A4 (B off; sin switch voluntario)
# -----------------------------------------------------------------------------
# A (_DAMAGE_ISO): U ≈ S_L+S_R; KO conjunto solo si nadie solo remata.
# B (_SPEED_AWARE): off.
# A2: clip % mismo foe; joint-KO synergy; Protect HP bajo.
# A3 (_SYNERGY_LITE): FO/HH/redirect/speed condicionados al partner.
# A4 (_OPP_THREAT + _MECH_SYNERGY): amenaza rival 1-ply (tipos) → Protect/
#   redirect; HH ×1.5 daño partner; Fake Out = daño + deny por amenaza.
# A5 (_RESIDUAL_SECOND): tras KO de ≥1 foe, bonus por daño/KO al otro vivo.
# B-lite (_FOCUS_SPEED): Spe/prio solo al apilar focus (no _SPEED_AWARE global).
# Campo Spe: TR/TW activos en orden 1-ply.
# A9 sinergia: (1) setup TW/TR condicional; (2) prio-win FO/ES/Prankster;
#   (3) merge spread + bonus split si focus no remata.
# Spe carrera: belief Spe abierto → PE (`_SPE_OPEN_POINT_ESTIMATE` en greedy_solo);
#   Atk/SpA/bulk siguen max. Afecta B-lite / A9 / campo vía apply_belief_stats.
# A6/A7/A8: off (N=100).
# Rollback Spe PE: greedy_solo._SPE_OPEN_POINT_ESTIMATE=False;
#   A9: _FIELD_SETUP/_PRIO_WIN_GATE/_SPREAD_SYNERGY=False → A5+B-lite+field Spe;
#   _FOCUS_SPEED=False → A5; _RESIDUAL_SECOND=False → A4; _DAMAGE_ISO=False → v_stable.
# Doc: docs/tesis_agente_greedy_joint.md
# =============================================================================
_DAMAGE_ISO = True
_SPEED_AWARE = False
_FOCUS_SPEED = True
_FOCUS_WASTE = False
_FOCUS_WASTE_PENALTY = 45.0
_SPE_RACE_GATE = False
_FO_PARTNER_PRESSURE = False
_FO_PARTNER_KO_BONUS = 28.0
_FO_PARTNER_PRESS_BONUS = 14.0
_FO_PARTNER_SPLIT_BONUS = 10.0
_FO_PARTNER_PRESS_PCT = 30.0

# A9 — sinergia residual 1-ply (sobre A5+B-lite+campo)
_FIELD_SETUP = True
_FIELD_SETUP_BONUS = 10.0
_FIELD_SETUP_WEAK = 2.0
_PRIO_WIN_GATE = True
_PRIO_WIN_SCALE_MISS = 0.25
_PRIO_TEMPO_BONUS = 8.0
_SPREAD_SYNERGY = True
_SPREAD_SPLIT_BONUS = 14.0

_PROTECT_BONUS = 8.0
_FAKE_OUT_BONUS = 12.0
_HELPING_HAND_BONUS = 10.0
_SPEED_CONTROL_BONUS = 6.0
_STATUS_BASE = 2.0
_FOCUS_SPLIT_BONUS = 0.0
_CHARGE_BASE = 2.0

_REDIRECT_BASE = 4.0
_REDIRECT_WITH_ATTACK = 15.0
_REDIRECT_WITH_KO = 25.0
_HIGH_DAMAGE_PCT = 55.0

# Tune A2
_FOCUS_CLIP = True
_JOINT_KO_SYNERGY = 25.0
_PROTECT_REACT_BONUS = 6.0
_PROTECT_REACT_HP_PCT = 40.0

# Tune A3 — sinergia lite (bonos planos condicionados)
_SYNERGY_LITE = True
_FO_PARTNER_BONUS = 8.0
_HH_PARTNER_BONUS = 7.0
_SPEED_PARTNER_BONUS = 4.0
_SCREEN_PARTNER_BONUS = 8.0  # Reflect/LS Screen/Aurora Veil + partner daño
_REDIRECT_LITE_SCALE = 0.55

# Tune A4 — amenaza + sinergia mecánica (sin switch)
_OPP_THREAT = True
_MECH_SYNERGY = True
_HH_DAMAGE_MULT = 1.5
_PROTECT_THREAT_PCT = 55.0
_PROTECT_THREAT_BONUS = 10.0
_REDIRECT_SAVE_THREAT_PCT = 55.0
_REDIRECT_SAVE_BONUS = 12.0
_THREAT_BASE_PCT = 35.0

# Tune A5 — residual al segundo foe tras asegurar KO
_RESIDUAL_SECOND = True
_RESIDUAL_SCALE = 0.35
_RESIDUAL_CAP = 28.0
_RESIDUAL_SECOND_KO = 12.0

# Sinergias fijas en _utility_score (v_stable; ignoradas si _DAMAGE_ISO):
#   Protect+partner dmg: +3 ; Helping Hand+partner dmg: +4


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
    threat_by_tid: Optional[Dict[int, float]] = None,
    trick_room: bool = False,
    our_tailwind: bool = False,
    opp_tailwind: bool = False,
) -> List[JointAtom]:
    from showdown.battle_bots.move_mechanics import (
        accuracy_scale_damage,
        accuracy_scale_ko_bonus,
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
    from showdown.battle_bots.move_mechanics.vgc_utility import fake_out_deny_from_threat

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
            # KO duro (tera/flag) solo con acc≈100%. E[KO] imperfecto:
            # accuracy_scale_ko_bonus en effect (como solo); el par no lo re-suma.
            raw_ko = guaranteed_ko(gated, cur_hp) and not foe.has_substitute
            acc = move_accuracy(move.name)
            ko = bool(raw_ko and acc >= 0.99)
            max_r = int(round((gated[1] if gated else 0) * acc))
            utility = 'damage'
            pen = _penalty(move.name, ko=raw_ko, firing=True)
            effect_b = float(eff.score) + accuracy_tiebreak(move.name)
            if raw_ko and not ko:
                effect_b += accuracy_scale_ko_bonus(KO_BONUS, move.name)
            if _MECH_SYNERGY and is_fakeout:
                deny = fake_out_deny_from_threat(
                    attacker=attacker,
                    threat_denied=float((threat_by_tid or {}).get(int(tid), 0.0)),
                )
                deny *= _prio_win_scale(
                    move.name,
                    attacker=attacker,
                    foe=foe,
                    terrain=terrain,
                    trick_room=trick_room,
                    our_tailwind=our_tailwind,
                    opp_tailwind=opp_tailwind,
                )
                effect_b += deny
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
        # A4: Fake Out se valora en el átomo de daño (+ deny); no duplicar utility vacío.
        if kind == 'fakeout' and _MECH_SYNERGY:
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
                # A9.2: Prankster/status prio solo cuenta del todo si gana el turno.
                sc *= _prio_win_scale(
                    move.name,
                    attacker=attacker,
                    foe=foe,
                    terrain=terrain,
                    trick_room=trick_room,
                    our_tailwind=our_tailwind,
                    opp_tailwind=opp_tailwind,
                )
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
                threat_foe = None
                living = [
                    (int(tid), f) for tid, f in foes
                    if int(f.cur_hp if f.cur_hp is not None else 1) > 0
                ]
                if living:
                    threat_foe = max(
                        living,
                        key=lambda tf: float((threat_by_tid or {}).get(int(tf[0]), 0.0)),
                    )[1]
                sc_self *= _prio_win_scale(
                    move.name,
                    attacker=attacker,
                    foe=threat_foe,
                    terrain=terrain,
                    trick_room=trick_room,
                    our_tailwind=our_tailwind,
                    opp_tailwind=opp_tailwind,
                )
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
    if _SPREAD_SYNERGY:
        return _merge_spread_damage_atoms(atoms, moves)
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


def _atom_alone_kos(atom: JointAtom, tid: int, cur_hp: int) -> bool:
    if cur_hp <= 0:
        return False
    if tid in atom.ko_tids:
        return True
    return int(atom.max_roll.get(tid, 0)) >= cur_hp


def _hp_pct(poke: Optional[CalcPoke]) -> float:
    if poke is None:
        return 100.0
    max_hp = int((_stats_or_default(poke).get('hp') or 1))
    cur = int(poke.cur_hp if poke.cur_hp is not None else max_hp)
    return 100.0 * max(0, cur) / max(max_hp, 1)


def _pairwise_type_threat(foe: CalcPoke, our: CalcPoke) -> float:
    """Amenaza 1-ply aproximada (tipos STAB foe → our), sin calc Node."""
    from showdown.battle_bots.greedy_solo.type_chart import type_effectiveness

    our_types = tuple(our.types or ())
    best = 1.0
    for atk_t in (foe.types or ()):
        if not atk_t:
            continue
        m = float(type_effectiveness(atk_t, our_types))
        if m > best:
            best = m
    est = float(_THREAT_BASE_PCT) * best
    hp = _hp_pct(our)
    if hp < 100.0:
        est *= 100.0 / max(hp, 25.0)
    return float(min(est, 120.0))


def _side_incoming_threat(
    our: Optional[CalcPoke],
    foes: Sequence[Tuple[int, CalcPoke]],
) -> float:
    if our is None or not foes:
        return 0.0
    return max((_pairwise_type_threat(foe, our) for _tid, foe in foes), default=0.0)


def _foe_threat_by_tid(
    left: Optional[CalcPoke],
    right: Optional[CalcPoke],
    foes: Sequence[Tuple[int, CalcPoke]],
) -> Dict[int, float]:
    """Amenaza que cada foe activo supone al bando propio (suma L+R)."""
    out: Dict[int, float] = {}
    for tid, foe in foes:
        t = 0.0
        if left is not None:
            t += _pairwise_type_threat(foe, left)
        if right is not None:
            t += _pairwise_type_threat(foe, right)
        out[int(tid)] = float(t)
    return out


def _pair_koed_tids(
    left: JointAtom,
    right: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    mult_l: float = 1.0,
    mult_r: float = 1.0,
    alone_ko: Optional[Dict[int, bool]] = None,
) -> Set[int]:
    """Foes que el par remata este turno (solo o conjunto)."""
    koed: Set[int] = set()
    for tid, foe in foes:
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else 1)
        if cur_hp <= 0 or foe.has_substitute:
            continue
        if alone_ko is not None and alone_ko.get(tid):
            koed.add(int(tid))
            continue
        l_roll = int(left.max_roll.get(tid, 0) * mult_l)
        r_roll = int(right.max_roll.get(tid, 0) * mult_r)
        if (
            tid in left.ko_tids
            or tid in right.ko_tids
            or l_roll >= cur_hp
            or r_roll >= cur_hp
            or (l_roll + r_roll) >= cur_hp
        ):
            koed.add(int(tid))
    return koed


def _residual_second_bonus(
    left: JointAtom,
    right: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    koed: Set[int],
    mult_l: float = 1.0,
    mult_r: float = 1.0,
) -> float:
    """A5: sinergia extra si tras KO de ≥1 foe el par aún daña/KO al otro vivo."""
    if not _RESIDUAL_SECOND or not koed:
        return 0.0
    residual_pct = 0.0
    second_ko = False
    for tid, foe in foes:
        if int(tid) in koed:
            continue
        max_hp = int((_stats_or_default(foe).get('hp') or 1))
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else max_hp)
        if cur_hp <= 0 or foe.has_substitute:
            continue
        l_pct = float(left.damage_pct.get(tid, 0.0)) * mult_l
        r_pct = float(right.damage_pct.get(tid, 0.0)) * mult_r
        if l_pct <= 0.0 and r_pct <= 0.0 and tid not in left.ko_tids and tid not in right.ko_tids:
            continue
        remain_pct = 100.0 * cur_hp / max(max_hp, 1)
        if _FOCUS_CLIP and l_pct > 0.0 and r_pct > 0.0:
            residual_pct += min(l_pct + r_pct, remain_pct)
        else:
            residual_pct += l_pct + r_pct
        l_roll = int(left.max_roll.get(tid, 0) * mult_l)
        r_roll = int(right.max_roll.get(tid, 0) * mult_r)
        if (
            tid in left.ko_tids
            or tid in right.ko_tids
            or l_roll >= cur_hp
            or r_roll >= cur_hp
            or (l_roll + r_roll) >= cur_hp
        ):
            second_ko = True
    if residual_pct <= 0.0 and not second_ko:
        return 0.0
    bonus = min(float(_RESIDUAL_SCALE) * residual_pct, float(_RESIDUAL_CAP))
    if second_ko:
        bonus += float(_RESIDUAL_SECOND_KO)
    return float(bonus)


def _focus_waste_penalty(
    left: JointAtom,
    right: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    alone_ko: Dict[int, bool],
    mult_l: float = 1.0,
    mult_r: float = 1.0,
) -> float:
    """A6: castigo si uno ya remata solo y el partner también pega al mismo foe."""
    if not _FOCUS_WASTE:
        return 0.0
    for tid, foe in foes:
        if not alone_ko.get(tid):
            continue
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else 1)
        if cur_hp <= 0 or foe.has_substitute:
            continue
        l_roll = int(left.max_roll.get(tid, 0) * mult_l)
        r_roll = int(right.max_roll.get(tid, 0) * mult_r)
        left_alone = tid in left.ko_tids or l_roll >= cur_hp
        right_alone = tid in right.ko_tids or r_roll >= cur_hp
        # Ambos pueden rematar solos → B-lite/Spe ordena; no es “waste” A6.
        if left_alone and right_alone:
            continue
        l_hits = (
            float(left.damage_pct.get(tid, 0.0)) * mult_l > 0.0
            or tid in left.ko_tids
            or l_roll > 0
        )
        r_hits = (
            float(right.damage_pct.get(tid, 0.0)) * mult_r > 0.0
            or tid in right.ko_tids
            or r_roll > 0
        )
        if (left_alone and r_hits) or (right_alone and l_hits):
            return float(_FOCUS_WASTE_PENALTY)
    return 0.0


def _foe_acts_before_us(
    our: Optional[CalcPoke],
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    terrain: Optional[str] = None,
    trick_room: bool = False,
    our_tailwind: bool = False,
    opp_tailwind: bool = False,
) -> bool:
    """True si algún foe vivo actuaría antes que `our` (ataque prio 0 vs prio 0)."""
    if our is None or not foes:
        return False
    from showdown.battle_bots.expectimax_bayesian.speed_order import (
        acts_before,
        poke_effective_spe,
    )

    spe_our = poke_effective_spe(our, tailwind=our_tailwind)
    for _tid, foe in foes:
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else 1)
        if cur_hp <= 0 or foe.has_substitute:
            continue
        spe_foe = poke_effective_spe(foe, tailwind=opp_tailwind)
        if acts_before(spe_foe, 0, spe_our, 0, trick_room=trick_room):
            return True
    return False


def _is_fakeout_atom(atom: JointAtom) -> bool:
    return _norm_move(atom.move_name) == 'fakeout'


def _living_foes(
    foes: Sequence[Tuple[int, CalcPoke]],
) -> List[Tuple[int, CalcPoke]]:
    out: List[Tuple[int, CalcPoke]] = []
    for tid, foe in foes:
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else 1)
        if cur_hp > 0:
            out.append((int(tid), foe))
    return out


def _mean_effective_spe(
    pokes: Sequence[Optional[CalcPoke]],
    *,
    tailwind: bool = False,
) -> float:
    from showdown.battle_bots.expectimax_bayesian.speed_order import poke_effective_spe

    vals = [
        float(poke_effective_spe(p, tailwind=tailwind))
        for p in pokes
        if p is not None
    ]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _move_acts_before_foe(
    move_name: str,
    *,
    attacker: Optional[CalcPoke],
    foe: CalcPoke,
    terrain: Optional[str] = None,
    trick_room: bool = False,
    our_tailwind: bool = False,
    opp_tailwind: bool = False,
) -> bool:
    """True si nuestro move (prio+Spe) va antes que un ataque prio-0 del foe."""
    if attacker is None:
        return True
    from showdown.battle_bots.expectimax_bayesian.speed_order import (
        acts_before,
        move_prio,
        poke_effective_spe,
    )

    spe_a = poke_effective_spe(attacker, tailwind=our_tailwind)
    spe_f = poke_effective_spe(foe, tailwind=opp_tailwind)
    p_a = move_prio(move_name, ability=attacker.ability, terrain=terrain)
    return bool(acts_before(spe_a, p_a, spe_f, 0, trick_room=trick_room))


def _prio_win_scale(
    move_name: str,
    *,
    attacker: Optional[CalcPoke],
    foe: Optional[CalcPoke],
    terrain: Optional[str] = None,
    trick_room: bool = False,
    our_tailwind: bool = False,
    opp_tailwind: bool = False,
) -> float:
    """1.0 si prio≤0 o ganamos la carrera; si no, escala baja (gate A9)."""
    if not _PRIO_WIN_GATE or foe is None or attacker is None:
        return 1.0
    from showdown.battle_bots.expectimax_bayesian.speed_order import move_prio

    if move_prio(move_name, ability=attacker.ability, terrain=terrain) <= 0:
        return 1.0
    if _move_acts_before_foe(
        move_name,
        attacker=attacker,
        foe=foe,
        terrain=terrain,
        trick_room=trick_room,
        our_tailwind=our_tailwind,
        opp_tailwind=opp_tailwind,
    ):
        return 1.0
    return float(_PRIO_WIN_SCALE_MISS)


def _would_outspeed_at_prio0(
    our: Optional[CalcPoke],
    foe: CalcPoke,
    *,
    trick_room: bool = False,
    our_tailwind: bool = False,
    opp_tailwind: bool = False,
) -> bool:
    if our is None:
        return False
    from showdown.battle_bots.expectimax_bayesian.speed_order import (
        acts_before,
        poke_effective_spe,
    )

    spe_o = poke_effective_spe(our, tailwind=our_tailwind)
    spe_f = poke_effective_spe(foe, tailwind=opp_tailwind)
    return bool(acts_before(spe_o, 0, spe_f, 0, trick_room=trick_room))


def _field_setup_bonus(
    atom: JointAtom,
    partner: JointAtom,
    *,
    left_poke: Optional[CalcPoke],
    right_poke: Optional[CalcPoke],
    foes: Sequence[Tuple[int, CalcPoke]],
    trick_room: bool,
    our_tailwind: bool,
    opp_tailwind: bool,
) -> float:
    """A9.1: TW/TR solo valen si el campo aún no está y Spe lo pide; else Icy Wind…"""
    if partner.total_damage_pct <= 0.0 and not partner.has_guaranteed_ko:
        return 0.0
    mid = _norm_move(atom.move_name)
    if mid not in ('tailwind', 'trickroom'):
        return float(_SPEED_PARTNER_BONUS)
    if not _FIELD_SETUP:
        return float(_SPEED_PARTNER_BONUS)

    living = [f for _tid, f in _living_foes(foes)]
    our_spe = _mean_effective_spe(
        (left_poke, right_poke), tailwind=our_tailwind,
    )
    foe_spe = _mean_effective_spe(living, tailwind=opp_tailwind)

    if mid == 'tailwind':
        if our_tailwind:
            return 0.0
        # TW útil si aún no outspeed-eamos (media).
        if our_spe < foe_spe:
            return float(_FIELD_SETUP_BONUS)
        return float(_FIELD_SETUP_WEAK)
    # trickroom
    if trick_room:
        return 0.0
    if our_spe < foe_spe:
        return float(_FIELD_SETUP_BONUS)
    return float(_FIELD_SETUP_WEAK)


def _spread_split_bonus(
    left: JointAtom,
    right: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    alone_ko: Dict[int, bool],
) -> float:
    """A9.3: spread que pega a ≥2 vivos si nadie remata solo (focus no cierra)."""
    if not _SPREAD_SYNERGY:
        return 0.0
    living = _living_foes(foes)
    if len(living) < 2:
        return 0.0
    if any(alone_ko.get(int(tid), False) for tid, _ in living):
        return 0.0
    living_ids = {tid for tid, _ in living}
    for atom in (left, right):
        hits = sum(
            1 for tid in living_ids
            if float(atom.damage_pct.get(tid, 0.0)) > 0.0 or tid in atom.ko_tids
        )
        if hits >= 2:
            return float(_SPREAD_SPLIT_BONUS)
    return 0.0


def _prio_tempo_bonus(
    left: JointAtom,
    right: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    left_poke: Optional[CalcPoke],
    right_poke: Optional[CalcPoke],
    terrain: Optional[str],
    trick_room: bool,
    our_tailwind: bool,
    opp_tailwind: bool,
) -> float:
    """A9.2: prio dañino (p.ej. Extreme Speed) si gana carrera Spe que prio0 pierde."""
    if not _PRIO_WIN_GATE:
        return 0.0
    from showdown.battle_bots.expectimax_bayesian.speed_order import move_prio

    best = 0.0
    for atom, poke, partner in (
        (left, left_poke, right),
        (right, right_poke, left),
    ):
        if poke is None:
            continue
        if atom.utility not in ('damage', 'fakeout'):
            continue
        if _is_fakeout_atom(atom):
            continue  # FO ya tiene deny gateado
        prio = move_prio(atom.move_name, ability=poke.ability, terrain=terrain)
        if prio <= 0:
            continue
        if partner.total_damage_pct <= 0.0 and not partner.has_guaranteed_ko:
            continue
        for tid, foe in _living_foes(foes):
            if float(atom.damage_pct.get(tid, 0.0)) <= 0.0 and tid not in atom.ko_tids:
                continue
            if not _move_acts_before_foe(
                atom.move_name,
                attacker=poke,
                foe=foe,
                terrain=terrain,
                trick_room=trick_room,
                our_tailwind=our_tailwind,
                opp_tailwind=opp_tailwind,
            ):
                continue
            # Solo premio tempo si sin prio perderíamos la Spe race.
            if _would_outspeed_at_prio0(
                poke, foe,
                trick_room=trick_room,
                our_tailwind=our_tailwind,
                opp_tailwind=opp_tailwind,
            ):
                continue
            chunk = float(_PRIO_TEMPO_BONUS)
            if tid in atom.ko_tids or int(atom.max_roll.get(tid, 0)) >= int(
                foe.cur_hp if foe.cur_hp is not None else 1
            ):
                chunk *= 1.25
            best = max(best, chunk)
    return float(best)


def _merge_spread_damage_atoms(
    atoms: List[JointAtom],
    moves: Sequence[MoveOption],
) -> List[JointAtom]:
    """Un átomo por (slot, mega) spread con daño en todos los foes golpeados."""
    from showdown.battle_bots.greedy_solo.greedy import _is_spread_move

    spread_slots = {
        int(m.slot) for m in moves
        if _is_damaging(m) and _is_spread_move(m)
    }
    if not spread_slots:
        return atoms

    merged: Dict[Tuple[int, int], JointAtom] = {}
    others: List[JointAtom] = []
    for atom in atoms:
        if atom.utility != 'damage' or int(atom.move_slot) not in spread_slots:
            others.append(atom)
            continue
        key = (int(atom.move_slot), int(atom.mega))
        prev = merged.get(key)
        if prev is None:
            merged[key] = JointAtom(
                move_slot=atom.move_slot,
                target=atom.target,
                mega=atom.mega,
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
            continue
        prev.damage_pct.update(atom.damage_pct)
        prev.max_roll.update(atom.max_roll)
        prev.ko_tids |= set(atom.ko_tids)
        prev.effect_bonus = max(float(prev.effect_bonus), float(atom.effect_bonus))
        prev.timing_penalty = max(float(prev.timing_penalty), float(atom.timing_penalty))
        if atom.hits_ally:
            prev.hits_ally = True
            prev.ally_damage_pct = max(prev.ally_damage_pct, atom.ally_damage_pct)
            prev.ally_max_roll = max(prev.ally_max_roll, atom.ally_max_roll)
            prev.ally_ko = prev.ally_ko or atom.ally_ko
        # target canónico: primer foe id (spread no necesita slot rival único)
        if prev.target <= 0 and atom.target > 0:
            prev.target = atom.target
    return others + list(merged.values())


def _fo_acts_before_partner(
    fo_atom: JointAtom,
    partner: JointAtom,
    *,
    fo_poke: Optional[CalcPoke],
    partner_poke: Optional[CalcPoke],
    terrain: Optional[str],
    trick_room: bool,
    our_tailwind: bool = False,
) -> bool:
    """FO (prio +3) suele ir antes; si el partner iría primero, la sinergia es débil."""
    if fo_poke is None or partner_poke is None:
        return True
    from showdown.battle_bots.expectimax_bayesian.speed_order import (
        acts_before,
        move_prio,
        poke_effective_spe,
    )

    spe_f = poke_effective_spe(fo_poke, tailwind=our_tailwind)
    spe_p = poke_effective_spe(partner_poke, tailwind=our_tailwind)
    abi_f = fo_poke.ability if fo_poke is not None else None
    abi_p = partner_poke.ability if partner_poke is not None else None
    p_f = move_prio(fo_atom.move_name, ability=abi_f, terrain=terrain)
    p_p = move_prio(partner.move_name, ability=abi_p, terrain=terrain)
    if acts_before(spe_p, p_p, spe_f, p_f, trick_room=trick_room):
        return False
    return True


def _fo_partner_pressure_bonus(
    left: JointAtom,
    right: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    left_poke: Optional[CalcPoke] = None,
    right_poke: Optional[CalcPoke] = None,
    terrain: Optional[str] = None,
    trick_room: bool = False,
    our_tailwind: bool = False,
) -> float:
    """A8: Fake Out + KO/presión del partner en el mismo turno."""
    if not _FO_PARTNER_PRESSURE:
        return 0.0
    best = 0.0
    for fo_atom, partner, fo_poke, partner_poke in (
        (left, right, left_poke, right_poke),
        (right, left, right_poke, left_poke),
    ):
        if not _is_fakeout_atom(fo_atom):
            continue
        if partner.total_damage_pct <= 0.0 and not partner.has_guaranteed_ko:
            continue
        fo_tid = int(fo_atom.target)
        scale = 1.0
        if not _fo_acts_before_partner(
            fo_atom, partner,
            fo_poke=fo_poke, partner_poke=partner_poke,
            terrain=terrain, trick_room=trick_room,
            our_tailwind=our_tailwind,
        ):
            scale = 0.25

        partner_ko_fo = fo_tid in partner.ko_tids
        partner_ko_any = partner.has_guaranteed_ko
        # Soft KO al target del FO
        for tid, foe in foes:
            if int(tid) != fo_tid:
                continue
            cur_hp = int(foe.cur_hp if foe.cur_hp is not None else 1)
            if cur_hp > 0 and int(partner.max_roll.get(tid, 0)) >= cur_hp:
                partner_ko_fo = True
            break

        dmg_on_fo = float(partner.damage_pct.get(fo_tid, 0.0))
        dmg_elsewhere = float(partner.total_damage_pct) - dmg_on_fo

        chunk = 0.0
        if partner_ko_fo or (partner_ko_any and fo_tid in partner.ko_tids):
            chunk = float(_FO_PARTNER_KO_BONUS)
        elif partner_ko_any:
            # FO niega a uno; partner remata al otro
            chunk = float(_FO_PARTNER_KO_BONUS) * 0.85
        elif dmg_on_fo >= float(_FO_PARTNER_PRESS_PCT):
            chunk = float(_FO_PARTNER_PRESS_BONUS)
        elif dmg_elsewhere >= float(_FO_PARTNER_PRESS_PCT):
            chunk = float(_FO_PARTNER_SPLIT_BONUS)
        elif dmg_on_fo > 0.0 or partner.total_damage_pct > 0.0:
            chunk = float(_FO_PARTNER_SPLIT_BONUS) * 0.5
        best = max(best, chunk * scale)
    return float(best)


def _ordered_atoms(
    left: JointAtom,
    right: JointAtom,
    *,
    left_poke: Optional[CalcPoke],
    right_poke: Optional[CalcPoke],
    terrain: Optional[str],
    trick_room: bool = False,
    our_tailwind: bool = False,
) -> List[JointAtom]:
    """Orden de resolución 1-ply: prioridad de move, luego Spe (TR/Tailwind)."""
    from showdown.battle_bots.expectimax_bayesian.speed_order import (
        acts_before,
        move_prio,
        poke_effective_spe,
    )

    spe_l = (
        poke_effective_spe(left_poke, tailwind=our_tailwind)
        if left_poke is not None else 0
    )
    spe_r = (
        poke_effective_spe(right_poke, tailwind=our_tailwind)
        if right_poke is not None else 0
    )
    abi_l = left_poke.ability if left_poke is not None else None
    abi_r = right_poke.ability if right_poke is not None else None
    p_l = move_prio(left.move_name, ability=abi_l, terrain=terrain)
    p_r = move_prio(right.move_name, ability=abi_r, terrain=terrain)
    if acts_before(spe_l, p_l, spe_r, p_r, trick_room=trick_room):
        return [left, right]
    if acts_before(spe_r, p_r, spe_l, p_l, trick_room=trick_room):
        return [right, left]
    return [left, right]


def _score_pair(
    left: JointAtom,
    right: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    left_poke: Optional[CalcPoke] = None,
    right_poke: Optional[CalcPoke] = None,
    include_friendly_fire: bool = True,
    terrain: Optional[str] = None,
    trick_room: bool = False,
    our_tailwind: bool = False,
    opp_tailwind: bool = False,
    threat_left: float = 0.0,
    threat_right: float = 0.0,
) -> float:
    from showdown.battle_bots.move_mechanics import (
        friendly_fire_pair_penalty,
        move_accuracy,
    )

    if not foes:
        return _utility_score(
            left, right, foes=foes, has_damage=False,
            left_poke=left_poke, right_poke=right_poke,
            threat_left=threat_left, threat_right=threat_right,
            terrain=terrain, trick_room=trick_room,
            our_tailwind=our_tailwind, opp_tailwind=opp_tailwind,
        )

    if _DAMAGE_ISO:
        score = _score_pair_damage_iso(
            left, right, foes,
            move_accuracy=move_accuracy,
            left_poke=left_poke,
            right_poke=right_poke,
            terrain=terrain,
            trick_room=trick_room,
            our_tailwind=our_tailwind,
            opp_tailwind=opp_tailwind,
            threat_left=threat_left,
            threat_right=threat_right,
        )
    else:
        score = _score_pair_v_stable(
            left, right, foes, move_accuracy=move_accuracy,
            left_poke=left_poke, right_poke=right_poke,
        )

    score += float(left.effect_bonus) + float(right.effect_bonus)
    score -= float(left.timing_penalty) + float(right.timing_penalty)
    if include_friendly_fire:
        score -= friendly_fire_pair_penalty(
            left, right, left_poke, right_poke,
        )
    return score


def _score_pair_damage_iso(
    left: JointAtom,
    right: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    move_accuracy,
    left_poke: Optional[CalcPoke] = None,
    right_poke: Optional[CalcPoke] = None,
    terrain: Optional[str] = None,
    trick_room: bool = False,
    our_tailwind: bool = False,
    opp_tailwind: bool = False,
    threat_left: float = 0.0,
    threat_right: float = 0.0,
) -> float:
    """Propuesta A (+B si _SPEED_AWARE; +B-lite focus Spe si _FOCUS_SPEED)."""
    if _SPEED_AWARE and (left_poke is not None or right_poke is not None):
        return _score_pair_damage_iso_speed(
            left, right, foes,
            move_accuracy=move_accuracy,
            left_poke=left_poke,
            right_poke=right_poke,
            terrain=terrain,
            trick_room=trick_room,
            our_tailwind=our_tailwind,
            threat_left=threat_left,
            threat_right=threat_right,
        )

    score = 0.0
    alone_ko: Dict[int, bool] = {}
    seq_scored: Set[int] = set()
    # A4: Helping Hand mecánico — el partner que recibe HH escala ×1.5.
    hh_boost_left = _MECH_SYNERGY and right.utility == 'helpinghand'
    hh_boost_right = _MECH_SYNERGY and left.utility == 'helpinghand'
    mult_l = float(_HH_DAMAGE_MULT) if hh_boost_left else 1.0
    mult_r = float(_HH_DAMAGE_MULT) if hh_boost_right else 1.0

    use_focus_spe = (
        _FOCUS_SPEED
        and (left_poke is not None or right_poke is not None)
    )
    order: Optional[List[JointAtom]] = None
    if use_focus_spe:
        order = _ordered_atoms(
            left, right,
            left_poke=left_poke, right_poke=right_poke,
            terrain=terrain, trick_room=trick_room,
            our_tailwind=our_tailwind,
        )

    for tid, foe in foes:
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else 1)
        if foe.has_substitute or cur_hp <= 0:
            alone_ko[tid] = False
            continue
        l_roll = int(left.max_roll.get(tid, 0) * mult_l)
        r_roll = int(right.max_roll.get(tid, 0) * mult_r)
        alone_ko[tid] = (
            (tid in left.ko_tids or l_roll >= cur_hp)
            or (tid in right.ko_tids or r_roll >= cur_hp)
        )

    for tid, foe in foes:
        max_hp = int((_stats_or_default(foe).get('hp') or 1))
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else max_hp)
        if cur_hp <= 0:
            continue

        l_pct = float(left.damage_pct.get(tid, 0.0)) * mult_l
        r_pct = float(right.damage_pct.get(tid, 0.0)) * mult_r
        l_hits = l_pct > 0.0 or tid in left.ko_tids
        r_hits = r_pct > 0.0 or tid in right.ko_tids

        # B-lite: ambos al mismo foe → aplicar hits en orden prio/Spe.
        if use_focus_spe and order is not None and l_hits and r_hits:
            remain_hp = cur_hp
            for atom in order:
                if remain_hp <= 0:
                    break
                mult = mult_l if atom is left else mult_r
                pct = float(atom.damage_pct.get(tid, 0.0)) * mult
                roll = int(atom.max_roll.get(tid, 0) * mult)
                if pct <= 0.0 and tid not in atom.ko_tids and roll <= 0:
                    continue
                remain_pct = 100.0 * remain_hp / max(max_hp, 1)
                if pct > 0.0 and remain_pct > 0.0:
                    score += min(pct, remain_pct)
                hard = tid in atom.ko_tids
                kos_now = hard or (roll >= remain_hp and remain_hp > 0)
                if kos_now:
                    if hard:
                        score += KO_BONUS
                    elif remain_hp < cur_hp:
                        acc = float(move_accuracy(atom.move_name))
                        score += KO_BONUS * acc
                        score += _JOINT_KO_SYNERGY * acc
                    remain_hp = 0
                else:
                    remain_hp = max(0, remain_hp - max(roll, 0))
            seq_scored.add(int(tid))
            continue

        remain_pct = 100.0 * cur_hp / max(max_hp, 1)
        l_c = 0.0 if (
            l_pct > 0.0 and (
                tid in right.ko_tids
                or int(right.max_roll.get(tid, 0) * mult_r) >= cur_hp
            )
        ) else l_pct
        r_c = 0.0 if (
            r_pct > 0.0 and (
                tid in left.ko_tids
                or int(left.max_roll.get(tid, 0) * mult_l) >= cur_hp
            )
        ) else r_pct
        if _FOCUS_CLIP and l_c > 0.0 and r_c > 0.0:
            score += min(l_c + r_c, remain_pct)
        else:
            score += l_c + r_c

    for tid, foe in foes:
        if int(tid) in seq_scored:
            continue
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else 1)
        if cur_hp <= 0 or foe.has_substitute:
            continue
        if alone_ko.get(tid):
            if tid in left.ko_tids or tid in right.ko_tids:
                score += KO_BONUS
            continue
        combined = (
            int(left.max_roll.get(tid, 0) * mult_l)
            + int(right.max_roll.get(tid, 0) * mult_r)
        )
        if combined < cur_hp:
            continue
        accs = [
            move_accuracy(a.move_name)
            for a in (left, right)
            if float(a.damage_pct.get(tid, 0.0)) > 0 or tid in a.ko_tids
        ]
        score += KO_BONUS * (min(accs) if accs else 1.0)
        score += _JOINT_KO_SYNERGY * (min(accs) if accs else 1.0)

    koed = _pair_koed_tids(
        left, right, foes, mult_l=mult_l, mult_r=mult_r, alone_ko=alone_ko,
    )
    score += _residual_second_bonus(
        left, right, foes, koed=koed, mult_l=mult_l, mult_r=mult_r,
    )
    score -= _focus_waste_penalty(
        left, right, foes, alone_ko=alone_ko, mult_l=mult_l, mult_r=mult_r,
    )
    score += _spread_split_bonus(left, right, foes, alone_ko=alone_ko)
    score += _prio_tempo_bonus(
        left, right, foes,
        left_poke=left_poke, right_poke=right_poke,
        terrain=terrain, trick_room=trick_room,
        our_tailwind=our_tailwind, opp_tailwind=opp_tailwind,
    )

    has_damage = (left.total_damage_pct + right.total_damage_pct) > 0
    score += _utility_score(
        left, right, foes=foes, has_damage=has_damage,
        left_poke=left_poke, right_poke=right_poke,
        threat_left=threat_left, threat_right=threat_right,
        terrain=terrain, trick_room=trick_room,
        our_tailwind=our_tailwind, opp_tailwind=opp_tailwind,
    )
    return score


def _score_pair_damage_iso_speed(
    left: JointAtom,
    right: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    move_accuracy,
    left_poke: Optional[CalcPoke],
    right_poke: Optional[CalcPoke],
    terrain: Optional[str],
    trick_room: bool,
    our_tailwind: bool = False,
    threat_left: float = 0.0,
    threat_right: float = 0.0,
) -> float:
    """A+B: por foe, aplicar hits en orden prio/Spe; clip HP; KO del primero cuenta."""
    order = _ordered_atoms(
        left, right,
        left_poke=left_poke, right_poke=right_poke,
        terrain=terrain, trick_room=trick_room,
        our_tailwind=our_tailwind,
    )
    score = 0.0

    for tid, foe in foes:
        max_hp = int((_stats_or_default(foe).get('hp') or 1))
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else max_hp)
        if cur_hp <= 0 or foe.has_substitute:
            continue

        hitters = [
            a for a in order
            if float(a.damage_pct.get(tid, 0.0)) > 0.0 or tid in a.ko_tids
        ]
        if not hitters:
            continue

        remain_hp = cur_hp
        for atom in hitters:
            if remain_hp <= 0:
                break
            pct = float(atom.damage_pct.get(tid, 0.0))
            roll = int(atom.max_roll.get(tid, 0))
            remain_pct = 100.0 * remain_hp / max(max_hp, 1)
            if pct > 0.0 and remain_pct > 0.0:
                score += min(pct, remain_pct)

            hard = tid in atom.ko_tids
            kos_now = hard or (roll >= remain_hp and remain_hp > 0)
            if kos_now:
                if hard:
                    score += KO_BONUS
                elif remain_hp < cur_hp:
                    # KO solo tras chip del partner más rápido (sinergia Spe).
                    score += KO_BONUS * float(move_accuracy(atom.move_name))
                # Soft KO desde HP lleno → ya en effect_bonus del átomo.
                remain_hp = 0
            else:
                remain_hp = max(0, remain_hp - max(roll, 0))

    alone_ko: Dict[int, bool] = {}
    for tid, foe in foes:
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else 1)
        if foe.has_substitute or cur_hp <= 0:
            alone_ko[tid] = False
            continue
        alone_ko[tid] = (
            (tid in left.ko_tids or int(left.max_roll.get(tid, 0)) >= cur_hp)
            or (tid in right.ko_tids or int(right.max_roll.get(tid, 0)) >= cur_hp)
        )
    koed = _pair_koed_tids(left, right, foes, alone_ko=alone_ko)
    score += _residual_second_bonus(left, right, foes, koed=koed)
    score -= _focus_waste_penalty(left, right, foes, alone_ko=alone_ko)
    score += _spread_split_bonus(left, right, foes, alone_ko=alone_ko)
    score += _prio_tempo_bonus(
        left, right, foes,
        left_poke=left_poke, right_poke=right_poke,
        terrain=terrain, trick_room=trick_room,
        our_tailwind=our_tailwind, opp_tailwind=False,
    )

    has_damage = (left.total_damage_pct + right.total_damage_pct) > 0
    score += _utility_score(
        left, right, foes=foes, has_damage=has_damage,
        left_poke=left_poke, right_poke=right_poke,
        threat_left=threat_left, threat_right=threat_right,
        terrain=terrain, trick_room=trick_room,
        our_tailwind=our_tailwind, opp_tailwind=False,
    )
    return score


def _score_pair_v_stable(
    left: JointAtom,
    right: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    move_accuracy,
    left_poke: Optional[CalcPoke] = None,
    right_poke: Optional[CalcPoke] = None,
) -> float:
    """Scoring previo (v_stable): daño combinado clip + utilities fijas."""
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
            hard = tid in left.ko_tids or tid in right.ko_tids
            solo_soft = False
            for atom in (left, right):
                if tid in atom.ko_tids:
                    continue
                if int(atom.max_roll.get(tid, 0)) >= cur_hp:
                    solo_soft = True
                    break
            if hard:
                score += KO_BONUS
            elif not solo_soft:
                accs = [
                    move_accuracy(a.move_name)
                    for a in (left, right)
                    if float(a.damage_pct.get(tid, 0.0)) > 0 or tid in a.ko_tids
                ]
                score += KO_BONUS * (min(accs) if accs else 1.0)

    if targets_hit >= 2:
        score += _FOCUS_SPLIT_BONUS

    has_damage = (left.total_damage_pct + right.total_damage_pct) > 0
    score += _utility_score(
        left, right, foes=foes, has_damage=has_damage,
        left_poke=left_poke, right_poke=right_poke,
    )
    return score


def _utility_score(
    left: JointAtom,
    right: JointAtom,
    *,
    foes: Sequence[Tuple[int, CalcPoke]],
    has_damage: bool,
    left_poke: Optional[CalcPoke] = None,
    right_poke: Optional[CalcPoke] = None,
    threat_left: float = 0.0,
    threat_right: float = 0.0,
    terrain: Optional[str] = None,
    trick_room: bool = False,
    our_tailwind: bool = False,
    opp_tailwind: bool = False,
) -> float:
    # Propuesta A (+A3/A4): support condicionado; Spe usa TR/Tailwind de campo.
    if _DAMAGE_ISO:
        return _utility_score_damage_iso(
            left, right, foes=foes,
            left_poke=left_poke, right_poke=right_poke,
            threat_left=threat_left, threat_right=threat_right,
            terrain=terrain, trick_room=trick_room,
            our_tailwind=our_tailwind, opp_tailwind=opp_tailwind,
        )

    s = 0.0
    for atom in (left, right):
        if atom.utility == 'protect':
            s += _PROTECT_BONUS if has_damage else _STATUS_BASE
        elif atom.utility == 'redirect':
            s += _REDIRECT_BASE if not has_damage else 0.0
        elif atom.utility == 'fakeout':
            s += _FAKE_OUT_BONUS
        elif atom.utility == 'helpinghand':
            s += _HELPING_HAND_BONUS if has_damage else _STATUS_BASE
        elif atom.utility == 'speed':
            s += _SPEED_CONTROL_BONUS
        elif atom.utility == 'charge':
            s += _charge_utility(atom)
        elif atom.utility == 'status':
            if atom.effect_bonus <= 0:
                s += _STATUS_BASE

    if left.utility == 'protect' and right.total_damage_pct > 0:
        s += 3.0
    if right.utility == 'protect' and left.total_damage_pct > 0:
        s += 3.0
    if left.utility == 'helpinghand' and right.total_damage_pct > 0:
        s += 4.0
    if right.utility == 'helpinghand' and left.total_damage_pct > 0:
        s += 4.0

    if left.utility == 'redirect':
        s += _redirect_synergy(left, right, foes)
    if right.utility == 'redirect':
        s += _redirect_synergy(right, left, foes)
    return s


def _utility_score_damage_iso(
    left: JointAtom,
    right: JointAtom,
    *,
    foes: Sequence[Tuple[int, CalcPoke]],
    left_poke: Optional[CalcPoke],
    right_poke: Optional[CalcPoke],
    threat_left: float = 0.0,
    threat_right: float = 0.0,
    terrain: Optional[str] = None,
    trick_room: bool = False,
    our_tailwind: bool = False,
    opp_tailwind: bool = False,
) -> float:
    """A2/A3/A4/A8 utilities bajo damage-iso."""
    s = 0.0
    for atom, poke, partner, self_threat, partner_threat in (
        (left, left_poke, right, threat_left, threat_right),
        (right, right_poke, left, threat_right, threat_left),
    ):
        if atom.utility == 'charge':
            s += _charge_utility(atom)
            continue

        partner_dmg = partner.total_damage_pct > 0
        partner_ko = partner.has_guaranteed_ko

        if atom.utility == 'protect':
            if not partner_dmg:
                continue
            hp = _hp_pct(poke)
            if hp <= _PROTECT_REACT_HP_PCT:
                s += _PROTECT_REACT_BONUS
            elif _OPP_THREAT and self_threat >= _PROTECT_THREAT_PCT:
                race_ok = (
                    not _SPE_RACE_GATE
                    or _foe_acts_before_us(
                        poke, foes, terrain=terrain, trick_room=trick_room,
                        our_tailwind=our_tailwind, opp_tailwind=opp_tailwind,
                    )
                )
                if race_ok:
                    s += _PROTECT_THREAT_BONUS
            continue

        if not _SYNERGY_LITE and not _MECH_SYNERGY:
            continue

        if atom.utility == 'fakeout' and (partner_dmg or partner_ko):
            # Sin A4: bono plano. Con A4 el FO damage-atom ya lleva deny.
            # A8 añade presión/KO del partner vía _fo_partner_pressure_bonus.
            if not _MECH_SYNERGY:
                s += _FO_PARTNER_BONUS
        elif atom.utility == 'helpinghand' and partner_dmg:
            # A4: el ×1.5 ya está en daño; bono plano solo si A3 sin mecánica.
            if not _MECH_SYNERGY:
                s += _HH_PARTNER_BONUS
            else:
                s += 2.0  # pequeño nudge para preferir HH+ataque vs doble ataque débil
        elif atom.utility == 'speed' and partner_dmg:
            s += _field_setup_bonus(
                atom, partner,
                left_poke=left_poke, right_poke=right_poke,
                foes=foes,
                trick_room=trick_room,
                our_tailwind=our_tailwind,
                opp_tailwind=opp_tailwind,
            )
        elif atom.utility == 'redirect' and (partner_dmg or partner_ko):
            s += _REDIRECT_LITE_SCALE * _redirect_synergy(atom, partner, foes)
            if _OPP_THREAT and partner_threat >= _REDIRECT_SAVE_THREAT_PCT:
                partner_poke = right_poke if atom is left else left_poke
                race_ok = (
                    not _SPE_RACE_GATE
                    or _foe_acts_before_us(
                        partner_poke, foes,
                        terrain=terrain, trick_room=trick_room,
                        our_tailwind=our_tailwind, opp_tailwind=opp_tailwind,
                    )
                )
                if race_ok:
                    s += _REDIRECT_SAVE_BONUS

    s += _fo_partner_pressure_bonus(
        left, right, foes,
        left_poke=left_poke, right_poke=right_poke,
        terrain=terrain, trick_room=trick_room,
        our_tailwind=our_tailwind,
    )
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

    threat_left = _side_incoming_threat(left_atk, foes) if _OPP_THREAT else 0.0
    threat_right = _side_incoming_threat(right_atk, foes) if _OPP_THREAT else 0.0
    foe_threats = (
        _foe_threat_by_tid(left_atk, right_atk, foes)
        if (_OPP_THREAT or _MECH_SYNERGY) else {}
    )

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
            threat_by_tid=foe_threats,
            trick_room=bool(ctx.trick_room),
            our_tailwind=bool(ctx.tailwind),
            opp_tailwind=bool(ctx.opp_tailwind),
        )
        if left_atk and left_moves else []
    )
    right_atoms = (
        _atoms_from_side(
            right_moves, right_meta, right_ranges, foes,
            can_mega=can_mega_right, attacker=right_atk,
            weather=ctx.weather, terrain=ctx.terrain, gravity=ctx.gravity,
            ally=left_atk, ally_hit_by_slot=ally_hit_r,
            threat_by_tid=foe_threats,
            trick_room=bool(ctx.trick_room),
            our_tailwind=bool(ctx.tailwind),
            opp_tailwind=bool(ctx.opp_tailwind),
        )
        if right_atk and right_moves else []
    )

    if not left_atoms and not right_atoms:
        note_turn_kind('none')
        return None, None, 0.0
    def _solo_side_key(a: JointAtom) -> float:
        s = a.total_damage_pct + float(a.effect_bonus) - float(a.timing_penalty)
        if a.utility == 'charge':
            s += _charge_utility(a)
        if not _DAMAGE_ISO:
            if a.utility == 'fakeout':
                s += _FAKE_OUT_BONUS
            elif a.utility == 'redirect':
                s += _REDIRECT_BASE
        return s

    if not left_atoms:
        best_r = max(right_atoms, key=_solo_side_key)
        note_turn_kind(classify_pair(None, best_r))
        return None, _atom_to_scored(best_r), best_r.total_damage_pct
    if not right_atoms:
        best_l = max(left_atoms, key=_solo_side_key)
        note_turn_kind(classify_pair(best_l, None))
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
                terrain=ctx.terrain,
                trick_room=bool(ctx.trick_room),
                our_tailwind=bool(ctx.tailwind),
                opp_tailwind=bool(ctx.opp_tailwind),
                threat_left=threat_left,
                threat_right=threat_right,
            )
            if s > best_score:
                best_score = s
                best_pair = (pair_l, pair_r)

    assert best_pair is not None
    bl, br = best_pair
    note_turn_kind(classify_pair(bl, br))
    return _atom_to_scored(bl), _atom_to_scored(br), best_score


def _apply_env_overrides() -> None:
    """Diagnóstico de ablación: ``JOINT_OVERRIDES="_OPP_THREAT=0;_RESIDUAL_SCALE=0.2"``.

    Solo cambia constantes de módulo ya existentes (bool/float); sin la variable no hace nada.
    """
    import os

    raw = os.environ.get('JOINT_OVERRIDES', '').strip()
    if not raw:
        return
    g = globals()
    for item in raw.split(';'):
        if '=' not in item:
            continue
        name, value = (x.strip() for x in item.split('=', 1))
        if name not in g or name.startswith('__'):
            raise ValueError(f'JOINT_OVERRIDES: constante desconocida {name!r}')
        cur = g[name]
        if isinstance(cur, bool):
            g[name] = value.lower() in ('1', 'true', 'yes', 'on')
        elif isinstance(cur, (int, float)):
            g[name] = type(cur)(float(value))
        else:
            raise ValueError(f'JOINT_OVERRIDES: {name!r} no es bool/numérica')


_apply_env_overrides()
