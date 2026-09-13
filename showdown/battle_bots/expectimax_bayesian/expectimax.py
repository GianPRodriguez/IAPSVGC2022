"""Expectimax bayesiano 1-ply anytime (fallback Greedy-Joint).

U(a) ≈ Σ_j P(θ_j) Σ_b π_opp(b|θ_j) E_rolls[V(S'|a,b,θ_j)]

Mejoras:
  A) Moves OTS/revelados → π_opp / belief
  B) Orden de turno Spe + prioridad en V
  C) Switches voluntarios con utilidad continua (no V→0)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from showdown.battle_bots.greedy_joint.joint import (
    JointAtom,
    _atoms_from_side,
    _norm_move,
    _score_pair,
    _strip_mega,
    best_joint_double_action,
)
from showdown.battle_bots.greedy_solo.calc_async import calc_damage_batch
from showdown.battle_bots.greedy_solo.greedy import (
    CalcPoke,
    FieldContext,
    MoveOption,
    ScoredAction,
    _collect_side_requests,
    _stats_or_default,
    apply_pending_mega,
)
from showdown.battle_bots.greedy_solo.type_chart import matchup_vs_actives
from showdown.battle_bots.expectimax_bayesian.belief import (
    belief_nodes_for_foes,
    iter_worlds,
    profile_to_calc,
    resolve_opp_move_names,
)
from showdown.battle_bots.expectimax_bayesian.chance import (
    SOFT_KO_BONUS,
    combined_expected_damage,
    combined_p_ko,
    expected_damage,
    p_ko,
)
from showdown.battle_bots.expectimax_bayesian.opp_policy import (
    OppAction,
    is_control_move,
    move_options_from_names,
    opp_policy_distribution,
)
from showdown.battle_bots.expectimax_bayesian.speed_order import (
    acts_before,
    move_prio,
    poke_effective_spe,
    race_scale,
)
from showdown.battle_bots.expectimax_bayesian.utility import (
    FieldPositionBonus,
    JointActionUtility,
    field_control_bonus,
    switch_preservation_value,
)


@dataclass
class ExpectimaxConfig:
    deadline_ms: float = 700.0
    phase1_budget_ms: float = 50.0
    phase2_end_ms: float = 680.0
    k_profiles: int = 5
    max_worlds: int = 16
    max_our_pairs: int = 36
    soft_ko_bonus: float = SOFT_KO_BONUS
    enable_voluntary_switch: bool = True
    max_switch_hybrids: int = 8
    loss_weight: float = 1000.0
    save_weight: float = 220.0
    chip_soft_cap: float = 30.0
    switch_min_stay_p_ko: float = 0.25


@dataclass
class ExpectimaxMeta:
    used_fallback: bool = True
    worlds: int = 0
    pairs_scored: int = 0
    elapsed_ms: float = 0.0
    phase: str = 'fallback'
    utility: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReserveOption:
    """Bench candidato a switch voluntario."""

    slot: int  # 3 o 4 (IAP) / índice poke-env mapeado
    poke: CalcPoke
    types: Tuple[Optional[str], ...] = ()


def _elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _speed_control(name: str) -> bool:
    return _norm_move(name) in (
        'tailwind', 'trickroom', 'icywind', 'electroweb', 'bulldoze', 'glaciate',
    )


def _atom_key(a: JointAtom) -> Tuple[int, int, int, str]:
    return (a.move_slot, a.target, a.mega, a.utility)


def _pair_key(l: JointAtom, r: JointAtom) -> Tuple:
    return (_atom_key(l), _atom_key(r))


def _switch_atom(slot: int) -> JointAtom:
    return JointAtom(
        move_slot=int(slot),
        target=0,
        mega=0,
        utility='switch',
        move_name=f'Switch {slot}',
    )


def _candidate_pairs(
    left_atoms: Sequence[JointAtom],
    right_atoms: Sequence[JointAtom],
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    max_pairs: int,
    switch_left: Sequence[JointAtom] = (),
    switch_right: Sequence[JointAtom] = (),
    max_switch_hybrids: int = 8,
) -> List[Tuple[JointAtom, JointAtom]]:
    move_left = [a for a in left_atoms if a.utility != 'switch']
    move_right = [a for a in right_atoms if a.utility != 'switch']

    if not move_left and not move_right and not switch_left and not switch_right:
        return []

    scored: List[Tuple[float, JointAtom, JointAtom]] = []
    if move_left and move_right:
        for a1 in move_left:
            for a2 in move_right:
                pl, pr = a1, a2
                if a1.mega and a2.mega:
                    if a1.total_damage_pct >= a2.total_damage_pct:
                        pr = _strip_mega(a2)
                    else:
                        pl = _strip_mega(a1)
                s = _score_pair(pl, pr, foes)
                scored.append((s, pl, pr))
    elif move_left:
        for a1 in move_left:
            scored.append((a1.total_damage_pct, a1, JointAtom(0, 0, utility='status')))
    elif move_right:
        for a2 in move_right:
            scored.append((a2.total_damage_pct, JointAtom(0, 0, utility='status'), a2))

    scored.sort(key=lambda x: -x[0])
    budget = max(1, int(max_pairs) - int(max_switch_hybrids))
    out: List[Tuple[JointAtom, JointAtom]] = []
    seen = set()
    for _, pl, pr in scored:
        key = _pair_key(pl, pr)
        if key in seen:
            continue
        seen.add(key)
        out.append((pl, pr))
        if len(out) >= budget:
            break

    top_l: List[JointAtom] = []
    top_r: List[JointAtom] = []
    for _, pl, pr in scored[:6]:
        if pl.utility != 'switch' and pl.move_slot > 0 and len(top_l) < 2:
            top_l.append(pl)
        if pr.utility != 'switch' and pr.move_slot > 0 and len(top_r) < 2:
            top_r.append(pr)
    if not top_l:
        top_l = list(move_left[:2])
    if not top_r:
        top_r = list(move_right[:2])

    hybrids = 0
    pad = JointAtom(0, 0, utility='status')
    for sw in switch_left:
        for atk in (top_r or [pad]):
            key = _pair_key(sw, atk)
            if key in seen:
                continue
            seen.add(key)
            out.append((sw, atk))
            hybrids += 1
            if hybrids >= max_switch_hybrids:
                break
        if hybrids >= max_switch_hybrids:
            break
    if hybrids < max_switch_hybrids:
        for sw in switch_right:
            for atk in (top_l or [pad]):
                key = _pair_key(atk, sw)
                if key in seen:
                    continue
                seen.add(key)
                out.append((atk, sw))
                hybrids += 1
                if hybrids >= max_switch_hybrids:
                    break
            if hybrids >= max_switch_hybrids:
                break
    return out[: max(1, int(max_pairs))]


def _hit_map_from_batch(
    meta: Sequence[Tuple[str, MoveOption, int, CalcPoke, int]],
    ranges: Sequence[Optional[Tuple[int, int]]],
    *,
    attackers: Optional[Dict[str, Optional[CalcPoke]]] = None,
    weather: Optional[str] = None,
) -> Dict[Tuple[str, int, int, int], Optional[Tuple[int, int]]]:
    from showdown.battle_bots.move_mechanics import gate_damage_range

    out: Dict[Tuple[str, int, int, int], Optional[Tuple[int, int]]] = {}
    atk_map = attackers or {}
    for (side, move, tid, _foe, special), rng in zip(meta, ranges):
        atk = atk_map.get(side)
        gated = gate_damage_range(
            rng,
            move.name,
            weather=weather,
            item=atk.item if atk is not None else None,
            already_charging=bool(atk.is_two_turn_charging) if atk is not None else False,
            must_recharge=bool(atk.must_recharge) if atk is not None else False,
        )
        out[(side, move.slot, tid, special)] = gated
    return out


def _atom_ranges_on_foe(
    side: str,
    atom: JointAtom,
    tid: int,
    hit_map: Dict[Tuple[str, int, int, int], Optional[Tuple[int, int]]],
) -> Optional[Tuple[int, int]]:
    if atom.utility == 'switch' or atom.move_slot <= 0:
        return None
    if atom.utility not in ('damage',) and not atom.damage_pct and not atom.max_roll:
        return None
    special = 2 if atom.mega == 2 else 0
    rng = hit_map.get((side, atom.move_slot, tid, special))
    if rng is None and special == 2:
        rng = hit_map.get((side, atom.move_slot, tid, 0))
    if rng is not None:
        return rng
    mx = int(atom.max_roll.get(tid, 0))
    if mx > 0:
        return (max(1, int(round(mx * 0.85))), mx)
    return None


def _our_pair_offense_detail(
    left: JointAtom,
    right: JointAtom,
    foes: Sequence[Tuple[int, CalcPoke]],
    hit_map: Dict[Tuple[str, int, int, int], Optional[Tuple[int, int]]],
    *,
    soft_ko_bonus: float,
) -> Tuple[float, float, Dict[int, float], Dict[str, float]]:
    """
    Returns offense, mean_foe_hp_after, p_ko_by_foe_tid, offense_share_by_side.
    """
    score = 0.0
    foe_remain = 0.0
    n_foe = 0
    p_ko_foe: Dict[int, float] = {}
    share = {'L': 0.0, 'R': 0.0}
    for tid, foe in foes:
        max_hp = int((_stats_or_default(foe).get('hp') or 1))
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else max_hp)
        ranges: List[Optional[Tuple[int, int]]] = []
        side_ranges: Dict[str, List[Optional[Tuple[int, int]]]] = {'L': [], 'R': []}
        for side, atom in (('L', left), ('R', right)):
            rng = _atom_ranges_on_foe(side, atom, tid, hit_map)
            if rng is not None:
                ranges.append(rng)
                side_ranges[side].append(rng)

        remain = max(0.0, 100.0 * cur_hp / max(max_hp, 1))
        if foe.has_substitute:
            e_dmg = combined_expected_damage(ranges)
            pct = 100.0 * e_dmg / float(max_hp) if max_hp else 0.0
            score += min(pct, remain)
            p_ko_foe[tid] = 0.0
            foe_remain += remain
            n_foe += 1
            continue

        e_dmg = combined_expected_damage(ranges)
        pk = combined_p_ko(ranges, cur_hp)
        pct = 100.0 * e_dmg / float(max_hp) if max_hp else 0.0
        part = min(pct, remain) + pk * soft_ko_bonus
        score += part
        p_ko_foe[tid] = pk
        foe_remain += remain * (1.0 - pk)
        n_foe += 1
        # Rough share by side expected dmg
        for side in ('L', 'R'):
            share[side] += combined_expected_damage(side_ranges[side])

    mean_foe = foe_remain / float(n_foe) if n_foe else 0.0
    return score, mean_foe, p_ko_foe, share


def _opp_threat_detail(
    our_left: Optional[CalcPoke],
    our_right: Optional[CalcPoke],
    left_atom: JointAtom,
    right_atom: JointAtom,
    opp_hit_map: Dict[Tuple[str, str, str], Optional[Tuple[int, int]]],
    opp_dists: Dict[str, List[Tuple[OppAction, float]]],
    *,
    soft_ko_bonus: float,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Returns threat_total, p_ko_on_side (OL/OR), e_dmg_pct_on_side.
    Switch/Protect → 0 amenaza sobre ese activo.
    """
    threat = 0.0
    p_ko_side: Dict[str, float] = {'OL': 0.0, 'OR': 0.0}
    pct_side: Dict[str, float] = {'OL': 0.0, 'OR': 0.0}
    sides = [
        ('OL', our_left, left_atom),
        ('OR', our_right, right_atom),
    ]
    for side_key, our, our_atom in sides:
        if our is None:
            continue
        if our_atom.utility in ('switch', 'protect') or our.is_protected:
            continue
        max_hp = int((_stats_or_default(our).get('hp') or 1))
        cur_hp = int(our.cur_hp if our.cur_hp is not None else max_hp)
        dist = opp_dists.get(side_key) or []
        side_threat = 0.0
        side_pk = 0.0
        side_pct = 0.0
        for action, p_b in dist:
            best_rng = None
            for atk_tag in ('F1', 'F2'):
                rng = opp_hit_map.get((atk_tag, _norm_move(action.move_name), side_key))
                if rng is not None:
                    best_rng = rng
                    break
            if best_rng is None:
                continue
            e_pct = 100.0 * expected_damage(best_rng) / float(max_hp) if max_hp else 0.0
            pk = p_ko(best_rng, cur_hp)
            contrib = float(p_b) * (min(e_pct, 100.0 * cur_hp / max(max_hp, 1)) + pk * soft_ko_bonus)
            side_threat += contrib
            side_pk += float(p_b) * pk
            side_pct += float(p_b) * e_pct
        threat += side_threat
        p_ko_side[side_key] = min(1.0, side_pk)
        pct_side[side_key] = side_pct
    return threat, p_ko_side, pct_side


def _bench_incoming_pct(
    reserve: ReserveOption,
    foes: Sequence[Tuple[int, CalcPoke]],
    opp_hit_map: Dict[Tuple[str, str, str], Optional[Tuple[int, int]]],
    opp_move_names: Sequence[str],
) -> float:
    """Proxy: max E[dmg%] de moves rival vs bench (si hay calc); si no, tipado."""
    max_hp = int((_stats_or_default(reserve.poke).get('hp') or 1))
    best = 0.0
    for name in opp_move_names:
        if is_control_move(name):
            continue
        mid = _norm_move(name)
        for atk_tag in ('F1', 'F2'):
            # Reuse calcs vs OL/OR as rough proxy if same bulk order — weak.
            # Prefer type proxy below if no dedicated bench keys.
            for def_tag in ('OL', 'OR'):
                rng = opp_hit_map.get((atk_tag, mid, def_tag))
                if rng is None:
                    continue
                pct = 100.0 * expected_damage(rng) / float(max_hp) if max_hp else 0.0
                # Scale by HP ratio active→bench if stats differ
                best = max(best, pct * 0.85)
    if best > 0:
        return best
    foe_types = [tuple(f.types or ()) for _, f in foes]
    mu = matchup_vs_actives(reserve.types or reserve.poke.types, foe_types or [tuple()])
    # mu alto = buen matchup defensivo → menos chip
    return max(0.0, 35.0 - float(mu) * 12.0)


def _apply_speed_race(
    *,
    offense: float,
    threat: float,
    share: Dict[str, float],
    p_ko_foe: Dict[int, float],
    p_ko_side: Dict[str, float],
    left: JointAtom,
    right: JointAtom,
    left_atk: Optional[CalcPoke],
    right_atk: Optional[CalcPoke],
    world_foes: Sequence[Tuple[int, CalcPoke]],
    pos: FieldPositionBonus,
    terrain: Optional[str],
) -> Tuple[float, float]:
    """Escala offense/threat según quién actúa primero (Spe + prioridad)."""
    from showdown.battle_bots.move_mechanics import priority_move_fails

    tr = bool(pos.our_trick_room or pos.opp_trick_room)
    our_tw = bool(pos.our_tailwind)
    opp_tw = bool(pos.opp_tailwind)
    foe_mons = [f for _, f in world_foes]

    spe_l = poke_effective_spe(left_atk, tailwind=our_tw) if left_atk else 0
    spe_r = poke_effective_spe(right_atk, tailwind=our_tw) if right_atk else 0

    def _our_prio(atom: JointAtom, atk: Optional[CalcPoke]) -> int:
        if atom.utility == 'switch' or atk is None:
            return 0
        if _norm_move(atom.move_name) == 'fakeout' and not atk.just_switched:
            return 0
        defender = None
        for tid, foe in world_foes:
            if tid == atom.target:
                defender = foe
                break
        if defender is None and world_foes:
            defender = world_foes[0][1]
        if priority_move_fails(
            atom.move_name,
            attacker=atk,
            defender=defender,
            defender_side=foe_mons,
            terrain=terrain,
        ):
            return 0
        return move_prio(atom.move_name, ability=atk.ability, terrain=terrain)

    prio_l = _our_prio(left, left_atk)
    prio_r = _our_prio(right, right_atk)

    threat_scale = 1.0
    for i, (tid, foe) in enumerate(world_foes):
        foe_spe = poke_effective_spe(foe, tailwind=opp_tw)
        # Prioridad rival: asume 0 (daño estándar); Fake Out rival raro sin just_switched
        foe_prio = 0
        if foe.just_switched:
            foe_prio = 1  # leve sesgo first-turn
        # ¿Algún nuestro que pega a este foe va antes?
        we_first = False
        for atom, spe, prio in ((left, spe_l, prio_l), (right, spe_r, prio_r)):
            if atom.utility == 'switch':
                continue
            if atom.utility == 'damage' or atom.damage_pct or atom.target == tid:
                if acts_before(spe, prio, foe_spe, foe_prio, trick_room=tr):
                    we_first = True
                    break
            elif acts_before(spe, prio, foe_spe, foe_prio, trick_room=tr):
                we_first = True
                break
        if we_first:
            threat_scale *= race_scale(p_ko_foe.get(tid, 0.0))

    # Si un foe nos KO-ea yendo primero, reduce share ofensivo del slot
    off_l = share.get('L', 0.0)
    off_r = share.get('R', 0.0)
    total_share = off_l + off_r
    if total_share <= 1e-9:
        return offense * 1.0, threat * threat_scale

    scale_l = 1.0
    scale_r = 1.0
    for tid, foe in world_foes:
        foe_spe = poke_effective_spe(foe, tailwind=opp_tw)
        foe_prio = 0
        if left_atk and acts_before(foe_spe, foe_prio, spe_l, prio_l, trick_room=tr):
            scale_l = min(scale_l, race_scale(p_ko_side.get('OL', 0.0)))
        if right_atk and acts_before(foe_spe, foe_prio, spe_r, prio_r, trick_room=tr):
            scale_r = min(scale_r, race_scale(p_ko_side.get('OR', 0.0)))

    offense_scale = (off_l * scale_l + off_r * scale_r) / total_share
    return offense * offense_scale, threat * threat_scale


def _scored_from_atom(atom: JointAtom, score: float, foes: Sequence[Tuple[int, CalcPoke]]) -> ScoredAction:
    if atom.utility == 'switch':
        return ScoredAction(
            score=score,
            move_slot=atom.move_slot,
            target=0,
            mega=0,
            damage_pct=0.0,
            ko=False,
            kind='switch',
        )
    return ScoredAction(
        score=score,
        move_slot=atom.move_slot,
        target=atom.target if atom.target else (foes[0][0] if foes else 0),
        mega=atom.mega,
        damage_pct=atom.total_damage_pct,
        ko=atom.has_guaranteed_ko or atom.mega == 2,
        kind='move',
    )


async def _build_atoms(
    left_attacker: Optional[CalcPoke],
    right_attacker: Optional[CalcPoke],
    left_moves: Sequence[MoveOption],
    right_moves: Sequence[MoveOption],
    foes: Sequence[Tuple[int, CalcPoke]],
    *,
    field_ctx: FieldContext,
    can_mega_left: bool,
    can_mega_right: bool,
    can_tera_left: bool,
    can_tera_right: bool,
    tera_left: Optional[str],
    tera_right: Optional[str],
) -> Tuple[List[JointAtom], List[JointAtom]]:
    left_atk = apply_pending_mega(left_attacker) if (left_attacker and can_mega_left) else left_attacker
    right_atk = apply_pending_mega(right_attacker) if (right_attacker and can_mega_right) else right_attacker
    requests: List[dict] = []
    meta: List[Tuple[str, MoveOption, int, CalcPoke, int]] = []
    if left_atk and left_moves:
        r, m = _collect_side_requests(
            'L', left_atk, right_atk, foes, left_moves,
            field_ctx=field_ctx, can_tera=can_tera_left, tera_type=tera_left,
        )
        requests.extend(r)
        meta.extend(m)
    if right_atk and right_moves:
        r, m = _collect_side_requests(
            'R', right_atk, left_atk, foes, right_moves,
            field_ctx=field_ctx, can_tera=can_tera_right, tera_type=tera_right,
        )
        requests.extend(r)
        meta.extend(m)
    try:
        ranges = await calc_damage_batch(requests) if requests else []
    except Exception:
        ranges = [None] * len(requests)

    left_meta = [m for m in meta if m[0] == 'L']
    right_meta = [m for m in meta if m[0] == 'R']
    left_ranges = [rng for m, rng in zip(meta, ranges) if m[0] == 'L']
    right_ranges = [rng for m, rng in zip(meta, ranges) if m[0] == 'R']
    left_atoms = (
        _atoms_from_side(
            left_moves, left_meta, left_ranges, foes,
            can_mega=can_mega_left, attacker=left_atk,
            weather=field_ctx.weather, terrain=field_ctx.terrain,
            gravity=field_ctx.gravity,
        )
        if left_atk and left_moves else []
    )
    right_atoms = (
        _atoms_from_side(
            right_moves, right_meta, right_ranges, foes,
            can_mega=can_mega_right, attacker=right_atk,
            weather=field_ctx.weather, terrain=field_ctx.terrain,
            gravity=field_ctx.gravity,
        )
        if right_atk and right_moves else []
    )
    return left_atoms, right_atoms


async def best_expectimax_bayesian_action(
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
    beliefs=None,
    position: Optional[FieldPositionBonus] = None,
    config: Optional[ExpectimaxConfig] = None,
    foe_known_moves: Optional[Dict[int, Sequence[str]]] = None,
    reserves: Optional[Sequence[ReserveOption]] = None,
) -> Tuple[Optional[ScoredAction], Optional[ScoredAction], float, ExpectimaxMeta]:
    """Anytime Bayesian expectimax 1-ply; nunca falla sin acción (fallback joint)."""
    cfg = config or ExpectimaxConfig()
    pos = position or FieldPositionBonus(
        tera_available_left=can_tera_left,
        tera_available_right=can_tera_right,
        mega_available_left=can_mega_left,
        mega_available_right=can_mega_right,
    )
    ctx = field_ctx or FieldContext(weather=weather, terrain=terrain)
    meta = ExpectimaxMeta()
    t0 = time.perf_counter()
    known_moves = foe_known_moves or {}

    # --- Phase 1: deterministic joint fallback ---
    try:
        fb_l, fb_r, fb_score = await best_joint_double_action(
            left_attacker, right_attacker, left_moves, right_moves, foes,
            weather=weather, terrain=terrain,
            can_mega_left=can_mega_left, can_mega_right=can_mega_right,
            can_tera_left=can_tera_left, can_tera_right=can_tera_right,
            tera_left=tera_left, tera_right=tera_right,
            field_ctx=ctx,
        )
    except Exception as exc:
        meta.errors.append(f'phase1:{exc!r}')
        meta.elapsed_ms = _elapsed_ms(t0)
        return None, None, 0.0, meta

    best_l, best_r, best_u = fb_l, fb_r, float(fb_score)
    meta.utility = best_u
    meta.phase = 'phase1'
    meta.used_fallback = True

    if _elapsed_ms(t0) >= cfg.phase2_end_ms:
        meta.elapsed_ms = _elapsed_ms(t0)
        return best_l, best_r, best_u, meta

    # --- Phase 2: Bayesian refinement ---
    try:
        left_atoms, right_atoms = await _build_atoms(
            left_attacker, right_attacker, left_moves, right_moves, foes,
            field_ctx=ctx,
            can_mega_left=can_mega_left, can_mega_right=can_mega_right,
            can_tera_left=can_tera_left, can_tera_right=can_tera_right,
            tera_left=tera_left, tera_right=tera_right,
        )

        switch_left: List[JointAtom] = []
        switch_right: List[JointAtom] = []
        reserve_list = list(reserves or [])
        if cfg.enable_voluntary_switch and reserve_list:
            for res in reserve_list[:4]:
                sw = _switch_atom(res.slot)
                if left_attacker is not None:
                    switch_left.append(sw)
                if right_attacker is not None:
                    switch_right.append(sw)

        pairs = _candidate_pairs(
            left_atoms, right_atoms, foes,
            max_pairs=cfg.max_our_pairs,
            switch_left=switch_left,
            switch_right=switch_right,
            max_switch_hybrids=cfg.max_switch_hybrids if cfg.enable_voluntary_switch else 0,
        )
        if not pairs:
            meta.elapsed_ms = _elapsed_ms(t0)
            return best_l, best_r, best_u, meta

        nodes = belief_nodes_for_foes(
            foes, beliefs=beliefs, k=cfg.k_profiles, foe_known_moves=known_moves,
        )
        foe_order = [tid for tid, _ in foes]
        worlds = iter_worlds(nodes, foe_order, max_worlds=cfg.max_worlds)
        meta.worlds = len(worlds)

        utilities: Dict[Tuple, JointActionUtility] = {}
        for pl, pr in pairs:
            utilities[_pair_key(pl, pr)] = JointActionUtility(
                left_slot=pl.move_slot,
                right_slot=pr.move_slot,
                left_target=pl.target,
                right_target=pr.target,
                left_mega=pl.mega,
                right_mega=pr.mega,
                left_move=pl.move_name,
                right_move=pr.move_name,
            )

        left_atk = apply_pending_mega(left_attacker) if (left_attacker and can_mega_left) else left_attacker
        right_atk = apply_pending_mega(right_attacker) if (right_attacker and can_mega_right) else right_attacker
        reserve_by_slot = {int(r.slot): r for r in reserve_list}

        for world_profiles, p_world in worlds:
            if _elapsed_ms(t0) >= cfg.phase2_end_ms:
                break

            world_foes: List[Tuple[int, CalcPoke]] = []
            for tid, foe in foes:
                prof = world_profiles.get(tid)
                if prof is None:
                    world_foes.append((tid, foe))
                    continue
                base_stats = None
                try:
                    if beliefs is not None and hasattr(beliefs, 'beliefs'):
                        b = beliefs.beliefs.get(prof.species) or beliefs.beliefs.get(foe.species)
                        if b is not None:
                            base_stats = getattr(b, 'base_stats', None)
                except Exception:
                    base_stats = None
                try:
                    world_foes.append((tid, profile_to_calc(foe, prof, base_stats=base_stats)))
                except Exception:
                    world_foes.append((tid, foe))

            reqs: List[dict] = []
            meta_hits: List[Tuple[str, MoveOption, int, CalcPoke, int]] = []
            if left_atk and left_moves:
                r, m = _collect_side_requests(
                    'L', left_atk, right_atk, world_foes, left_moves,
                    field_ctx=ctx, can_tera=can_tera_left, tera_type=tera_left,
                )
                reqs.extend(r)
                meta_hits.extend(m)
            if right_atk and right_moves:
                r, m = _collect_side_requests(
                    'R', right_atk, left_atk, world_foes, right_moves,
                    field_ctx=ctx, can_tera=can_tera_right, tera_type=tera_right,
                )
                reqs.extend(r)
                meta_hits.extend(m)

            opp_reqs: List[dict] = []
            opp_meta: List[Tuple[str, str, str]] = []
            opp_dists: Dict[str, List[Tuple[OppAction, float]]] = {'OL': [], 'OR': []}
            all_opp_move_names: List[str] = []

            for i, (tid, foe) in enumerate(world_foes):
                tag = 'F1' if i == 0 else 'F2'
                prof = world_profiles.get(tid)
                move_names = list(resolve_opp_move_names(prof, known_moves.get(tid) or ()))
                if not move_names:
                    continue
                all_opp_move_names.extend(move_names)
                opp_moves = move_options_from_names(move_names)
                dmg_slots = {
                    m.slot: (0.0 if is_control_move(m.name) else 1.0)
                    for m in opp_moves
                }
                for def_tag, our in (('OL', left_atk), ('OR', right_atk)):
                    if our is None:
                        continue
                    if not opp_dists[def_tag]:
                        opp_dists[def_tag] = opp_policy_distribution(
                            opp_moves,
                            damage_by_slot=dmg_slots,
                            default_target=1 if def_tag == 'OL' else 2,
                        )
                    for m in opp_moves:
                        if is_control_move(m.name):
                            continue
                        r, _mta = _collect_side_requests(
                            tag, foe, None, [(1 if def_tag == 'OL' else 2, our)], [m],
                            field_ctx=ctx, can_tera=False, tera_type=None,
                        )
                        for _req in r:
                            opp_reqs.append(_req)
                            opp_meta.append((tag, _norm_move(m.name), def_tag))

            try:
                our_ranges = await calc_damage_batch(reqs) if reqs else []
            except Exception as exc:
                meta.errors.append(f'calc_our:{exc!r}')
                our_ranges = [None] * len(reqs)
            try:
                opp_ranges = await calc_damage_batch(opp_reqs) if opp_reqs else []
            except Exception as exc:
                meta.errors.append(f'calc_opp:{exc!r}')
                opp_ranges = [None] * len(opp_reqs)

            hit_map = _hit_map_from_batch(
                meta_hits,
                our_ranges,
                attackers={'L': left_atk, 'R': right_atk},
                weather=ctx.weather,
            )
            opp_hit_map: Dict[Tuple[str, str, str], Optional[Tuple[int, int]]] = {}
            from showdown.battle_bots.move_mechanics import gate_damage_range, priority_move_fails
            foe_by_tag = {
                'F1': world_foes[0][1] if len(world_foes) > 0 else None,
                'F2': world_foes[1][1] if len(world_foes) > 1 else None,
            }
            our_by_def = {'OL': left_atk, 'OR': right_atk}
            our_side = [p for p in (left_atk, right_atk) if p is not None]
            for key, rng in zip(opp_meta, opp_ranges):
                tag, move_id, def_tag = key
                foe = foe_by_tag.get(tag)
                our = our_by_def.get(def_tag)
                gated = gate_damage_range(
                    rng,
                    move_id,
                    weather=ctx.weather,
                    item=foe.item if foe is not None else None,
                    already_charging=bool(foe.is_two_turn_charging) if foe is not None else False,
                    must_recharge=bool(foe.must_recharge) if foe is not None else False,
                )
                if gated is not None and priority_move_fails(
                    move_id,
                    attacker=foe,
                    defender=our,
                    defender_side=our_side,
                    terrain=ctx.terrain,
                    gravity=ctx.gravity,
                ):
                    gated = None
                opp_hit_map[key] = gated

            foe_types = [tuple(f.types or ()) for _, f in world_foes]

            for pl, pr in pairs:
                if _elapsed_ms(t0) >= cfg.phase2_end_ms:
                    break

                offense, foe_hp, p_ko_foe, share = _our_pair_offense_detail(
                    pl, pr, world_foes, hit_map, soft_ko_bonus=cfg.soft_ko_bonus,
                )
                threat, p_ko_side, _pct_side = _opp_threat_detail(
                    left_atk, right_atk, pl, pr, opp_hit_map, opp_dists,
                    soft_ko_bonus=cfg.soft_ko_bonus,
                )

                # B: orden de turno
                offense, threat = _apply_speed_race(
                    offense=offense,
                    threat=threat,
                    share=share,
                    p_ko_foe=p_ko_foe,
                    p_ko_side=p_ko_side,
                    left=pl,
                    right=pr,
                    left_atk=left_atk,
                    right_atk=right_atk,
                    world_foes=world_foes,
                    pos=pos,
                    terrain=terrain or ctx.terrain,
                )

                # C: switches voluntarios
                switch_v = 0.0
                for atom, side_key in ((pl, 'OL'), (pr, 'OR')):
                    if atom.utility != 'switch':
                        continue
                    res = reserve_by_slot.get(int(atom.move_slot))
                    if res is None:
                        continue
                    stay_pk = float(p_ko_side.get(side_key, 0.0))
                    # Si no hay amenaza calc (dists vacíos), estimar por tipado del activo
                    if stay_pk < 1e-9 and side_key == 'OL' and left_atk is not None:
                        mu = matchup_vs_actives(left_atk.types, foe_types or [tuple()])
                        stay_pk = max(0.0, min(0.9, 0.15 - mu * 0.08))
                    if stay_pk < 1e-9 and side_key == 'OR' and right_atk is not None:
                        mu = matchup_vs_actives(right_atk.types, foe_types or [tuple()])
                        stay_pk = max(0.0, min(0.9, 0.15 - mu * 0.08))
                    if stay_pk < cfg.switch_min_stay_p_ko and stay_pk < 0.15:
                        # Matchup horrible también justifica mirar switch
                        mu_b = matchup_vs_actives(res.types or res.poke.types, foe_types or [tuple()])
                        if mu_b < 0.5 and stay_pk < 0.1:
                            switch_v -= 40.0  # switch frívolo
                            continue
                    chip = _bench_incoming_pct(res, world_foes, opp_hit_map, all_opp_move_names)
                    mu = matchup_vs_actives(res.types or res.poke.types, foe_types or [tuple()])
                    switch_v += switch_preservation_value(
                        stay_p_ko=max(stay_pk, cfg.switch_min_stay_p_ko * 0.5),
                        bench_incoming_pct=chip,
                        matchup=mu,
                        loss_weight=cfg.loss_weight,
                        save_weight=cfg.save_weight,
                        chip_soft_cap=cfg.chip_soft_cap,
                        soft_ko_bonus=cfg.soft_ko_bonus,
                    )

                syn = 0.0
                if pl.utility != 'switch' and pr.utility != 'switch':
                    try:
                        syn = _score_pair(pl, pr, world_foes) - (
                            pl.total_damage_pct + pr.total_damage_pct
                        )
                        if syn > cfg.soft_ko_bonus:
                            syn = min(syn, 80.0)
                    except Exception:
                        syn = 0.0

                used_sc = _speed_control(pl.move_name) or _speed_control(pr.move_name)
                used_tera = pl.mega == 2 or pr.mega == 2
                field_b = field_control_bonus(pos, used_speed_control=used_sc, used_tera=used_tera)

                our_hp = max(0.0, 100.0 - min(threat / max(cfg.soft_ko_bonus / 100.0, 1.0), 100.0))
                v = (
                    offense - 0.85 * threat + 0.15 * (our_hp - foe_hp)
                    + 0.35 * syn + field_b + switch_v
                )

                utilities[_pair_key(pl, pr)].add(v, weight=p_world)

            meta.pairs_scored = len(utilities)

        best_key = None
        best_mean = float('-inf')
        for key, ju in utilities.items():
            if ju.samples <= 0:
                continue
            m = ju.mean
            if m > best_mean:
                best_mean = m
                best_key = key

        if best_key is not None and best_mean > float('-inf'):
            for pl, pr in pairs:
                if _pair_key(pl, pr) == best_key:
                    best_l = _scored_from_atom(pl, best_mean, foes)
                    best_r = _scored_from_atom(pr, best_mean, foes)
                    best_u = best_mean
                    meta.used_fallback = False
                    meta.phase = 'phase2'
                    meta.utility = best_u
                    break

    except Exception as exc:
        meta.errors.append(f'phase2:{exc!r}')

    meta.elapsed_ms = _elapsed_ms(t0)
    if meta.elapsed_ms >= cfg.deadline_ms:
        meta.phase = meta.phase + '+timeout'
    return best_l, best_r, best_u, meta
