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

_REDIRECT_BASE = 8.0
_REDIRECT_WITH_ATTACK = 40.0
_REDIRECT_WITH_KO = 70.0
_HIGH_DAMAGE_PCT = 45.0


def _norm_move(name: str) -> str:
    return str(name or '').strip().lower().replace(' ', '').replace('-', '')


def _utility_kind(move: MoveOption) -> str:
    mid = _norm_move(move.name)
    if mid in (
        'protect', 'detect', 'spikyshield', 'banefulbunker', 'silktrap',
        'burningbulwark', 'obstruct',
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
) -> List[JointAtom]:
    atoms: List[JointAtom] = []
    base_ko: Dict[Tuple[int, int], bool] = {}
    for (_side, move, tid, foe, special), rng in zip(meta, ranges):
        max_hp = int((_stats_or_default(foe).get('hp') or 1))
        cur_hp = int(foe.cur_hp if foe.cur_hp is not None else max_hp)
        pct = damage_pct(rng, max_hp)
        ko = guaranteed_ko(rng, cur_hp) and not foe.has_substitute
        max_r = int(rng[1]) if rng else 0
        key = (move.slot, tid)
        if special == 0:
            base_ko[key] = ko
            atoms.append(JointAtom(
                move_slot=move.slot,
                target=tid,
                mega=1 if can_mega else 0,
                damage_pct={tid: pct},
                max_roll={tid: max_r},
                ko_tids={tid} if ko else set(),
                utility='damage',
                move_name=move.name,
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
            ))

    for move in moves:
        kind = _utility_kind(move)
        if kind == 'damage':
            continue
        atoms.append(JointAtom(
            move_slot=move.slot,
            target=0,
            mega=1 if can_mega else 0,
            damage_pct={},
            max_roll={},
            ko_tids=set(),
            utility=kind,
            move_name=move.name,
        ))
    return atoms


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
) -> float:
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
            s += _FAKE_OUT_BONUS
        elif atom.utility == 'helpinghand':
            s += _HELPING_HAND_BONUS if has_damage else _STATUS_BASE
        elif atom.utility == 'speed':
            s += _SPEED_CONTROL_BONUS
        elif atom.utility == 'status':
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

    ranges = await calc_damage_batch(requests) if requests else []

    left_meta = [m for m in meta if m[0] == 'L']
    right_meta = [m for m in meta if m[0] == 'R']
    left_ranges = [rng for m, rng in zip(meta, ranges) if m[0] == 'L']
    right_ranges = [rng for m, rng in zip(meta, ranges) if m[0] == 'R']

    left_atoms = (
        _atoms_from_side(left_moves, left_meta, left_ranges, foes, can_mega=can_mega_left)
        if left_atk and left_moves else []
    )
    right_atoms = (
        _atoms_from_side(right_moves, right_meta, right_ranges, foes, can_mega=can_mega_right)
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
            s = _score_pair(pair_l, pair_r, foes)
            if s > best_score:
                best_score = s
                best_pair = (pair_l, pair_r)

    assert best_pair is not None
    bl, br = best_pair
    return _atom_to_scored(bl), _atom_to_scored(br), best_score
