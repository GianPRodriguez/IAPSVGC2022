"""Política del oponente π_opp: mezcla greedy-foe / control / uniforme."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from showdown.battle_bots.greedy_solo.greedy import MoveOption

# Pesos Gemini (renormalizados si falta alguna clase)
W_GREEDY = 0.70
W_CONTROL = 0.20
W_UNIFORM = 0.10

_CONTROL_MOVES = frozenset({
    'protect', 'detect', 'spikyshield', 'banefulbunker', 'silktrap',
    'burningbulwark', 'obstruct', 'substitute', 'followme', 'ragepowder',
    'spotlight', 'wideguard', 'quickguard', 'helpinghand', 'craftyshield',
})


def _norm_move(name: str) -> str:
    return str(name or '').strip().lower().replace(' ', '').replace('-', '')


def is_control_move(name: str) -> bool:
    return _norm_move(name) in _CONTROL_MOVES


@dataclass(frozen=True)
class OppAction:
    """Acción atómica del rival (un mon)."""

    move_name: str
    target: int  # 1/2 nuestros slots Showdown, 0 = none/self
    kind: str  # 'greedy' | 'control' | 'other'


def move_options_from_names(
    names: Sequence[str],
    *,
    start_slot: int = 1,
) -> List[MoveOption]:
    out: List[MoveOption] = []
    for i, name in enumerate(names):
        if not name:
            continue
        out.append(MoveOption(
            slot=start_slot + i,
            name=str(name),
            target_type='normal',
            category='Physical',
        ))
    return out


def classify_opp_moves(
    moves: Sequence[MoveOption],
    *,
    damage_by_slot: Optional[Dict[int, float]] = None,
) -> Dict[str, List[MoveOption]]:
    """Particiona moves en greedy / control / other."""
    dmg = damage_by_slot or {}
    greedy: List[MoveOption] = []
    control: List[MoveOption] = []
    other: List[MoveOption] = []
    for m in moves:
        if is_control_move(m.name):
            control.append(m)
            continue
        score = float(dmg.get(m.slot, -1.0))
        if score > 0.0:
            greedy.append(m)
        elif score == 0.0:
            other.append(m)
        else:
            # Sin scores: no-control → greedy (política agresiva por defecto)
            greedy.append(m)
    if not greedy:
        greedy = [m for m in moves if not is_control_move(m.name)]
    if not other:
        other = list(moves)
    return {'greedy': greedy, 'control': control, 'other': other}


def opp_policy_distribution(
    moves: Sequence[MoveOption],
    *,
    damage_by_slot: Optional[Dict[int, float]] = None,
    default_target: int = 1,
) -> List[Tuple[OppAction, float]]:
    """
    π_opp(b): 70% masa en moves de daño (proporcional a daño),
    20% en control (uniforme), 10% uniforme global.
    """
    if not moves:
        return []

    parts = classify_opp_moves(moves, damage_by_slot=damage_by_slot)
    dmg = damage_by_slot or {}

    def _uniform(ms: Sequence[MoveOption], kind: str) -> List[Tuple[OppAction, float]]:
        if not ms:
            return []
        p = 1.0 / float(len(ms))
        return [
            (OppAction(move_name=m.name, target=default_target, kind=kind), p)
            for m in ms
        ]

    # Greedy: softmax-ish proporcial a daño (o uniforme si todos 0)
    greedy_moves = parts['greedy']
    greedy_dist: List[Tuple[OppAction, float]] = []
    if greedy_moves:
        weights = [max(0.0, float(dmg.get(m.slot, 0.0))) for m in greedy_moves]
        if sum(weights) <= 1e-9:
            greedy_dist = _uniform(greedy_moves, 'greedy')
        else:
            total = sum(weights)
            for m, w in zip(greedy_moves, weights):
                greedy_dist.append((
                    OppAction(m.name, default_target, 'greedy'),
                    w / total,
                ))

    control_dist = _uniform(parts['control'], 'control')
    uniform_dist = _uniform(moves, 'other')

    # Mezcla
    buckets: List[Tuple[List[Tuple[OppAction, float]], float]] = []
    if greedy_dist:
        buckets.append((greedy_dist, W_GREEDY))
    if control_dist:
        buckets.append((control_dist, W_CONTROL))
    if uniform_dist:
        buckets.append((uniform_dist, W_UNIFORM))
    if not buckets:
        return _uniform(moves, 'other')

    w_sum = sum(w for _, w in buckets) or 1.0
    merged: Dict[Tuple[str, int], float] = {}
    kind_of: Dict[Tuple[str, int], str] = {}
    for dist, w_bucket in buckets:
        scale = (w_bucket / w_sum)
        for action, p in dist:
            key = (_norm_move(action.move_name), int(action.target))
            merged[key] = merged.get(key, 0.0) + scale * float(p)
            kind_of[key] = action.kind

    out = [
        (OppAction(move_name=k[0], target=k[1], kind=kind_of[k]), p)
        for k, p in merged.items()
    ]
    # Restore readable move names from originals
    name_by_norm = {_norm_move(m.name): m.name for m in moves}
    out = [
        (OppAction(name_by_norm.get(a.move_name, a.move_name), a.target, a.kind), p)
        for a, p in out
    ]
    total = sum(p for _, p in out) or 1.0
    return [(a, p / total) for a, p in out]
