"""Política del oponente π_opp: mezcla greedy-foe / control / uniforme."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from showdown.battle_bots.greedy_solo.greedy import MoveOption

# Pesos π_opp: greedy dominante pero con masa de control (no hiperagresivo puro).
# Overrideables vía ExpectimaxConfig (validación vs greedy_joint).
W_GREEDY = 0.75
W_CONTROL = 0.15
W_UNIFORM = 0.10
W_SWITCH = 0.10

# Acción sintética: pivot rival (0 dmg este turno).
SWITCH_MOVE_NAME = '__switch__'

# Diagnóstico (opp_model_trace): recibe {'moves': [...], 'merged': {(move, target, kind): p}}.
_PRED_SINK = None


def set_pred_sink(fn) -> None:
    global _PRED_SINK
    _PRED_SINK = fn


def emit_pred(move_names, merged, norm: str = 'global') -> None:
    """Publica una predicción al diagnóstico (no-op si no hay sink)."""
    if _PRED_SINK is None:
        return
    try:
        _PRED_SINK({'moves': list(move_names), 'merged': dict(merged), 'norm': norm})
    except Exception:
        pass

_CONTROL_MOVES = frozenset({
    'protect', 'detect', 'spikyshield', 'banefulbunker', 'silktrap',
    'burningbulwark', 'obstruct', 'substitute', 'followme', 'ragepowder',
    'spotlight', 'wideguard', 'quickguard', 'helpinghand', 'craftyshield',
})


def _norm_move(name: str) -> str:
    return str(name or '').strip().lower().replace(' ', '').replace('-', '')


def is_control_move(name: str) -> bool:
    return _norm_move(name) in _CONTROL_MOVES


def is_switch_action(action: 'OppAction') -> bool:
    return action.kind == 'switch' or _norm_move(action.move_name) == _norm_move(SWITCH_MOVE_NAME)


@dataclass(frozen=True)
class OppAction:
    """Acción atómica del rival (un mon)."""

    move_name: str
    target: int  # 1/2 nuestros slots Showdown, 0 = none/self/switch
    kind: str  # 'greedy' | 'control' | 'other' | 'switch'


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
    w_greedy: float = W_GREEDY,
    w_control: float = W_CONTROL,
    w_uniform: float = W_UNIFORM,
) -> List[Tuple[OppAction, float]]:
    """
    π_opp(b): masa en moves de daño (proporcional a daño),
    control (uniforme), y uniforme global.

    `damage_by_slot` debe ser daño real (post-calc). Si todos ≈1, la política
    es casi uniforme entre ataques (comportamiento legacy / pre-calc).
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

    # Greedy: proporcial a daño (o uniforme si todos 0)
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

    # Mezcla (renormaliza si falta algún bucket)
    buckets: List[Tuple[List[Tuple[OppAction, float]], float]] = []
    wg, wc, wu = float(w_greedy), float(w_control), float(w_uniform)
    if greedy_dist and wg > 0:
        buckets.append((greedy_dist, wg))
    if control_dist and wc > 0:
        buckets.append((control_dist, wc))
    if uniform_dist and wu > 0:
        buckets.append((uniform_dist, wu))
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


def opp_policy_targeted(
    moves: Sequence[MoveOption],
    *,
    damage_by_slot_target: Dict[int, Dict[int, float]],
    available_targets: Sequence[int] = (1, 2),
    w_greedy: float = W_GREEDY,
    w_control: float = W_CONTROL,
    w_uniform: float = W_UNIFORM,
    w_switch: float = W_SWITCH,
) -> Dict[str, List[Tuple[OppAction, float]]]:
    """π_opp con targets reales + masa de switch.

    - Greedy: masa ∝ daño por (move, target); no se duplica entre OL/OR.
    - Control / switch: target=0 (0 dmg); restan masa de ataque.
    - Retorna {'OL': [...], 'OR': [...]} con probs absolutas (suma global ≤ 1).
      Solo ataques con target acorde al lado; control/switch no van a las listas
      de amenaza (su masa ya se restó del bucket greedy).
    """
    out: Dict[str, List[Tuple[OppAction, float]]] = {'OL': [], 'OR': []}
    if not moves:
        return out

    tgts = [int(t) for t in available_targets if int(t) in (1, 2)]
    if not tgts:
        tgts = [1, 2]
    side_of = {1: 'OL', 2: 'OR'}

    parts = classify_opp_moves(
        moves,
        damage_by_slot={
            m.slot: max(
                (float(damage_by_slot_target.get(m.slot, {}).get(t, 0.0)) for t in tgts),
                default=0.0,
            )
            for m in moves
        },
    )

    # --- greedy: (move, target) weights ---
    greedy_weighted: List[Tuple[OppAction, float]] = []
    for m in parts['greedy']:
        per_t = damage_by_slot_target.get(m.slot) or {}
        for t in tgts:
            w = max(0.0, float(per_t.get(t, 0.0)))
            if w > 0.0:
                greedy_weighted.append((OppAction(m.name, t, 'greedy'), w))
        # Sin daño a ningún target: reparte uniforme entre targets vivos
        if not any(float((per_t or {}).get(t, 0.0)) > 0.0 for t in tgts):
            share = 1.0 / float(len(tgts))
            for t in tgts:
                greedy_weighted.append((OppAction(m.name, t, 'greedy'), share))

    greedy_dist: List[Tuple[OppAction, float]] = []
    if greedy_weighted:
        tot = sum(w for _, w in greedy_weighted) or 1.0
        greedy_dist = [(a, w / tot) for a, w in greedy_weighted]

    control_moves = parts['control']
    control_dist: List[Tuple[OppAction, float]] = []
    if control_moves:
        p = 1.0 / float(len(control_moves))
        control_dist = [
            (OppAction(m.name, 0, 'control'), p) for m in control_moves
        ]

    # Uniform over damaging (move, target) pairs (no control)
    uniform_pairs: List[OppAction] = []
    for m in moves:
        if is_control_move(m.name):
            continue
        for t in tgts:
            uniform_pairs.append(OppAction(m.name, t, 'other'))
    uniform_dist: List[Tuple[OppAction, float]] = []
    if uniform_pairs:
        p = 1.0 / float(len(uniform_pairs))
        uniform_dist = [(a, p) for a in uniform_pairs]

    switch_dist = [(OppAction(SWITCH_MOVE_NAME, 0, 'switch'), 1.0)]

    buckets: List[Tuple[List[Tuple[OppAction, float]], float]] = []
    wg, wc, wu, ws = float(w_greedy), float(w_control), float(w_uniform), float(w_switch)
    # Renormaliza si falta algún bucket
    if greedy_dist and wg > 0:
        buckets.append((greedy_dist, wg))
    if control_dist and wc > 0:
        buckets.append((control_dist, wc))
    if uniform_dist and wu > 0:
        buckets.append((uniform_dist, wu))
    if ws > 0:
        buckets.append((switch_dist, ws))
    if not buckets:
        # Fallback: ataque uniforme por target
        for t in tgts:
            side = side_of[t]
            n = max(1, len([m for m in moves if not is_control_move(m.name)]))
            for m in moves:
                if is_control_move(m.name):
                    continue
                out[side].append((OppAction(m.name, t, 'other'), 1.0 / (n * len(tgts))))
        return out

    w_sum = sum(w for _, w in buckets) or 1.0
    merged: Dict[Tuple[str, int, str], float] = {}
    for dist, w_bucket in buckets:
        scale = w_bucket / w_sum
        for action, p in dist:
            key = (_norm_move(action.move_name), int(action.target), action.kind)
            merged[key] = merged.get(key, 0.0) + scale * float(p)

    if _PRED_SINK is not None:
        try:
            _PRED_SINK({'moves': [m.name for m in moves], 'merged': dict(merged)})
        except Exception:
            pass

    name_by_norm = {_norm_move(m.name): m.name for m in moves}
    name_by_norm[_norm_move(SWITCH_MOVE_NAME)] = SWITCH_MOVE_NAME

    for (mid, tgt, kind), p in merged.items():
        if kind in ('control', 'switch') or tgt == 0:
            continue  # masa ya restada; no aportan dmg a OL/OR
        side = side_of.get(int(tgt))
        if side is None:
            continue
        out[side].append((
            OppAction(name_by_norm.get(mid, mid), int(tgt), kind),
            float(p),
        ))

    return out
