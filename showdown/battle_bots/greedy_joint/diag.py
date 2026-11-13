"""Diagnóstico C: composición de pares elegidos por greedy_joint (wins vs losses).

No cambia la política. Solo cuenta tipos de par en turnos de batalla.
Escritura: logs/greedy_joint_pair_diag.txt (vía dump al final del robust E2E).
"""
from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from showdown.battle_bots.greedy_joint.joint import JointAtom

# Último kind del turno (1 batalla concurrente por proceso).
LAST_KIND: str = 'none'

_LOCK = threading.Lock()
_TURNS_WIN: Counter = Counter()
_TURNS_LOSS: Counter = Counter()
_BATTLES_WIN = 0
_BATTLES_LOSS = 0

KIND_ORDER = (
    'double_attack_focus',
    'double_attack_split',
    'protect_hit',
    'fakeout',
    'helpinghand',
    'redirect',
    'speed',
    'damage_mix',
    'other_support',
    'single_side',
    'none',
)


def classify_pair(
    left: Optional['JointAtom'],
    right: Optional['JointAtom'],
) -> str:
    """Clasifica el par elegido (utilities + foco)."""
    if left is None and right is None:
        return 'none'
    if left is None or right is None:
        return 'single_side'

    lu = str(left.utility or 'damage')
    ru = str(right.utility or 'damage')
    utils = {lu, ru}
    l_dmg = float(left.total_damage_pct) > 0.0
    r_dmg = float(right.total_damage_pct) > 0.0

    if 'protect' in utils and (l_dmg or r_dmg):
        return 'protect_hit'
    if 'redirect' in utils:
        return 'redirect'
    if 'fakeout' in utils:
        return 'fakeout'
    if 'helpinghand' in utils:
        return 'helpinghand'
    if 'speed' in utils:
        return 'speed'
    if lu == 'damage' and ru == 'damage':
        if int(left.target) == int(right.target) and int(left.target) != 0:
            return 'double_attack_focus'
        return 'double_attack_split'
    if l_dmg or r_dmg:
        return 'damage_mix'
    return 'other_support'


def note_turn_kind(kind: str) -> None:
    global LAST_KIND
    LAST_KIND = str(kind or 'none')


def record_battle(won: Optional[bool], kinds: Counter) -> None:
    """Acumula turnos de una batalla terminada."""
    global _BATTLES_WIN, _BATTLES_LOSS
    if won is None or not kinds:
        return
    with _LOCK:
        if won:
            _BATTLES_WIN += 1
            _TURNS_WIN.update(kinds)
        else:
            _BATTLES_LOSS += 1
            _TURNS_LOSS.update(kinds)


def reset() -> None:
    global _BATTLES_WIN, _BATTLES_LOSS, LAST_KIND
    with _LOCK:
        _TURNS_WIN.clear()
        _TURNS_LOSS.clear()
        _BATTLES_WIN = 0
        _BATTLES_LOSS = 0
        LAST_KIND = 'none'


def _pct(n: int, total: int) -> str:
    if total <= 0:
        return '-'
    return f'{100.0 * n / total:.1f}%'


def format_report() -> str:
    with _LOCK:
        tw = Counter(_TURNS_WIN)
        tl = Counter(_TURNS_LOSS)
        bw, bl = _BATTLES_WIN, _BATTLES_LOSS
    total_w = sum(tw.values())
    total_l = sum(tl.values())
    lines = [
        '=== Greedy-Joint pair diagnosis (proposal C) ===',
        'policy: A5+B-lite+field Spe+A9 + Spe PE (open belief); telemetry only',
        f'battles: W={bw} L={bl}',
        f'turns:   W={total_w} L={total_l}',
        '',
        f'{"kind":<24} {"win_n":>7} {"win_%":>7} {"loss_n":>7} {"loss_%":>7} {"d_pp":>7}',
        '-' * 64,
    ]
    kinds = list(KIND_ORDER)
    for k in sorted(set(tw) | set(tl)):
        if k not in kinds:
            kinds.append(k)
    for k in kinds:
        wn, ln = int(tw.get(k, 0)), int(tl.get(k, 0))
        if wn == 0 and ln == 0:
            continue
        wp = (100.0 * wn / total_w) if total_w else 0.0
        lp = (100.0 * ln / total_l) if total_l else 0.0
        lines.append(
            f'{k:<24} {wn:7d} {wp:6.1f}% {ln:7d} {lp:6.1f}% {wp - lp:+6.1f}'
        )
    lines.append('')
    lines.append(
        'd_pp = win_% - loss_% (positivo → más frecuente en victorias).'
    )
    lines.append(
        'Interpretar: protect_hit/redirect/fakeout altos en losses → support caro; '
        'double_attack_* dominante en ambos → poca coordinación distinta de solo.'
    )
    return '\n'.join(lines) + '\n'


def dump(path: Optional[Path] = None) -> Path:
    out = path or (
        Path(__file__).resolve().parents[3] / 'logs' / 'greedy_joint_pair_diag.txt'
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    text = format_report()
    out.write_text(text, encoding='utf-8')
    return out
