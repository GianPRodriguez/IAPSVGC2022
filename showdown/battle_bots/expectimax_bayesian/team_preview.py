"""Team Preview propio de expectimax: tipado + roles de campo + calidad de lead.

Contrato: docs/expectimax_bayesian_spec.md §3 (Team Preview).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from showdown.battle_bots.greedy_solo.team_preview import (
    PreviewMon,
    PreviewScore,
    score_combination,
)
from teams.team_converter import normalize_name

# --- Pesos tipado/BST (≠ greedy α=β=1, γ=0.01) ---
PREVIEW_ALPHA: float = 1.05
PREVIEW_BETA: float = 1.25
PREVIEW_GAMMA: float = 0.008
LEAD_QUALITY_WEIGHT: float = 0.45
LEAD_DEF_SCALE: float = 1.35
LEAD_OFF_SCALE: float = 1.15
LEAD_ROLE_BONUS: float = 10.0

# --- Bonos de rol (presencia ≥1 especie del set; no apilan por cantidad) ---
ROLE_BONUS_REDIRECT: float = 28.0
ROLE_BONUS_TAILWIND: float = 32.0
ROLE_BONUS_TRICK_ROOM: float = 26.0
ROLE_BONUS_SPEED_CONTROL: float = 18.0
ROLE_BONUS_FAKE_OUT: float = 14.0
ROLE_DIVERSITY_BONUS: float = 12.0  # si n_role_buckets ≥ 2

# Roles por especie (ids normalize_name). Solo lo visible en preview — no OTS/moves.
REDIRECT_SPECIES: Set[str] = frozenset({
    'amoonguss', 'indeedee', 'indeedeef', 'sableye', 'clefairy',
})
TAILWIND_SPECIES: Set[str] = frozenset({
    'tornadus', 'whimsicott', 'kilowattrel', 'suicune', 'pelipper', 'emolga',
    'talonflame', 'braviary', 'braviaryhisui', 'zapdos',
})
TRICK_ROOM_SPECIES: Set[str] = frozenset({
    'indeedee', 'indeedeef', 'hatenna', 'hattrem', 'hatterene', 'slowking',
    'slowkinggalar', 'porygon2', 'dusknoir', 'bronzong', 'oranguru',
})
SPEED_CONTROL_SPECIES: Set[str] = frozenset({
    'incineroar', 'rillaboom', 'urshifu', 'urshifurapidstrike', 'landorustherian',
    'landorus', 'tinglu', 'greattusk',
})
FAKE_OUT_SPECIES: Set[str] = frozenset({
    'incineroar', 'scrafty', 'kangaskhan', 'persian', 'meowth', 'sneasler',
    'baxcalibur', 'farigiraf',
})

# Aliases internos (tests / spec); mismos objetos frozenset.
_REDIRECT = REDIRECT_SPECIES
_TAILWIND = TAILWIND_SPECIES
_TRICK_ROOM = TRICK_ROOM_SPECIES
_SPEED_CONTROL = SPEED_CONTROL_SPECIES
_FAKE_OUT = FAKE_OUT_SPECIES


def _nid(name: str) -> str:
    """Id de rol: normalize_name + quitar '-' / '_' (Showdown deja formes con guión)."""
    return (normalize_name(name or '') or '').replace('-', '').replace('_', '')


def role_bucket_flags(ids: Set[str]) -> Dict[str, bool]:
    """Tres buckets de diversidad: redirect | speed-axis (TW∨TR) | disrupt (spdCtrl∨FO)."""
    return {
        'redirect': bool(ids & _REDIRECT),
        'speed_axis': bool(ids & _TAILWIND) or bool(ids & _TRICK_ROOM),
        'disrupt': bool(ids & _SPEED_CONTROL) or bool(ids & _FAKE_OUT),
    }


def _role_bonus(mons: Sequence[PreviewMon]) -> float:
    ids = {_nid(m.name) for m in mons}
    s = 0.0
    if ids & _REDIRECT:
        s += ROLE_BONUS_REDIRECT
    if ids & _TAILWIND:
        s += ROLE_BONUS_TAILWIND
    if ids & _TRICK_ROOM:
        s += ROLE_BONUS_TRICK_ROOM
    if ids & _SPEED_CONTROL:
        s += ROLE_BONUS_SPEED_CONTROL
    if ids & _FAKE_OUT:
        s += ROLE_BONUS_FAKE_OUT
    buckets = role_bucket_flags(ids)
    if sum(1 for v in buckets.values() if v) >= 2:
        s += ROLE_DIVERSITY_BONUS
    return s


def _role_tags(mons: Sequence[PreviewMon]) -> str:
    ids = {_nid(m.name) for m in mons}
    tags: List[str] = []
    if ids & _REDIRECT:
        tags.append('redirect')
    if ids & _TAILWIND:
        tags.append('TW')
    if ids & _TRICK_ROOM:
        tags.append('TR')
    if ids & _SPEED_CONTROL:
        tags.append('spdCtrl')
    if ids & _FAKE_OUT:
        tags.append('FO')
    return '+'.join(tags) if tags else '—'


def _lead_quality(leads: Sequence[PreviewMon], foes: Sequence[PreviewMon]) -> float:
    """Lead debe tipar bien ya; banca puede ser el pivot."""
    from showdown.battle_bots.greedy_solo.type_chart import (
        defensive_resistance_score,
        offensive_coverage_score,
    )

    foe_types = [m.types for m in foes if m.types] or [tuple()]
    s = 0.0
    for mon in leads:
        s += LEAD_DEF_SCALE * defensive_resistance_score(mon.types, foe_types)
        s += LEAD_OFF_SCALE * offensive_coverage_score(mon.types, foe_types)
        nid = _nid(mon.name)
        if nid in _REDIRECT or nid in _FAKE_OUT or nid in _TAILWIND:
            s += LEAD_ROLE_BONUS
    return s


def _short_names(mons: Sequence[PreviewMon], limit: int = 4) -> str:
    parts = []
    for m in list(mons)[:limit]:
        raw = str(m.name or '?').partition(',')[0].strip()
        parts.append(raw.replace(' ', '')[:12] or '?')
    return ','.join(parts)


@dataclass
class PreviewDecisionMeta:
    """Reporte CMD: por qué esos 4 de 6 (+ lead)."""

    chosen: Optional[PreviewScore] = None
    ranked_lines: List[str] = field(default_factory=list)
    why_lines: List[str] = field(default_factory=list)
    n_combos_4: int = 0
    n_lead_orders: int = 0
    n_scored: int = 0


def best_expectimax_preview_order(
    our_mons: Sequence[PreviewMon],
    foe_mons: Sequence[PreviewMon],
    *,
    alpha: float = PREVIEW_ALPHA,
    beta: float = PREVIEW_BETA,
    gamma: float = PREVIEW_GAMMA,
    top_report: int = 8,
) -> PreviewScore:
    """Wrapper: devuelve PreviewScore; meta vía best_expectimax_preview_with_meta."""
    scored, _meta = best_expectimax_preview_with_meta(
        our_mons, foe_mons, alpha=alpha, beta=beta, gamma=gamma, top_report=top_report,
    )
    return scored


def best_expectimax_preview_with_meta(
    our_mons: Sequence[PreviewMon],
    foe_mons: Sequence[PreviewMon],
    *,
    alpha: float = PREVIEW_ALPHA,
    beta: float = PREVIEW_BETA,
    gamma: float = PREVIEW_GAMMA,
    top_report: int = 8,
) -> Tuple[PreviewScore, PreviewDecisionMeta]:
    """
    Exhaustivo C(6,4)×C(4,2) — no hay Top-K.
    score = alpha*off + beta*def + gamma*BST + role_bonus + LEAD_QUALITY_WEIGHT*lead_quality
    """
    meta = PreviewDecisionMeta()
    pool = list(our_mons)
    if len(pool) < 4:
        idxs = [m.index for m in pool] + list(range(1, 7))
        uniq: List[int] = []
        for i in idxs:
            if i not in uniq:
                uniq.append(i)
            if len(uniq) == 4:
                break
        while len(uniq) < 4:
            uniq.append(len(uniq) + 1)
        scored = PreviewScore(tuple(uniq[:4]), 0.0, 0.0, 0.0, 0.0)  # type: ignore[arg-type]
        meta.chosen = scored
        meta.why_lines = ['pool < 4; orden por defecto']
        return scored, meta

    ranked: List[Tuple[float, PreviewScore, float, float, str, str]] = []
    # (total, PreviewScore, role, lead_q, role_tags, lead_names)
    for combo in itertools.combinations(pool, 4):
        combo_list = list(combo)
        role = _role_bonus(combo_list)
        tags = _role_tags(combo_list)
        meta.n_combos_4 += 1
        for lead in itertools.combinations(combo_list, 2):
            meta.n_lead_orders += 1
            bench = [m for m in combo_list if m not in lead]
            ordered = list(lead) + bench
            base = score_combination(
                ordered, foe_mons, alpha=alpha, beta=beta, gamma=gamma,
            )
            lead_q = _lead_quality(list(lead), foe_mons)
            total = base.score + role + LEAD_QUALITY_WEIGHT * lead_q
            ps = PreviewScore(
                order=base.order,
                offensive=base.offensive,
                defensive=base.defensive,
                bst=base.bst,
                score=total,
            )
            ranked.append((
                total, ps, role, lead_q, tags, _short_names(list(lead)),
            ))

    ranked.sort(key=lambda t: (-t[0], -t[1].bst))
    meta.n_scored = len(ranked)
    best = ranked[0][1]
    meta.chosen = best

    lines: List[str] = []
    for i, (total, ps, role, lead_q, tags, lead_names) in enumerate(ranked[: max(1, top_report)], 1):
        mark = '>' if i == 1 else ' '
        names = []
        by_idx = {m.index: m for m in pool}
        for idx in ps.order:
            m = by_idx.get(idx)
            names.append((m.name if m else f'#{idx}').partition(',')[0].replace(' ', '')[:10])
        lines.append(
            f'{mark}{i:2d}. S={total:7.1f}  '
            f'[{",".join(names)}]  lead={lead_names}  '
            f'off={ps.offensive:.0f} def={ps.defensive:.0f}  '
            f'roles={tags}(+{role:.0f}) leadQ={lead_q:.0f}'
        )
    meta.ranked_lines = lines

    second = ranked[1] if len(ranked) > 1 else None
    why: List[str] = [
        f'criterio: exhaustivo C(6,4)xC(4,2) = {meta.n_scored} ordenes '
        f'(no Top-K); alpha={alpha} beta={beta} gamma={gamma}',
        f'score = alpha*off + beta*def + gamma*BST + roles + '
        f'{LEAD_QUALITY_WEIGHT}*leadQ',
        f'elegido S={ranked[0][0]:.1f}  roles={ranked[0][4]}(+{ranked[0][2]:.0f})  '
        f'leadQ={ranked[0][3]:.0f}  off={best.offensive:.0f} def={best.defensive:.0f} '
        f'bst={best.bst:.0f}',
    ]
    if second is not None:
        why.append(
            f'vs 2º S={second[0]:.1f}  margen={ranked[0][0] - second[0]:+.1f}  '
            f'roles={second[4]}'
        )
    meta.why_lines = why
    return best, meta
