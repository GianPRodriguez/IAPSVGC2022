"""Team Preview propio de expectimax: tipado + roles de campo + calidad de lead."""
from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Set, Tuple

from showdown.battle_bots.greedy_solo.team_preview import (
    PreviewMon,
    PreviewScore,
    score_combination,
)
from teams.team_converter import normalize_name

# Roles por especie (Champions / VGC frecuente). No usa OTS: solo lo visible en preview.
_REDIRECT: Set[str] = {
    'amoonguss', 'indeedee', 'indeedeef', 'sableye', 'clefairy', 'riesnther',
}
_TAILWIND: Set[str] = {
    'tornadus', 'whimsicott', 'kilowattrel', 'suicune', 'pelipper', 'emolga',
    'talonflame', 'braviary', 'braviaryhisui', 'zapdos',
}
_TRICK_ROOM: Set[str] = {
    'indeedee', 'indeedeef', 'hatenna', 'hattrem', 'hatterene', 'slowking',
    'slowkinggalar', 'porygon2', 'dusknoir', 'bronzong', 'oranguru',
}
_SPEED_CONTROL: Set[str] = {
    'incineroar', 'rillaboom', 'urshifu', 'urshifurapidstrike', 'landorustherian',
    'landorus', 'tinglu', 'greattusk',
}
_FAKE_OUT: Set[str] = {
    'incineroar', 'scrafty', 'kangaskhan', 'persian', 'meowth', 'sneasler',
    'baxcalibur', 'farigiraf',
}


def _nid(name: str) -> str:
    return normalize_name(name or '') or ''


def _role_bonus(mons: Sequence[PreviewMon]) -> float:
    ids = {_nid(m.name) for m in mons}
    s = 0.0
    if ids & _REDIRECT:
        s += 28.0
    if ids & _TAILWIND:
        s += 32.0
    if ids & _TRICK_ROOM:
        s += 26.0
    if ids & _SPEED_CONTROL:
        s += 18.0
    if ids & _FAKE_OUT:
        s += 14.0
    # Diversidad de roles (evitar 4 attackers sin soporte)
    n_roles = sum([
        bool(ids & _REDIRECT),
        bool(ids & _TAILWIND) or bool(ids & _TRICK_ROOM),
        bool(ids & _SPEED_CONTROL) or bool(ids & _FAKE_OUT),
    ])
    if n_roles >= 2:
        s += 12.0
    return s


def _lead_quality(leads: Sequence[PreviewMon], foes: Sequence[PreviewMon]) -> float:
    """Lead debe tipar bien ya; banca puede ser el pivot."""
    from showdown.battle_bots.greedy_solo.type_chart import (
        defensive_resistance_score,
        offensive_coverage_score,
    )

    foe_types = [m.types for m in foes if m.types] or [tuple()]
    s = 0.0
    for mon in leads:
        s += 1.35 * defensive_resistance_score(mon.types, foe_types)
        s += 1.15 * offensive_coverage_score(mon.types, foe_types)
        nid = _nid(mon.name)
        if nid in _REDIRECT or nid in _FAKE_OUT or nid in _TAILWIND:
            s += 10.0
    return s


def best_expectimax_preview_order(
    our_mons: Sequence[PreviewMon],
    foe_mons: Sequence[PreviewMon],
    *,
    alpha: float = 1.05,
    beta: float = 1.25,
    gamma: float = 0.008,
) -> PreviewScore:
    """
    Igual que joint (4 de 6) pero:
    - más peso defensivo en la combinación,
    - bonus de roles (Tailwind / TR / redirect / FO),
    - calidad extra del par de lead (no solo tipado agregado).
    """
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
        return PreviewScore(tuple(uniq[:4]), 0.0, 0.0, 0.0, 0.0)  # type: ignore[arg-type]

    best: Optional[PreviewScore] = None
    best_total = float('-inf')
    for combo in itertools.combinations(pool, 4):
        combo_list = list(combo)
        role = _role_bonus(combo_list)
        for lead in itertools.combinations(combo_list, 2):
            bench = [m for m in combo_list if m not in lead]
            ordered = list(lead) + bench
            base = score_combination(
                ordered, foe_mons, alpha=alpha, beta=beta, gamma=gamma,
            )
            lead_q = _lead_quality(list(lead), foe_mons)
            total = base.score + role + 0.45 * lead_q
            if total > best_total + 1e-9:
                best_total = total
                best = PreviewScore(
                    order=base.order,
                    offensive=base.offensive,
                    defensive=base.defensive,
                    bst=base.bst,
                    score=total,
                )
            elif best is not None and abs(total - best_total) < 1e-9:
                if base.bst > best.bst:
                    best = PreviewScore(
                        order=base.order,
                        offensive=base.offensive,
                        defensive=base.defensive,
                        bst=base.bst,
                        score=total,
                    )
                    best_total = total
    assert best is not None
    return best
