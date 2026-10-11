"""Módulo A: selección Team Preview por cobertura de tipos + BST.

Contrato: docs/greedy_solo_spec.md §3 (compartido por greedy_joint).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from API.PokemonData import getPkmnInfo

from showdown.battle_bots.greedy_solo.species_names import resolve_species_name
from showdown.battle_bots.greedy_solo.type_chart import (
    defensive_resistance_score,
    offensive_coverage_score,
    pokemon_types,
)

# Pesos tipado/BST (≠ Expectimax PREVIEW_ALPHA/BETA/GAMMA).
PREVIEW_ALPHA: float = 1.0
PREVIEW_BETA: float = 1.0
PREVIEW_GAMMA: float = 0.01
SCORE_EPS: float = 1e-9  # tolerancia al comparar scores / empates


@dataclass(frozen=True)
class PreviewMon:
    index: int  # 1-based Showdown slot
    name: str
    types: Tuple[Optional[str], ...]
    bst: int


@dataclass(frozen=True)
class PreviewScore:
    order: Tuple[int, int, int, int]
    offensive: float
    defensive: float
    bst: float
    score: float


def _bst_and_types(species: str) -> Tuple[Tuple[Optional[str], ...], int]:
    species = resolve_species_name(species)
    info = getPkmnInfo(species)
    if not info:
        return tuple(), 0
    t1, t2, hp, atk, df, spa, spd, spe = info
    types = pokemon_types(t1, t2)
    bst = int(hp or 0) + int(atk or 0) + int(df or 0) + int(spa or 0) + int(spd or 0) + int(spe or 0)
    return types, bst


def mon_from_details(index: int, details: str) -> PreviewMon:
    name = (details or '').partition(',')[0].strip() or f'Unknown{index}'
    types, bst = _bst_and_types(name)
    return PreviewMon(index=index, name=name, types=types, bst=bst)


def mon_from_species(index: int, species: str) -> PreviewMon:
    types, bst = _bst_and_types(species)
    return PreviewMon(index=index, name=species, types=types, bst=bst)


def score_combination(
    combo: Sequence[PreviewMon],
    foe_mons: Sequence[PreviewMon],
    *,
    alpha: float = PREVIEW_ALPHA,
    beta: float = PREVIEW_BETA,
    gamma: float = PREVIEW_GAMMA,
) -> PreviewScore:
    foe_type_sets = [m.types for m in foe_mons if m.types]
    if not foe_type_sets:
        foe_type_sets = [tuple()]

    offensive = 0.0
    defensive = 0.0
    for mon in combo:
        offensive += offensive_coverage_score(mon.types, foe_type_sets)
        defensive += defensive_resistance_score(mon.types, foe_type_sets)

    bst = float(sum(m.bst for m in combo))
    total = alpha * offensive + beta * defensive + gamma * bst
    order = tuple(int(m.index) for m in combo)  # type: ignore[assignment]
    return PreviewScore(
        order=order,  # type: ignore[arg-type]
        offensive=offensive,
        defensive=defensive,
        bst=bst,
        score=total,
    )


def best_team_preview_order(
    our_mons: Sequence[PreviewMon],
    foe_mons: Sequence[PreviewMon],
    *,
    alpha: float = PREVIEW_ALPHA,
    beta: float = PREVIEW_BETA,
    gamma: float = PREVIEW_GAMMA,
    lead_preference: Optional[Sequence[int]] = None,
) -> PreviewScore:
    """
    Elige la mejor combinación de 4 de 6.

    Retorna orden (a,b,c,d) 1-based: a,b salen de lead; c,d banca.
    Empate de score → mayor BST. `lead_preference` reservado (no usado en v1.1).
    """
    del lead_preference  # API estable; sin efecto (spec §3 non-goals)
    pool = list(our_mons)
    if len(pool) < 4:
        # Relleno defensivo
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
    for combo in itertools.combinations(pool, 4):
        # Leads = C(4,2); banca = los 2 restantes (orden relativo del combo)
        combo_list = list(combo)
        for lead in itertools.combinations(combo_list, 2):
            bench = [m for m in combo_list if m not in lead]
            ordered = list(lead) + bench
            scored = score_combination(
                ordered, foe_mons, alpha=alpha, beta=beta, gamma=gamma
            )
            if best is None or scored.score > best.score + SCORE_EPS:
                best = scored
            elif best is not None and abs(scored.score - best.score) < SCORE_EPS:
                if scored.bst > best.bst:
                    best = scored

    assert best is not None
    return best


def order_to_team_string(order: Sequence[int]) -> str:
    """Cadena exacta para WebSocket: ej. '1234'."""
    return ''.join(str(int(i)) for i in order)


def foe_mons_from_names(names: Iterable[str]) -> List[PreviewMon]:
    out: List[PreviewMon] = []
    for i, name in enumerate(names, start=1):
        out.append(mon_from_species(i, name))
    return out
