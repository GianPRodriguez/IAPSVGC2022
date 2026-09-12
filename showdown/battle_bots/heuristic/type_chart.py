"""Matriz de efectividad de tipos (Gen 6+ / Champions)."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

TYPES = (
    'Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice',
    'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic', 'Bug',
    'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy',
)

# Atacante -> Defensor -> multiplicador (omitidos = 1.0)
_CHART = {
    'Normal': {'Rock': 0.5, 'Ghost': 0.0, 'Steel': 0.5},
    'Fire': {
        'Fire': 0.5, 'Water': 0.5, 'Grass': 2.0, 'Ice': 2.0,
        'Bug': 2.0, 'Rock': 0.5, 'Dragon': 0.5, 'Steel': 2.0,
    },
    'Water': {
        'Fire': 2.0, 'Water': 0.5, 'Grass': 0.5, 'Ground': 2.0,
        'Rock': 2.0, 'Dragon': 0.5,
    },
    'Electric': {
        'Water': 2.0, 'Electric': 0.5, 'Grass': 0.5, 'Ground': 0.0,
        'Flying': 2.0, 'Dragon': 0.5,
    },
    'Grass': {
        'Fire': 0.5, 'Water': 2.0, 'Grass': 0.5, 'Poison': 0.5,
        'Ground': 2.0, 'Flying': 0.5, 'Bug': 0.5, 'Rock': 2.0,
        'Dragon': 0.5, 'Steel': 0.5,
    },
    'Ice': {
        'Fire': 0.5, 'Water': 0.5, 'Grass': 2.0, 'Ice': 0.5,
        'Ground': 2.0, 'Flying': 2.0, 'Dragon': 2.0, 'Steel': 0.5,
    },
    'Fighting': {
        'Normal': 2.0, 'Ice': 2.0, 'Poison': 0.5, 'Flying': 0.5,
        'Psychic': 0.5, 'Bug': 0.5, 'Rock': 2.0, 'Ghost': 0.0,
        'Dark': 2.0, 'Steel': 2.0, 'Fairy': 0.5,
    },
    'Poison': {
        'Grass': 2.0, 'Poison': 0.5, 'Ground': 0.5, 'Rock': 0.5,
        'Ghost': 0.5, 'Steel': 0.0, 'Fairy': 2.0,
    },
    'Ground': {
        'Fire': 2.0, 'Electric': 2.0, 'Grass': 0.5, 'Poison': 2.0,
        'Flying': 0.0, 'Bug': 0.5, 'Rock': 2.0, 'Steel': 2.0,
    },
    'Flying': {
        'Electric': 0.5, 'Grass': 2.0, 'Fighting': 2.0, 'Bug': 2.0,
        'Rock': 0.5, 'Steel': 0.5,
    },
    'Psychic': {
        'Fighting': 2.0, 'Poison': 2.0, 'Psychic': 0.5, 'Dark': 0.0,
        'Steel': 0.5,
    },
    'Bug': {
        'Fire': 0.5, 'Grass': 2.0, 'Fighting': 0.5, 'Poison': 0.5,
        'Flying': 0.5, 'Psychic': 2.0, 'Ghost': 0.5, 'Dark': 2.0,
        'Steel': 0.5, 'Fairy': 0.5,
    },
    'Rock': {
        'Fire': 2.0, 'Ice': 2.0, 'Fighting': 0.5, 'Ground': 0.5,
        'Flying': 2.0, 'Bug': 2.0, 'Steel': 0.5,
    },
    'Ghost': {
        'Normal': 0.0, 'Psychic': 2.0, 'Ghost': 2.0, 'Dark': 0.5,
    },
    'Dragon': {
        'Dragon': 2.0, 'Steel': 0.5, 'Fairy': 0.0,
    },
    'Dark': {
        'Fighting': 0.5, 'Psychic': 2.0, 'Ghost': 2.0, 'Dark': 0.5,
        'Fairy': 0.5,
    },
    'Steel': {
        'Fire': 0.5, 'Water': 0.5, 'Electric': 0.5, 'Ice': 2.0,
        'Rock': 2.0, 'Steel': 0.5, 'Fairy': 2.0,
    },
    'Fairy': {
        'Fire': 0.5, 'Fighting': 2.0, 'Poison': 0.5, 'Dragon': 2.0,
        'Dark': 2.0, 'Steel': 0.5,
    },
}


def normalize_type(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    key = str(t).strip()
    if not key or key.lower() in ('none', 'null', '???', 'typeless'):
        return None
    titled = key[:1].upper() + key[1:].lower()
    # aliases
    aliases = {
        'Electrict': 'Electric',
        'Fight': 'Fighting',
    }
    titled = aliases.get(titled, titled)
    if titled in _CHART:
        return titled
    for known in TYPES:
        if known.lower() == key.lower():
            return known
    return titled


def type_effectiveness(attack_type: Optional[str], defend_types: Sequence[Optional[str]]) -> float:
    atk = normalize_type(attack_type)
    if not atk:
        return 1.0
    mult = 1.0
    row = _CHART.get(atk, {})
    for dt in defend_types:
        d = normalize_type(dt)
        if not d:
            continue
        mult *= float(row.get(d, 1.0))
    return mult


def offensive_coverage_score(
    attacker_types: Iterable[Optional[str]],
    defender_type_sets: Iterable[Sequence[Optional[str]]],
) -> float:
    """Suma de super-efectividades (x2+) de nuestros tipos ofensivos vs rival."""
    score = 0.0
    atk_types = [normalize_type(t) for t in attacker_types if normalize_type(t)]
    for defs in defender_type_sets:
        best = 0.0
        for atk in atk_types:
            m = type_effectiveness(atk, defs)
            if m > best:
                best = m
        if best >= 2.0:
            score += best
        elif best > 0:
            score += best * 0.25
    return score


def defensive_resistance_score(
    defender_types: Sequence[Optional[str]],
    attacker_type_sets: Iterable[Sequence[Optional[str]]],
) -> float:
    """Cuánto resistimos los tipos ofensivos del rival (menor daño = mejor)."""
    score = 0.0
    for atk_types in attacker_type_sets:
        worst = 1.0
        for atk in atk_types:
            m = type_effectiveness(atk, defender_types)
            if m > worst:
                worst = m
        # Resistir (≤0.5) suma; ser SE (≥2) resta
        if worst == 0.0:
            score += 2.0
        elif worst <= 0.5:
            score += 1.0 / max(worst, 0.25)
        elif worst >= 2.0:
            score -= worst
        else:
            score += 0.1
    return score


def matchup_vs_actives(
    candidate_types: Sequence[Optional[str]],
    foe_type_sets: Sequence[Sequence[Optional[str]]],
) -> float:
    """Ventaja defensiva de un mon de banca vs los activos rivales."""
    return defensive_resistance_score(candidate_types, foe_type_sets)


def pokemon_types(type1: Optional[str], type2: Optional[str] = None) -> Tuple[Optional[str], ...]:
    t1 = normalize_type(type1)
    t2 = normalize_type(type2)
    if t1 and t2 and t1 != t2:
        return (t1, t2)
    if t1:
        return (t1,)
    if t2:
        return (t2,)
    return tuple()
