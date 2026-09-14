"""Friendly fire: moves allAdjacent / all dañan al compañero si no Protect/inmune.

Generaliza Protect+EQ / switch Flying sin hardcodear jugadas.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from showdown.battle_bots.move_mechanics.accuracy import move_accuracy
from showdown.battle_bots.move_mechanics.charge import norm_id

# Castigo ≥ daño rival: freír al partner es muy malo.
ALLY_DAMAGE_WEIGHT = 1.15
# Alineado con KO_BONUS de greedy (perder un activo propio).
ALLY_KO_BONUS = 1000.0

# Targets PS que golpean adyacentes propios (no allAdjacentFoes).
_HITS_ALLY_TARGETS = frozenset({'alladjacent', 'all'})


def normalize_target_type(target_type: Optional[str]) -> str:
    return str(target_type or '').strip().lower().replace('_', '').replace('-', '')


def _moves_json() -> Dict[str, Any]:
    try:
        from showdown.battle_modifier import _moves_json as _load
        return _load() or {}
    except Exception:
        return {}


@lru_cache(maxsize=512)
def move_target_type(move_name: Optional[str]) -> str:
    """Target del dex (allAdjacent, allAdjacentFoes, …) normalizado."""
    mid = norm_id(move_name)
    if not mid:
        return ''
    entry = _moves_json().get(mid) or {}
    return normalize_target_type(entry.get('target'))


def hits_ally(
    target_type: Optional[str] = None,
    *,
    move_name: Optional[str] = None,
) -> bool:
    """True si el move puede dañar al compañero (EQ, Surf, Explosion…)."""
    t = normalize_target_type(target_type)
    if not t and move_name:
        t = move_target_type(move_name)
    return t in _HITS_ALLY_TARGETS


def partner_avoids_friendly_fire(
    partner_utility: Optional[str],
    partner_poke: Optional[Any] = None,
) -> bool:
    """Protect / switch / ya protegido → 0 friendly fire este turno."""
    u = str(partner_utility or '').strip().lower()
    if u in ('protect', 'switch'):
        return True
    if partner_poke is not None and bool(getattr(partner_poke, 'is_protected', False)):
        return True
    return False


def ally_hit_score_components(
    *,
    ally_damage_pct: float,
    ally_ko: bool,
    move_name: Optional[str] = None,
    damage_weight: float = ALLY_DAMAGE_WEIGHT,
    ko_bonus: float = ALLY_KO_BONUS,
) -> Tuple[float, float]:
    """
    Returns (penalty_to_subtract, p_ko_ally_approx).

    Escala por accuracy como el daño a foes.
    """
    acc = move_accuracy(move_name) if move_name else 1.0
    pct = max(0.0, float(ally_damage_pct)) * acc
    p_ko = float(acc) if ally_ko else 0.0
    pen = damage_weight * pct + (ko_bonus * p_ko if ally_ko else 0.0)
    return pen, p_ko


def friendly_fire_pair_penalty(
    left: Any,
    right: Any,
    left_poke: Optional[Any] = None,
    right_poke: Optional[Any] = None,
    *,
    damage_weight: float = ALLY_DAMAGE_WEIGHT,
    ko_bonus: float = ALLY_KO_BONUS,
) -> float:
    """
    Pena total del par A1×A2 por daño al compañero.

    left pega a right_poke si left.hits_ally y right no evita;
    simétrico para right→left.
    """
    pen = 0.0

    def _one(attacker_atom: Any, partner_atom: Any, partner_poke: Optional[Any]) -> float:
        if attacker_atom is None:
            return 0.0
        if not bool(getattr(attacker_atom, 'hits_ally', False)):
            # Fallback por nombre si el flag no vino seteado.
            if not hits_ally(move_name=getattr(attacker_atom, 'move_name', None)):
                return 0.0
        if partner_avoids_friendly_fire(
            getattr(partner_atom, 'utility', None), partner_poke,
        ):
            return 0.0
        if getattr(attacker_atom, 'utility', None) in ('switch', 'protect', 'charge'):
            return 0.0
        pct = float(getattr(attacker_atom, 'ally_damage_pct', 0.0) or 0.0)
        ko = bool(getattr(attacker_atom, 'ally_ko', False))
        if pct <= 0.0 and not ko:
            return 0.0
        part, _pk = ally_hit_score_components(
            ally_damage_pct=pct,
            ally_ko=ko,
            move_name=getattr(attacker_atom, 'move_name', None),
            damage_weight=damage_weight,
            ko_bonus=ko_bonus,
        )
        return part

    pen += _one(left, right, right_poke)
    pen += _one(right, left, left_poke)
    return pen


def friendly_fire_expectimax_adjust(
    *,
    threat: float,
    p_ko_side: Dict[str, float],
    left: Any,
    right: Any,
    left_poke: Optional[Any] = None,
    right_poke: Optional[Any] = None,
    soft_ko_bonus: float = 55.0,
    damage_weight: float = ALLY_DAMAGE_WEIGHT,
) -> Tuple[float, Dict[str, float]]:
    """
    Suma friendly fire a threat y a p_ko_side (OL/OR = víctima).

    left→right afecta OR; right→left afecta OL.
    """
    new_threat = float(threat)
    new_pko = {
        'OL': float(p_ko_side.get('OL', 0.0)),
        'OR': float(p_ko_side.get('OR', 0.0)),
    }

    def _apply(attacker_atom: Any, partner_atom: Any, victim_key: str) -> None:
        nonlocal new_threat
        if attacker_atom is None:
            return
        if not (
            bool(getattr(attacker_atom, 'hits_ally', False))
            or hits_ally(move_name=getattr(attacker_atom, 'move_name', None))
        ):
            return
        partner_poke = left_poke if victim_key == 'OL' else right_poke
        if partner_avoids_friendly_fire(
            getattr(partner_atom, 'utility', None), partner_poke,
        ):
            return
        if getattr(attacker_atom, 'utility', None) in ('switch', 'protect', 'charge'):
            return
        pct = float(getattr(attacker_atom, 'ally_damage_pct', 0.0) or 0.0)
        ko = bool(getattr(attacker_atom, 'ally_ko', False))
        if pct <= 0.0 and not ko:
            return
        acc = move_accuracy(getattr(attacker_atom, 'move_name', None))
        e_pct = pct * acc
        p_ko = acc if ko else 0.0
        new_threat += damage_weight * e_pct + p_ko * soft_ko_bonus
        new_pko[victim_key] = min(1.0, new_pko[victim_key] + p_ko)

    _apply(left, right, 'OR')
    _apply(right, left, 'OL')
    return new_threat, new_pko
