"""Bloqueo de moves con prioridad (Psychic Terrain + Dazzling / Queenly Majesty / Armor Tail).

Aproxima PS:
  - Psychic Terrain: falla prioridad > 0 vs objetivo grounded (no aliado/self).
  - Dazzling / Queenly Majesty / Armor Tail: fallan prioridad vs ese bando.
  - Mold Breaker / Teravolt / Turboblaze ignoran las abilities (no el terreno).
"""
from __future__ import annotations

from typing import Optional, Sequence

from showdown.battle_bots.greedy_solo.type_chart import normalize_type
from showdown.battle_bots.move_mechanics.charge import norm_id
from showdown.stat_inference import move_priority

PRIORITY_BLOCK_ABILITIES = frozenset({
    'dazzling',
    'queenlymajesty',
    'armortail',
})

_MOLD_BREAKER_LIKE = frozenset({
    'moldbreaker',
    'teravolt',
    'turboblaze',
})


def effective_priority(
    move_name: Optional[str],
    *,
    attacker_ability: Optional[str] = None,
    terrain: Optional[str] = None,
) -> int:
    raw = move_priority(move_name, attacker_ability, terrain)
    return int(raw) if raw is not None else 0


def is_priority_move(
    move_name: Optional[str],
    *,
    attacker_ability: Optional[str] = None,
    terrain: Optional[str] = None,
) -> bool:
    return effective_priority(
        move_name, attacker_ability=attacker_ability, terrain=terrain,
    ) > 0


def is_grounded(
    poke,
    *,
    gravity: bool = False,
) -> bool:
    """Aprox. isGrounded de PS (sin Magnet Rise / telekinesis volatiles)."""
    if poke is None:
        return True
    if gravity:
        return True
    item = norm_id(getattr(poke, 'item', None))
    if item == 'ironball':
        return True
    if item == 'airballoon':
        return False
    abi = norm_id(getattr(poke, 'ability', None))
    if abi == 'levitate':
        return False
    if bool(getattr(poke, 'is_terastallized', False)) and getattr(poke, 'tera_type', None):
        types = (getattr(poke, 'tera_type'),)
    else:
        types = tuple(getattr(poke, 'types', None) or ())
    for t in types:
        if normalize_type(t) == 'Flying':
            return False
    return True


def side_has_priority_block_ability(
    mons: Sequence,
    *,
    ignored_by_mold_breaker: bool = False,
) -> bool:
    if ignored_by_mold_breaker:
        return False
    for mon in mons:
        if mon is None:
            continue
        if norm_id(getattr(mon, 'ability', None)) in PRIORITY_BLOCK_ABILITIES:
            return True
    return False


def priority_move_fails(
    move_name: Optional[str],
    *,
    attacker=None,
    defender=None,
    defender_side: Sequence = (),
    terrain: Optional[str] = None,
    gravity: bool = False,
) -> bool:
    """
    True si el move con prioridad fallaría contra este objetivo / bando.

    - Terreno Psíquico: solo si el *defender* está grounded.
    - Armor Tail / Dazzling / Queenly Majesty: si cualquier mon del bando
      defensor la tiene (salvo Mold Breaker-like del atacante).
    """
    atk_abi = getattr(attacker, 'ability', None) if attacker is not None else None
    prio = effective_priority(move_name, attacker_ability=atk_abi, terrain=terrain)
    if prio <= 0:
        return False

    mold = norm_id(atk_abi) in _MOLD_BREAKER_LIKE
    side = list(defender_side) if defender_side else []
    if defender is not None and defender not in side:
        side = [defender, *side]

    if side_has_priority_block_ability(side, ignored_by_mold_breaker=mold):
        return True

    if norm_id(terrain) == 'psychicterrain' and defender is not None:
        if is_grounded(defender, gravity=gravity):
            return True

    return False
