"""OTS público del rival: todo lo visible en Champions excepto Stat Points.

El SP verdadero NUNCA vive aquí (eso es trampa / solo soundness del harness).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from teams.team_converter import normalize_name


def _norm(s: Optional[str]) -> str:
    return normalize_name(s or '') or ''


@dataclass(frozen=True)
class OpponentPublicOTS:
    """Hoja abierta usable en decisión (sin SP)."""

    species: str
    item: Optional[str] = None
    ability: Optional[str] = None
    nature: Optional[str] = None
    moves: Tuple[str, ...] = ()
    tera: Optional[str] = None

    @property
    def species_id(self) -> str:
        return _norm(self.species)

    def complete_for_decision(self) -> bool:
        """Item + ability + nature + ≥1 move: suficiente para clavar el set visible."""
        return bool(
            self.item
            and self.ability
            and self.nature
            and self.moves
        )


def ots_from_monset(mon) -> OpponentPublicOTS:
    """Desde TeamMeta.MonSet / objeto con los mismos campos — sin copiar SP."""
    moves = tuple(
        str(m) for m in (getattr(mon, 'moves', None) or [])
        if m
    )
    return OpponentPublicOTS(
        species=str(getattr(mon, 'species', '') or ''),
        item=str(getattr(mon, 'item', '') or '') or None,
        ability=str(getattr(mon, 'ability', '') or '') or None,
        nature=str(getattr(mon, 'nature', '') or '') or None,
        moves=moves,
        tera=str(getattr(mon, 'tera', '') or '') or None,
    )


def ots_sheet_from_team_meta(team) -> Dict[str, OpponentPublicOTS]:
    """species_id → OTS público. No incluye true_sp."""
    out: Dict[str, OpponentPublicOTS] = {}
    for mon in getattr(team, 'mons', None) or []:
        sheet = ots_from_monset(mon)
        sid = sheet.species_id
        if sid:
            out[sid] = sheet
    return out


def ots_from_iap_pokemon(pkmn) -> Optional[OpponentPublicOTS]:
    """Desde Pokémon IAP ya booteado con OTS (item/ability/moves/nature)."""
    if pkmn is None:
        return None
    name = str(getattr(pkmn, 'name', '') or '')
    if not name:
        return None
    moves: List[str] = []
    raw_moves = getattr(pkmn, 'moves', None) or []
    if isinstance(raw_moves, dict):
        raw_moves = list(raw_moves.values())
    for m in raw_moves:
        if not m:
            continue
        if isinstance(m, str):
            moves.append(m)
        else:
            nm = getattr(m, 'name', None) or getattr(m, 'id', None)
            if nm:
                moves.append(str(nm))
    return OpponentPublicOTS(
        species=name,
        item=str(getattr(pkmn, 'objectPkmn', None) or getattr(pkmn, 'item', None) or '') or None,
        ability=str(getattr(pkmn, 'abilityPkmn', None) or getattr(pkmn, 'ability', None) or '') or None,
        nature=str(getattr(pkmn, 'nature', None) or '') or None,
        moves=tuple(moves),
        tera=str(getattr(pkmn, 'tera_type', None) or '') or None,
    )


def lookup_ots(
    sheet: Optional[Dict[str, OpponentPublicOTS]],
    species: str,
) -> Optional[OpponentPublicOTS]:
    if not sheet:
        return None
    sid = _norm(species)
    if sid in sheet:
        return sheet[sid]
    for key, val in sheet.items():
        if _norm(key) == sid or _norm(val.species) == sid:
            return val
    # forme base: salamencemega → salamence
    compact = sid.replace('-', '')
    for key, val in sheet.items():
        k = _norm(key).replace('-', '')
        if compact.startswith(k) or k.startswith(compact):
            return val
    return None
