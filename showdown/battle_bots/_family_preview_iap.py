"""Shared IAP websocket Team Preview via frozen family policy."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from showdown.battle import Battle
from showdown.rl.frozen_preview import FIXED_EXPORT_TUPLE, family_preview_for_combat


def _species_from_side(pokemon_list: Sequence[dict]) -> List[str]:
    out: List[str] = []
    for row in pokemon_list[:6]:
        details = str(row.get('details') or '')
        out.append(details.partition(',')[0] or 'Unknown')
    return out


def iap_family_team_preview(combat_id: str, battle: Battle) -> Tuple[int, ...]:
    """Return 1-based order; queue UI names; fallback (1,2,3,4)."""
    side = battle.request_json.get('side', {}).get('pokemon', []) or []
    our = _species_from_side(side)
    # Foe OTS may be absent in some IAP paths → pad unknowns (policy still runs).
    foe_side = (
        battle.request_json.get('opponent', {}).get('pokemon', [])
        or battle.request_json.get('side', {}).get('foe', [])
        or []
    )
    foe = _species_from_side(foe_side) if foe_side else ['unknown'] * 6
    pol = family_preview_for_combat(combat_id)
    order = pol.action_tuple_from_species(our, foe)
    names = []
    for idx in order:
        if 0 <= idx - 1 < len(side):
            names.append(str(side[idx - 1].get('details', '')).partition(',')[0])
        else:
            names.append('Unknown')
    while len(names) < 4:
        names.append('Unknown')
    if getattr(battle, 'q', None) is not None:
        battle.q.put([1, names[0], names[1], names[2], names[3], 0, 0])
    return order if order else FIXED_EXPORT_TUPLE
