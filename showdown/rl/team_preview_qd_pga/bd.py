"""Preview BD b1 (speed-control) / b2 (soporte en la apertura) — spec §3.

v1.7 (26-11-2026, F4a): modo ``roles`` (PREVIEW_BD=roles) para el equipo propio fijo: b1 = fracción de
leads de soporte y b2 = fracción de físicos entre los 4 que se llevan, según los sets reales
(``showdown/rl/team_roles.py``). Con las listas de especies, el equipo fijo daba b2 siempre 0.

v1.6 (25-11-2026, I 2.0): b2 = fracción de los dos leads que son de soporte (redirección, Tailwind,
Trick Room o Fake Out). Antes contaba como "agresivo" todo lo que no fuera soporte y casi siempre
valía 1; ahora varía con la elección del preview (0, 1/2 o 1 por contexto; continuo en la media).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Sequence, Tuple

from showdown.battle_bots.expectimax_bayesian.team_preview import (
    FAKE_OUT_SPECIES,
    REDIRECT_SPECIES,
    SPEED_CONTROL_SPECIES,
    TAILWIND_SPECIES,
    TRICK_ROOM_SPECIES,
    _nid,
)
from showdown.rl.team_preview_action import decode_preview_action


def bd_mode() -> str:
    """``species`` (v1.6, por defecto) o ``roles`` (v1.7, equipo propio fijo)."""
    return 'roles' if os.environ.get('PREVIEW_BD', '').strip().lower() == 'roles' else 'species'


@lru_cache(maxsize=8)
def _own_roles(path: str) -> Dict[str, Dict[str, bool]]:
    from showdown.inference_sim.team_factory import load_team_file
    from showdown.rl.team_roles import mon_roles
    return {_nid(m.species): mon_roles(m) for m in load_team_file(path).mons}


def roles_bd(action: int, our_species: Sequence[str], roles: Dict[str, Dict[str, bool]]) -> Tuple[float, float]:
    """b1 = leads de soporte / 2; b2 = físicos entre los 4 que se llevan / 4."""
    i1, i2, i3, i4 = decode_preview_action(int(action))
    names = list(our_species)
    while len(names) < 6:
        names.append('Pikachu')
    def has(i, k):
        return bool(roles.get(_nid(names[i]), {}).get(k, False))
    b1 = (has(i1, 'support') + has(i2, 'support')) / 2.0
    b2 = sum(has(i, 'physical') for i in (i1, i2, i3, i4)) / 4.0
    return float(b1), float(b2)


def preview_bd_from_action(
    action: int,
    our_species: Sequence[str],
) -> Tuple[float, float]:
    """
    b1: TR / TW-speed-control bias.
    b2: fracción de leads de soporte (redirect / Tailwind / Trick Room / Fake Out).
    """
    if bd_mode() == 'roles':
        team = os.environ.get('PS_PREVIEW_OWN_TEAM', '').strip()
        if not team:
            raise ValueError('PREVIEW_BD=roles necesita el equipo propio fijo (PS_PREVIEW_OWN_TEAM / --own-team)')
        return roles_bd(action, our_species, _own_roles(team))
    i1, i2, i3, i4 = decode_preview_action(int(action))
    names = list(our_species)
    while len(names) < 6:
        names.append('Pikachu')
    bring_ids = {_nid(names[i]) for i in (i1, i2, i3, i4)}
    lead_ids = [_nid(names[i1]), _nid(names[i2])]
    bench_ids = [_nid(names[i3]), _nid(names[i4])]

    has_tr = bool(bring_ids & TRICK_ROOM_SPECIES)
    has_tw_sc = bool(bring_ids & TAILWIND_SPECIES) or bool(
        bring_ids & SPEED_CONTROL_SPECIES
    )
    b1 = float(max(0.0, min(1.0, 0.5 * float(has_tr) + 0.5 * float(has_tw_sc))))

    support = REDIRECT_SPECIES | TAILWIND_SPECIES | TRICK_ROOM_SPECIES | FAKE_OUT_SPECIES
    del bench_ids
    b2 = float(sum(1 for x in lead_ids if x in support)) / 2.0
    return b1, b2
