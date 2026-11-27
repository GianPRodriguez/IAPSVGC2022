"""Arquetipos de un equipo rival (para los diagnósticos D1/D2 del protocolo §4.16).

Etiquetas a partir de la hoja abierta (movimientos, habilidades y objetos):

- ``trick_room``, ``tailwind``: algún Pokémon lleva el movimiento.
- ``lluvia`` (Drizzle o Rain Dance), ``sol`` (Drought, Sunny Day o Charizardite Y), ``arena``, ``nieve``.
- ``anti_intimidate``: Defiant o Competitive (abrir con Intimidate los potencia).
- ``tierra_agua``: algún ataque de Tierra o Agua de potencia ≥ 70 (amenaza ×4 para Arcanine-Hisui).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, Optional, Set

TAGS = ('trick_room', 'tailwind', 'lluvia', 'sol', 'arena', 'nieve', 'anti_intimidate', 'tierra_agua',
        'tierra_agua_fuerte')


def _id(x) -> str:
    return ''.join(ch for ch in str(x or '').lower() if ch.isalnum())


def team_tags(team) -> Set[str]:
    from showdown.rl.team_preview_obs import _dex
    moves_dex = _dex()['moves']
    out: Set[str] = set()
    n_ga = 0
    for m in team.mons:
        mv = {_id(x) for x in (m.moves or [])}
        ab = _id(m.ability)
        it = _id(m.item)
        if 'trickroom' in mv:
            out.add('trick_room')
        if 'tailwind' in mv:
            out.add('tailwind')
        if ab == 'drizzle' or 'raindance' in mv:
            out.add('lluvia')
        if ab == 'drought' or 'sunnyday' in mv or it == 'charizarditey':
            out.add('sol')
        if ab in ('sandstream',) or 'sandstorm' in mv:
            out.add('arena')
        if ab in ('snowwarning',) or 'snowscape' in mv:
            out.add('nieve')
        if ab in ('defiant', 'competitive'):
            out.add('anti_intimidate')
        has_ga = False
        for x in mv:
            d = moves_dex.get(x) or {}
            if d.get('type') in ('Ground', 'Water') and d.get('category') != 'Status' \
                    and float(d.get('basePower') or 0) >= 70:
                has_ga = True
        if has_ga:
            out.add('tierra_agua')
            n_ga += 1
    if n_ga >= 2:
        out.add('tierra_agua_fuerte')
    return out


@lru_cache(maxsize=1)
def _all_teams() -> Dict[str, object]:
    from showdown.inference_sim.team_factory import active_teams
    from showdown.inference_sim.team_split import team_key
    return {team_key(t): t for t in active_teams('all')}


def tags_for_key(key: Optional[str]) -> Optional[Set[str]]:
    """Etiquetas del equipo con esa clave (``team_split.team_key``), o None si no se encuentra."""
    t = _all_teams().get(str(key or ''))
    return team_tags(t) if t is not None else None


def tag_counts(teams: Iterable) -> Dict[str, int]:
    c = {k: 0 for k in TAGS}
    for t in teams:
        for k in team_tags(t):
            c[k] += 1
    return c
