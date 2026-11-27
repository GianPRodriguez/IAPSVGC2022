"""Roles de un set (movimientos y habilidad), sin listas de especies.

Lo usan el selector de élites por rival (F2, ``qd_ppo/selector_f2.py``) para describir el equipo rival
y los descriptores ``roles`` del preview QD-PGA (F4a, team_preview_qd_pga_spec 1.7), que con el equipo
fijo necesitan mirar los sets reales: las listas de especies de soporte dejaban b2 siempre en 0.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence

FAKE_OUT = {'fakeout'}
REDIRECT_MOVES = {'followme', 'ragepowder', 'allyswitch'}
REDIRECT_ABILITIES = {'lightningrod', 'stormdrain'}
TAILWIND = {'tailwind'}
TRICK_ROOM = {'trickroom'}
SPEED_DROP = {'icywind', 'electroweb', 'bulldoze', 'rocktomb', 'scaryface', 'stringshot'}
INTIMIDATE = {'intimidate'}
SUPPORT_OTHER = {'helpinghand', 'wideguard', 'quickguard', 'spore', 'sleeppowder', 'willowisp',
                 'thunderwave', 'haze', 'clearsmog', 'snarl', 'partingshot', 'encore', 'taunt'}

ROLE_KEYS = ('support', 'physical', 'special', 'fake_out', 'redirect', 'tailwind', 'trick_room',
             'speed_drop', 'intimidate', 'setup')


def _id(x: str) -> str:
    return ''.join(ch for ch in str(x).lower() if ch.isalnum())


@lru_cache(maxsize=4096)
def _move(mid: str) -> dict:
    try:
        from showdown.rl import team_preview_obs as P
        return P._dex()['moves'].get(mid) or {}
    except Exception:  # pragma: no cover - dex ausente
        return {}


def mon_roles(mon) -> Dict[str, bool]:
    """Roles de un ``MonSet`` (``species``, ``ability``, ``moves``)."""
    moves = [_id(m) for m in (getattr(mon, 'moves', None) or [])]
    ability = _id(getattr(mon, 'ability', '') or '')
    phys = spec = 0
    setup = False
    for m in moves:
        d = _move(m)
        cat = str(d.get('category', ''))
        if cat == 'Physical':
            phys += 1
        elif cat == 'Special':
            spec += 1
        elif d.get('boosts') and str(d.get('target')) == 'self':
            setup = True
    r = {
        'fake_out': bool(FAKE_OUT & set(moves)),
        'redirect': bool(REDIRECT_MOVES & set(moves)) or ability in REDIRECT_ABILITIES,
        'tailwind': bool(TAILWIND & set(moves)),
        'trick_room': bool(TRICK_ROOM & set(moves)),
        'speed_drop': bool(SPEED_DROP & set(moves)),
        'intimidate': ability in INTIMIDATE,
        'setup': setup,
    }
    r['support'] = any(r[k] for k in ('fake_out', 'redirect', 'tailwind', 'trick_room', 'speed_drop',
                                      'intimidate')) or bool(SUPPORT_OTHER & set(moves))
    r['physical'] = phys > spec
    r['special'] = spec > phys
    return r


def team_roles(mons: Sequence) -> List[Dict[str, bool]]:
    return [mon_roles(m) for m in mons]


def team_role_counts(mons: Sequence) -> List[float]:
    """Cuántos Pokémon del equipo tienen cada rol (orden ``ROLE_KEYS``)."""
    rs = team_roles(mons)
    return [float(sum(1 for r in rs if r[k])) for k in ROLE_KEYS]
