"""Decisiones según el plan del rival y uso real de los arquetipos (memorizacion_equipos.md §12).

Dos piezas para ``policy_diag --context``:

1. **Contexto de cada decisión** (:func:`contexts`): ¿el rival tiene Trick Room activo, Tailwind
   activo, un Fake Out disponible, redirección en el campo, o un Pokémon que puede activar Trick
   Room antes de que esté puesto? Con eso se mide si el agente cambia su juego (Protect, cambios,
   a quién ataca) cuando el rival ejecuta su plan.
2. **Uso real de los movimientos clave** (:func:`count_events`): cuántas veces cada lado usó
   Trick Room, Tailwind, Fake Out y redirección en la partida, según el protocolo crudo de
   Showdown (``battle._replay_data``). Mide si los rivales juegan su arquetipo o no.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Set

KEY_MOVES = {'trickroom': 'trick_room', 'tailwind': 'tailwind', 'fakeout': 'fake_out',
             'followme': 'redirect', 'ragepowder': 'redirect'}
REDIRECT = {'followme', 'ragepowder'}
# Clima / campo (protocolo §4.17): inicio por habilidad ([of] el que la tiene) o por movimiento (último usuario).
WEATHER_KEYS = {'raindance': 'lluvia', 'primordialsea': 'lluvia', 'sunnyday': 'sol', 'desolateland': 'sol',
                'sandstorm': 'arena', 'snow': 'nieve', 'snowscape': 'nieve', 'hail': 'nieve'}
TERRAIN_KEYS = {'grassyterrain': 'campo_hierba', 'psychicterrain': 'campo_psiquico',
                'electricterrain': 'campo_electrico', 'mistyterrain': 'campo_niebla'}
FIELD_MOVES = {'raindance', 'sunnyday', 'sandstorm', 'snowscape', 'chillyreception', 'hail', 'grassyterrain',
               'psychicterrain', 'electricterrain', 'mistyterrain'}
CONTEXTS = ('rival_trick_room', 'propio_trick_room', 'rival_tailwind', 'rival_fake_out',
            'rival_redireccion', 'rival_puede_trick_room', 'turno_1', 'sin_contexto')


def _id(x) -> str:
    return ''.join(ch for ch in str(x or '').lower() if ch.isalnum())


def _side(ident) -> str:
    """'p2a: Hatterene' / '[of] p2a: X' / 'p2: Nombre' → 'p2'."""
    s = str(ident or '')
    if s.startswith('[of]'):
        s = s[4:].strip()
    return s[:2] if s[:2] in ('p1', 'p2') else ''


def _messages(battle) -> List[list]:
    return list(getattr(battle, '_replay_data', None) or [])


def count_events(battle) -> Dict[str, Dict[str, int]]:
    """Usos de movimientos clave por lado ('nuestro' / 'rival') y veces que se activó cada efecto."""
    role = getattr(battle, 'player_role', None) or 'p1'
    out = {'nuestro': Counter(), 'rival': Counter()}
    last_tr_user = ''
    last_field_user = ''
    for m in _messages(battle):
        if len(m) < 3:
            continue
        kind = m[1]
        if kind == 'move' and len(m) > 3 and not any(str(p).startswith('[from]') for p in m[4:]):
            mid = _id(m[3])
            key = KEY_MOVES.get(mid)
            side = _side(m[2])
            if key and side:
                out['nuestro' if side == role else 'rival'][key] += 1
                if mid == 'trickroom':
                    last_tr_user = side
            last_field_user = side if mid in FIELD_MOVES else ''
        elif kind == '-weather' and not any('[upkeep]' in str(p) for p in m[3:]):
            wkey = WEATHER_KEYS.get(_id(m[2]))
            side = next((_side(p) for p in m[3:] if str(p).startswith('[of]')), '') or last_field_user
            if wkey and side:
                c = out['nuestro' if side == role else 'rival']
                c['clima_activado'] += 1
                c[f'{wkey}_activado'] += 1
        elif kind == '-fieldstart' and 'terrain' in _id(m[2]):
            tkey = TERRAIN_KEYS.get(_id(str(m[2]).replace('move:', '')))
            side = next((_side(p) for p in m[3:] if str(p).startswith('[of]')), '') or last_field_user
            if tkey and side:
                c = out['nuestro' if side == role else 'rival']
                c['campo_activado'] += 1
                c[f'{tkey}_activado'] += 1
        elif kind == '-fieldstart' and 'trickroom' in _id(m[2]):
            side = next((_side(p) for p in m[3:] if str(p).startswith('[of]')), '') or last_tr_user
            if side:
                out['nuestro' if side == role else 'rival']['trick_room_activado'] += 1
        elif kind == '-sidestart' and len(m) > 3 and 'tailwind' in _id(m[3]):
            side = _side(m[2])
            if side:
                out['nuestro' if side == role else 'rival']['tailwind_activado'] += 1
    return {k: dict(v) for k, v in out.items()}


def trick_room_owner(battle) -> str:
    """'nuestro' / 'rival' / '' : quién puso el Trick Room activo ahora (según el protocolo)."""
    role = getattr(battle, 'player_role', None) or 'p1'
    owner, last_user = '', ''
    for m in _messages(battle):
        if len(m) < 3:
            continue
        if m[1] == 'move' and len(m) > 3 and _id(m[3]) == 'trickroom':
            last_user = _side(m[2])
        elif m[1] == '-fieldstart' and 'trickroom' in _id(m[2]):
            side = next((_side(p) for p in m[3:] if str(p).startswith('[of]')), '') or last_user
            owner = ('nuestro' if side == role else 'rival') if side else ''
        elif m[1] == '-fieldend' and 'trickroom' in _id(m[2]):
            owner = ''
    return owner


def _known(mon) -> Set[str]:
    return {_id(x) for x in (getattr(mon, 'moves', None) or ())}


def foe_actives(snap) -> List:
    return [m for m in (getattr(snap.foe, 'active', None) or [])[:2]]


def contexts(snap, battle) -> Set[str]:
    """Situaciones presentes en esta decisión (puede haber varias)."""
    out: Set[str] = set()
    f = snap.field
    if bool(getattr(f, 'trick_room', False)):
        owner = trick_room_owner(battle)
        if owner == 'rival':
            out.add('rival_trick_room')
        elif owner == 'nuestro':
            out.add('propio_trick_room')
    if bool(getattr(f, 'opp_tailwind', False)):
        out.add('rival_tailwind')
    fa = [m for m in foe_actives(snap) if m is not None and not getattr(m, 'fainted', False)]
    if any(getattr(m, 'is_active', True) and int(getattr(m, 'moves_since_switch', 1)) == 0
           and 'fakeout' in _known(m) for m in fa):
        out.add('rival_fake_out')
    if any(_known(m) & REDIRECT for m in fa):
        out.add('rival_redireccion')
    if not getattr(f, 'trick_room', False) and any('trickroom' in _known(m) for m in fa):
        out.add('rival_puede_trick_room')
    if int(getattr(snap, 'turn', 0) or 0) <= 1:
        out.add('turno_1')
    if not out:
        out.add('sin_contexto')
    return out


def target_flags(snap, target: int) -> Dict[str, float]:
    """Para un ataque a un rival (objetivo 0/1): ¿apunta al que puede activar Trick Room / al redirector?

    Incluye lo esperado si eligiera el objetivo al azar entre los rivales vivos
    (``esperado_*`` = fracción de rivales vivos que cumplen la condición), para comparar.
    """
    fa = foe_actives(snap)
    mon = fa[target] if 0 <= int(target) < len(fa) else None
    if mon is None:
        return {}
    alive = [m for m in fa if m is not None and not getattr(m, 'fainted', False)] or [mon]
    k = _known(mon)
    return {'al_que_activa_trick_room': float('trickroom' in k), 'al_redirector': float(bool(k & REDIRECT)),
            'esperado_trick_room': sum('trickroom' in _known(m) for m in alive) / len(alive),
            'esperado_redirector': sum(bool(_known(m) & REDIRECT) for m in alive) / len(alive)}


def summarize(ctx_counts: Dict[str, Dict[str, int]]) -> List[str]:
    """Tabla markdown: % de cada decisión por contexto."""
    cats = ('protect', 'cambio_voluntario', 'ataque_rival', 'area_rivales', 'otro_estado', 'estado_rival')
    L = ['| Contexto | Decisiones | ' + ' | '.join(cats) + ' | ataques al que activa TR | ataques al redirector |',
         '|---|---|' + '---|' * (len(cats) + 2)]
    for ctx in CONTEXTS:
        c = ctx_counts.get(ctx, {})
        n = int(c.get('decisiones', 0))
        if not n:
            continue
        cells = [f"{100 * c.get(k, 0) / n:.1f} %" for k in cats]
        at = c.get('ataques_un_rival', 0)
        tr = (f"{100 * c.get('al_que_activa_trick_room', 0) / at:.0f} % (azar {100 * c.get('esperado_trick_room', 0) / at:.0f} %)"
              if at and ctx == 'rival_puede_trick_room' else '—')
        rd = (f"{100 * c.get('al_redirector', 0) / at:.0f} % (azar {100 * c.get('esperado_redirector', 0) / at:.0f} %)"
              if at and ctx == 'rival_redireccion' else '—')
        L.append(f'| {ctx} | {n} | ' + ' | '.join(cells) + f' | {tr} | {rd} |')
    return L
