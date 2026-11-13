"""Registro de decisiones para diagnosticar no determinismo (``PS_DECISION_LOG``).

Por cada decisión de un jugador escribe una línea JSON con el turno, un hash de
la petición (``|request|``) que Showdown le envió y la orden elegida. Comparando
dos corridas con la misma semilla, la primera línea que difiere dice si el
estado que ve el bot ya es distinto (Showdown) o si, con el mismo estado, el bot
decide distinto (el bot).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from typing import Optional


def _req_hash(battle) -> str:
    """Huella canónica del estado que ve el bot (sin nombres de cuenta ni rqid)."""
    try:
        state = {
            'turn': getattr(battle, 'turn', None),
            'force': list(getattr(battle, 'force_switch', []) or []),
            'active': [m.species if m else None for m in battle.active_pokemon],
            'hp': [round(m.current_hp_fraction, 3) if m else None for m in battle.active_pokemon],
            'moves': [[mv.id for mv in ms] for ms in battle.available_moves],
            'switches': [[m.species for m in sw] for sw in battle.available_switches],
            'foe': [m.species if m else None for m in battle.opponent_active_pokemon],
            'foe_hp': [round(m.current_hp_fraction, 3) if m else None
                       for m in battle.opponent_active_pokemon],
        }
        raw = json.dumps(state, sort_keys=True, default=str)
    except Exception:
        raw = ''
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]


def install(player, label: str, path: Optional[str] = None) -> bool:
    path = path or os.environ.get('PS_DECISION_LOG', '').strip()
    if not path:
        return False
    path = f'{path}.{os.getpid()}'
    orig = player.choose_move

    def _write(battle, order) -> None:
        try:
            with open(path, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps({
                    't': round(time.time(), 2), 'tag': str(getattr(battle, 'battle_tag', '')),
                    'p': label, 'turn': int(getattr(battle, 'turn', 0) or 0),
                    'req': _req_hash(battle), 'order': str(getattr(order, 'message', order)),
                }) + chr(10))
        except OSError:
            pass

    def wrapped(battle):
        res = orig(battle)
        if inspect.isawaitable(res):
            async def _aw():
                order = await res
                _write(battle, order)
                return order
            return _aw()
        _write(battle, res)
        return res

    player.choose_move = wrapped
    return True
