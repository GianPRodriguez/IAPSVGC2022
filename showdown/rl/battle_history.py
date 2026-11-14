"""Historial del combate a partir del protocolo de Showdown (para la observación v2).

poke-env guarda todos los mensajes del combate en ``battle._replay_data``. De
ahí se reconstruye, por Pokémon (clave ``'p1: Nombre'``, la misma que usa
``battle.team`` / ``battle.opponent_team``):

- último movimiento usado y hace cuántos turnos;
- turnos en campo y movimientos usados desde su último cambio;
- veces que entró al campo;
- HP% perdido en el último turno;

y un resumen del último turno por bando (Protect, cambios, KOs, Fake Out, spread).

Movimientos llamados por otros (``[from]``: Instruct, Dancer…) no cuentan como
elección propia. Se cachea por (id de batalla, nº de mensajes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_STALLING = {'protect', 'detect', 'spikyshield', 'banefulbunker', 'kingsshield', 'silktrap',
             'obstruct', 'burningbulwark', 'maxguard'}
_FAKEOUT = {'fakeout', 'firstimpression'}


def _id(text: str) -> str:
    return ''.join(ch for ch in str(text or '').lower() if ch.isalnum())


def ident_key(ident: str) -> Optional[str]:
    """``'p2a: Kingambit'`` → ``'p2: Kingambit'`` (clave de ``battle.team``)."""
    s = str(ident or '').strip()
    if len(s) < 4 or s[0] != 'p' or ':' not in s:
        return None
    role, _, name = s.partition(':')
    return f'{role[:2]}: {name.strip()}'


def _hp_pct(hp_status: str) -> Optional[float]:
    s = str(hp_status or '').strip().split(' ')[0]
    if s.startswith('0') and ('fnt' in hp_status or s == '0'):
        return 0.0
    if '/' not in s:
        return None
    a, _, b = s.partition('/')
    try:
        return max(0.0, min(100.0, 100.0 * float(a) / float(b)))
    except (ValueError, ZeroDivisionError):
        return None


@dataclass
class MonHistory:
    last_move: str = ''
    last_move_turn: int = -1
    switch_in_turn: int = -1
    times_switched_in: int = 0
    moves_since_switch: int = 0
    hp_by_turn: Dict[int, float] = field(default_factory=dict)  # HP% al final de cada turno visto


@dataclass
class BattleHistory:
    turn: int = 0
    mons: Dict[str, MonHistory] = field(default_factory=dict)
    last_turn: Dict[str, Dict[str, float]] = field(default_factory=dict)  # role → contadores

    def mon(self, key: Optional[str]) -> Optional[MonHistory]:
        return self.mons.get(key) if key else None

    def hp_lost_last_turn(self, key: Optional[str]) -> float:
        h = self.mon(key)
        if h is None or self.turn < 1:
            return 0.0
        end_prev = self._hp_at(h, self.turn - 1)
        end_prev2 = self._hp_at(h, self.turn - 2)
        if end_prev is None or end_prev2 is None:
            return 0.0
        return max(0.0, end_prev2 - end_prev)

    @staticmethod
    def _hp_at(h: MonHistory, turn: int) -> Optional[float]:
        known = [t for t in h.hp_by_turn if t <= turn]
        return h.hp_by_turn[max(known)] if known else None


_CACHE: Tuple[Optional[int], int, Optional[BattleHistory]] = (None, -1, None)


def parse_history(messages: List[List[str]]) -> BattleHistory:
    hist = BattleHistory()
    turn = 0
    window: Dict[int, Dict[str, Dict[str, float]]] = {}

    def _bump(t: int, role: str, key: str, n: float = 1.0) -> None:
        window.setdefault(t, {}).setdefault(role, {})
        window[t][role][key] = window[t][role].get(key, 0.0) + n

    for msg in messages:
        if len(msg) < 2:
            continue
        kind = msg[1]
        if kind == 'turn' and len(msg) > 2:
            try:
                turn = int(msg[2])
            except ValueError:
                pass
            continue
        if kind in ('switch', 'drag', 'replace') and len(msg) > 2:
            key = ident_key(msg[2])
            if key is None:
                continue
            h = hist.mons.setdefault(key, MonHistory())
            h.switch_in_turn = turn
            h.times_switched_in += 1
            h.moves_since_switch = 0
            if len(msg) > 4:
                hp = _hp_pct(msg[4])
                if hp is not None:
                    h.hp_by_turn[turn] = hp
            if kind == 'switch' and turn > 0:
                _bump(turn, key[:2], 'switch')
            continue
        if kind == 'move' and len(msg) > 3:
            key = ident_key(msg[2])
            if key is None or any(str(p).startswith('[from]') for p in msg[4:]):
                continue
            h = hist.mons.setdefault(key, MonHistory())
            mid = _id(msg[3])
            h.last_move, h.last_move_turn = mid, turn
            h.moves_since_switch += 1
            role = key[:2]
            if mid in _STALLING:
                _bump(turn, role, 'protect')
            if mid in _FAKEOUT:
                _bump(turn, role, 'fakeout')
            if '[spread]' in ' '.join(str(p) for p in msg[4:]):
                _bump(turn, role, 'spread')
            continue
        if kind in ('-damage', '-heal', '-sethp') and len(msg) > 3:
            key = ident_key(msg[2])
            hp = _hp_pct(msg[3])
            if key is not None and hp is not None:
                hist.mons.setdefault(key, MonHistory()).hp_by_turn[turn] = hp
            continue
        if kind == 'faint' and len(msg) > 2:
            key = ident_key(msg[2])
            if key is not None:
                hist.mons.setdefault(key, MonHistory()).hp_by_turn[turn] = 0.0
                _bump(turn, key[:2], 'faint')
    hist.turn = turn
    hist.last_turn = window.get(turn - 1, {}) if turn > 0 else {}
    return hist


def history_from_battle(battle) -> Optional[BattleHistory]:
    """Historial de una batalla poke-env (cacheado mientras no lleguen mensajes)."""
    global _CACHE
    msgs = getattr(battle, '_replay_data', None)
    if not msgs:
        return None
    key = id(battle)
    if _CACHE[0] == key and _CACHE[1] == len(msgs) and _CACHE[2] is not None:
        return _CACHE[2]
    hist = parse_history(msgs)
    _CACHE = (key, len(msgs), hist)
    return hist
