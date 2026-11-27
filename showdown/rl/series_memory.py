"""Memoria de serie BO3: lo que el rival hizo en las partidas anteriores contra el mismo equipo.

rl_env_spec v1.35 (PPO v3 / DDQN v2, protocolo §4.16). En un BO3 los dos jugadores repiten equipo. Con
hojas abiertas, la partida 1 no revela movimientos ni objetos (ya se conocen), pero sí **qué 4 trajo el
rival, quién abrió y cómo jugó**. Este módulo:

1. extrae de cada partida terminada un ``GameRecord`` a partir del protocolo de Showdown que poke-env
   guarda en ``battle._replay_data`` (especies traídas y de apertura, KOs sufridos, movimientos, Protect,
   Trick Room, Tailwind, clima, campo, cambios, Fake Out, duración y resultado);
2. acumula los registros de la serie en ``SeriesMemory``;
3. produce el bloque de la observación (``MEM_DIM`` = 21 globales + 8 por cada uno de los 6 bloques
   rivales, en el mismo orden que ``observation._foe_slots``). Sin partidas previas el bloque es todo
   cero salvo el índice de partida, así que una red ampliada con pesos nuevos en cero juega la partida 1
   igual que sin memoria.

Las frecuencias son promedios sobre las partidas previas, no la última: un humano también se adapta
(adojs cambió el compañero de apertura en la partida 2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from showdown.rl.constants import MEM_DIM, MEM_GLOBAL_DIM, MEM_SLOT_DIM, N_FOE_MONS

PROTECT_IDS = frozenset({
    'protect', 'detect', 'spikyshield', 'banefulbunker', 'kingsshield', 'silktrap', 'obstruct',
    'burningbulwark', 'maxguard',
})
FAKE_OUT_IDS = frozenset({'fakeout'})
WEATHERS = ('rain', 'sun', 'sand', 'snow')
_WEATHER_KEY = {
    'raindance': 'rain', 'primordialsea': 'rain',
    'sunnyday': 'sun', 'desolateland': 'sun',
    'sandstorm': 'sand',
    'snow': 'snow', 'snowscape': 'snow', 'hail': 'snow',
}
_MEGA_SUFFIX = re.compile(r'(megax|megay|mega)$')


def to_id(text) -> str:
    return re.sub(r'[^a-z0-9]', '', str(text or '').lower())


def species_key(name) -> str:
    """Especie normalizada sin la forma Mega (la hoja y el combate muestran la base o la Mega)."""
    s = to_id(name)
    return _MEGA_SUFFIX.sub('', s) if len(s) > 6 else s


def _role_of(ident: str) -> str:
    """``'p2a: Farigiraf'`` → ``'p2'``."""
    s = str(ident or '').strip()
    return s[:2] if len(s) >= 2 and s[0] == 'p' and s[1] in '1234' else ''


def _position(ident: str) -> str:
    """``'p2a: Farigiraf'`` → ``'p2a'``."""
    s = str(ident or '').strip()
    return s.split(':', 1)[0].strip() if ':' in s else ''


def _nick(ident: str) -> str:
    s = str(ident or '')
    return s.split(':', 1)[1].strip() if ':' in s else s.strip()


_DAMAGING_CACHE: Dict[str, bool] = {}


def _is_damaging(move_id: str) -> bool:
    if move_id not in _DAMAGING_CACHE:
        try:
            from showdown.rl.team_preview_obs import _dex
            m = _dex()['moves'].get(move_id) or {}
            _DAMAGING_CACHE[move_id] = str(m.get('category', 'Status')) != 'Status'
        except Exception:  # pragma: no cover - sin dex: se considera ataque
            _DAMAGING_CACHE[move_id] = True
    return _DAMAGING_CACHE[move_id]


@dataclass
class MonRecord:
    """Lo que hizo un Pokémon rival en una partida."""
    brought: bool = False
    lead: bool = False
    fainted: bool = False
    active_turns: int = 0
    moves: int = 0
    protects: int = 0
    damaging: int = 0
    protect_turn1: bool = False


@dataclass
class GameRecord:
    """Resumen de una partida terminada, desde el punto de vista del agente (el rival es ``foe_role``)."""
    foe_role: str = 'p2'
    mons: Dict[str, MonRecord] = field(default_factory=dict)  # species_key -> registro
    turns: int = 0
    won: Optional[bool] = None
    trick_room_turn: Optional[int] = None   # primer turno en que el rival activó Trick Room
    tailwind_turn: Optional[int] = None
    weather: Set[str] = field(default_factory=set)  # climas que puso el rival
    terrain: bool = False
    protects: int = 0
    active_turns: int = 0
    switches: int = 0  # cambios voluntarios del rival (no reemplazos tras un KO)
    fake_out_turn1: bool = False

    def brought(self) -> List[str]:
        return sorted(k for k, m in self.mons.items() if m.brought)

    def leads(self) -> List[str]:
        return sorted(k for k, m in self.mons.items() if m.lead)

    @classmethod
    def from_messages(cls, messages: Iterable[Sequence[str]], *, our_role: str,
                      won: Optional[bool] = None) -> 'GameRecord':
        """Desde mensajes ya separados por ``|`` (``battle._replay_data``: ``['', 'switch', ...]``)."""
        foe = 'p2' if str(our_role or 'p1').startswith('p1') else 'p1'
        rec = cls(foe_role=foe, won=won)
        turn = 0
        nick_species: Dict[str, str] = {}   # 'p2: Nick' -> species_key
        active: Dict[str, str] = {}         # 'p2a' -> species_key
        fainted_pos: Set[str] = set()       # posiciones del rival vaciadas por un KO
        last_move_role = ''

        def mon(key: str) -> MonRecord:
            return rec.mons.setdefault(key, MonRecord())

        for parts in messages:
            parts = list(parts)
            if len(parts) < 2:
                continue
            kind = parts[1]
            if kind == 'turn' and len(parts) > 2:
                try:
                    turn = int(parts[2])
                except ValueError:
                    continue
                for pos, key in active.items():
                    if pos.startswith(foe) and key:
                        mon(key).active_turns += 1
                        rec.active_turns += 1
                rec.turns = max(rec.turns, turn)
            elif kind in ('switch', 'drag', 'replace') and len(parts) > 3:
                ident, details = parts[2], parts[3]
                if _role_of(ident) != foe:
                    continue
                key = species_key(details.split(',', 1)[0])
                pos = _position(ident)
                nick_species[f'{foe}: {_nick(ident)}'] = key
                m = mon(key)
                m.brought = True
                if turn == 0:
                    m.lead = True
                elif kind == 'switch' and pos not in fainted_pos:
                    rec.switches += 1
                fainted_pos.discard(pos)
                active[pos] = key
            elif kind == 'faint' and len(parts) > 2:
                ident = parts[2]
                if _role_of(ident) != foe:
                    continue
                key = nick_species.get(f'{foe}: {_nick(ident)}') or active.get(_position(ident))
                if key:
                    mon(key).fainted = True
                fainted_pos.add(_position(ident))
                active.pop(_position(ident), None)
            elif kind == 'move' and len(parts) > 3:
                ident = parts[2]
                last_move_role = _role_of(ident)
                if last_move_role != foe or any(str(x).startswith('[from]') for x in parts[4:]):
                    continue  # movimientos llamados por otros (Instruct, Dancer…) no son elección propia
                key = nick_species.get(f'{foe}: {_nick(ident)}') or active.get(_position(ident))
                mid = to_id(parts[3])
                m = mon(key) if key else None
                if m is not None:
                    m.moves += 1
                if mid in PROTECT_IDS:
                    rec.protects += 1
                    if m is not None:
                        m.protects += 1
                        m.protect_turn1 = m.protect_turn1 or turn == 1
                elif m is not None and _is_damaging(mid):
                    m.damaging += 1
                if mid in FAKE_OUT_IDS and turn == 1:
                    rec.fake_out_turn1 = True
            elif kind == '-fieldstart' and len(parts) > 2:
                what = to_id(parts[2].replace('move:', ''))
                setter = _setter_role(parts[3:], last_move_role)
                if setter != foe:
                    continue
                if what == 'trickroom':
                    if rec.trick_room_turn is None:
                        rec.trick_room_turn = turn
                elif what.endswith('terrain'):
                    rec.terrain = True
            elif kind == '-sidestart' and len(parts) > 3:
                side = str(parts[2]).strip()[:2]
                if side == foe and to_id(parts[3].replace('move:', '')) == 'tailwind':
                    if rec.tailwind_turn is None:
                        rec.tailwind_turn = turn
            elif kind == '-weather' and len(parts) > 2:
                w = _WEATHER_KEY.get(to_id(parts[2]))
                if w is None or any(str(x).startswith('[upkeep]') for x in parts[3:]):
                    continue
                if _setter_role(parts[3:], last_move_role) == foe:
                    rec.weather.add(w)
        return rec

    @classmethod
    def from_battle(cls, battle) -> 'GameRecord':
        """Desde una batalla de poke-env terminada (``_replay_data`` + ``won``)."""
        msgs = list(getattr(battle, '_replay_data', None) or [])
        won = getattr(battle, 'won', None)
        return cls.from_messages(msgs, our_role=str(getattr(battle, 'player_role', None) or 'p1'),
                                 won=bool(won) if won is not None else None)


def _setter_role(rest: Sequence[str], last_move_role: str) -> str:
    """Quién activó un efecto: ``[of] p2a: X`` si viene (habilidad), si no el último que movió."""
    for x in rest:
        x = str(x)
        if x.startswith('[of]'):
            return _role_of(x[4:].strip())
    return last_move_role


@dataclass
class SeriesMemory:
    """Partidas terminadas de la serie actual (mismo rival y mismos equipos)."""
    records: List[GameRecord] = field(default_factory=list)

    @property
    def game_index(self) -> int:
        """Partida que se juega ahora (1, 2 o 3)."""
        return min(3, len(self.records) + 1)

    @property
    def our_wins(self) -> int:
        return sum(1 for r in self.records if r.won is True)

    @property
    def foe_wins(self) -> int:
        return sum(1 for r in self.records if r.won is False)

    @property
    def decided(self) -> bool:
        return self.our_wins >= 2 or self.foe_wins >= 2 or len(self.records) >= 3

    def add(self, record: GameRecord) -> None:
        self.records.append(record)

    def add_battle(self, battle) -> GameRecord:
        rec = GameRecord.from_battle(battle)
        self.add(rec)
        return rec

    def foe_brought(self) -> List[str]:
        """Especies (clave normalizada) que el rival trajo en alguna partida previa."""
        out: Set[str] = set()
        for r in self.records:
            out.update(r.brought())
        return sorted(out)

    # -- bloque de la observación ------------------------------------------------ #
    def global_vector(self) -> np.ndarray:
        v = np.zeros(MEM_GLOBAL_DIM, dtype=np.float32)
        v[self.game_index - 1] = 1.0                 # 0–2: partida 1/2/3
        n = len(self.records)
        if n == 0:
            return v
        recs = self.records
        v[3] = self.our_wins / 2.0
        v[4] = self.foe_wins / 2.0
        v[5] = n / 2.0
        tr = [r.trick_room_turn for r in recs if r.trick_room_turn is not None]
        tw = [r.tailwind_turn for r in recs if r.tailwind_turn is not None]
        v[6] = len(tr) / n
        v[7] = min(1.0, (sum(tr) / len(tr)) / 10.0) if tr else 0.0
        v[8] = len(tw) / n
        v[9] = min(1.0, (sum(tw) / len(tw)) / 10.0) if tw else 0.0
        for i, w in enumerate(WEATHERS):              # 10–13
            v[10 + i] = sum(1 for r in recs if w in r.weather) / n
        v[14] = sum(1 for r in recs if r.terrain) / n
        act = sum(r.active_turns for r in recs)
        turns = sum(r.turns for r in recs)
        v[15] = min(1.0, sum(r.protects for r in recs) / act) if act else 0.0
        v[16] = min(1.0, sum(r.switches for r in recs) / turns) if turns else 0.0
        v[17] = sum(1 for r in recs if r.fake_out_turn1) / n
        v[18] = min(1.0, (turns / n) / 20.0)
        v[19] = 1.0 if recs[-1].won is True else 0.0
        v[20] = 1.0                                   # hay partidas previas
        return v

    def slot_vector(self, species) -> np.ndarray:
        """Bloque de un Pokémon rival (por especie); cero si no hay partidas previas o no hay Pokémon."""
        v = np.zeros(MEM_SLOT_DIM, dtype=np.float32)
        n = len(self.records)
        if n == 0 or not species:
            return v
        key = species_key(species)
        ms = [r.mons.get(key) for r in self.records]
        got = [m for m in ms if m is not None and m.brought]
        v[0] = 1.0                                    # hay memoria para este bloque
        v[1] = len(got) / n                           # traído
        v[2] = sum(1 for m in got if m.lead) / n      # abrió
        if got:
            v[3] = sum(1 for m in got if m.fainted) / len(got)
            act = sum(m.active_turns for m in got)
            mv = sum(m.moves for m in got)
            v[4] = min(1.0, sum(m.protects for m in got) / act) if act else 0.0
            v[5] = min(1.0, (act / len(got)) / 10.0)
            v[6] = sum(1 for m in got if m.protect_turn1) / len(got)
            v[7] = (sum(m.damaging for m in got) / mv) if mv else 0.0
        return v

    def vector(self, foe_species: Sequence[Optional[str]]) -> np.ndarray:
        """float32[MEM_DIM]: global + un bloque por cada especie de ``foe_species`` (6, orden de la obs)."""
        slots = list(foe_species)[:N_FOE_MONS] + [None] * max(0, N_FOE_MONS - len(foe_species))
        out = np.concatenate([self.global_vector()] + [self.slot_vector(s) for s in slots])
        assert out.shape[0] == MEM_DIM
        return out.astype(np.float32)


EMPTY = SeriesMemory()
