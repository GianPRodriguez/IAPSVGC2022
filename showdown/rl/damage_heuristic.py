"""Local damage heuristic — NO Node.js (Spec v1.1)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Tuple

from showdown.battle_bots.greedy_solo.type_chart import type_effectiveness

_MOVES_PATH = Path(__file__).resolve().parents[2] / 'API' / 'JSONs' / 'moves.json'


def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s or '').lower())


@lru_cache(maxsize=1)
def _moves_dex() -> dict:
    try:
        return json.loads(_MOVES_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {}


@lru_cache(maxsize=4096)
def move_entry(move_id: str) -> dict:
    dex = _moves_dex()
    key = _norm(move_id)
    if key in dex:
        return dex[key]
    for k, v in dex.items():
        if _norm(k) == key or _norm(str(v.get('name', ''))) == key:
            return v
    return {}


def estimate_damage_pct(
    move_name: str,
    *,
    attacker_types: Sequence[str] = (),
    defender_types: Sequence[str] = (),
    weather: str = '',
    base_atk_spa: float = 100.0,
    base_def_spd: float = 100.0,
    level: int = 50,
) -> Tuple[float, float]:
    """
    Crude Min%/Max% via BP × STAB × effectiveness × weather.
    Failsafe (0.0, 0.0). Never calls Node.
    """
    try:
        entry = move_entry(move_name)
        if not entry:
            return 0.0, 0.0
        cat = str(entry.get('category') or '')
        if cat.lower() == 'status':
            return 0.0, 0.0
        bp = float(entry.get('basePower') or 0)
        if bp <= 0:
            return 0.0, 0.0
        mtype = str(entry.get('type') or 'Normal')
        stab = 1.5 if any(_norm(t) == _norm(mtype) for t in attacker_types) else 1.0
        eff = float(type_effectiveness(mtype, list(defender_types) or ['Normal']))
        w = _norm(weather)
        weather_mod = 1.0
        if w in ('sunnyday', 'desolateland') and _norm(mtype) == 'fire':
            weather_mod = 1.5
        elif w in ('sunnyday', 'desolateland') and _norm(mtype) == 'water':
            weather_mod = 0.5
        elif w in ('raindance', 'primordialsea') and _norm(mtype) == 'water':
            weather_mod = 1.5
        elif w in ('raindance', 'primordialsea') and _norm(mtype) == 'fire':
            weather_mod = 0.5

        # Simplified Gen damage fraction of max-HP proxy (relative %).
        atk = max(1.0, float(base_atk_spa))
        df = max(1.0, float(base_def_spd))
        raw = (((2 * level / 5 + 2) * bp * (atk / df)) / 50.0 + 2.0)
        raw *= stab * eff * weather_mod
        # Map to rough % of a 150 HP mon
        pct = 100.0 * raw / 150.0
        lo = max(0.0, pct * 0.85)
        hi = max(0.0, pct * 1.0)
        return float(lo), float(hi)
    except Exception:
        return 0.0, 0.0
