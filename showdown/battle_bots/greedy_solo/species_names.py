"""Utilidades de nombre de especie para Champions / poke-env."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional


@lru_cache(maxsize=1)
def _species_keys() -> dict:
    path = Path(__file__).resolve().parents[3] / 'API' / 'JSONs' / 'champions_info_stats.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    return {re.sub(r'[^a-z0-9]', '', k.lower()): k for k in data.keys()}


def resolve_species_name(raw: Optional[str]) -> str:
    if not raw:
        return 'Magikarp'
    text = str(raw).strip()
    if not text:
        return 'Magikarp'
    keys = _species_keys()
    for k in keys.values():
        if k.lower() == text.lower():
            return k
    compact = re.sub(r'[^a-z0-9]', '', text.lower())
    if compact in keys:
        return keys[compact]
    return '-'.join(p.capitalize() for p in text.replace(' ', '-').split('-'))
