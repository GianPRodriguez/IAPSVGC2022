import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple


_STATS_PATH = Path(__file__).resolve().parent / "JSONs" / "champions_info_stats.json"
_MISS_WARNED: set = set()

# Showdown forme id → clave en champions_info_stats.json (lower).
# Floette Eternal Flower mega is "Floette-Mega" in Champions, not floette-eternal-mega.
# Basculegion male: PS/poke-env = "basculegion"; pastes a veces traen "Basculegion-M".
_SPECIES_ALIASES: Dict[str, str] = {
    'floette-eternal-mega': 'floette-mega',
    'basculegion-m': 'basculegion',
    'basculegion-male': 'basculegion',
    'basculegionm': 'basculegion',
    'basculegion-female': 'basculegion-f',
}


def _compact_key(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _stats_indexes() -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Carga una vez: índice lower + índice compacto (Mr-Mime → Mr. Mime)."""
    with _STATS_PATH.open("r", encoding="utf-8") as stats_file:
        stats_data = json.load(stats_file)
    by_lower: Dict[str, dict] = {}
    by_compact: Dict[str, dict] = {}
    for key, value in stats_data.items():
        low = str(key).lower()
        by_lower[low] = value
        compact = _compact_key(low)
        by_compact.setdefault(compact, value)
    return by_lower, by_compact


def _alias_key(key: str) -> str:
    low = (key or "").strip().lower()
    if not low:
        return low
    if low in _SPECIES_ALIASES:
        return _SPECIES_ALIASES[low]
    compact = _compact_key(low)
    for src, dst in _SPECIES_ALIASES.items():
        if _compact_key(src) == compact:
            return dst
    return low


def _lookup_pkmn(name_pkmn: str) -> Optional[dict]:
    by_lower, by_compact = _stats_indexes()
    key = _alias_key(name_pkmn)
    if not key:
        return None
    hit = by_lower.get(key)
    if hit:
        return hit
    for alt in (
        key.replace("-", ". "),
        key.replace("-", " "),
        key.replace(".", " "),
        key.replace(".", "-"),
    ):
        hit = by_lower.get(alt)
        if hit:
            return hit
    return by_compact.get(_compact_key(key))


def champions_species_key(name_pkmn: str) -> Optional[str]:
    """Clave canónica (lower) en champions_info_stats, o None si no hay hit."""
    by_lower, by_compact = _stats_indexes()
    key = _alias_key(name_pkmn)
    if not key:
        return None
    if key in by_lower:
        return key
    for alt in (
        key.replace("-", ". "),
        key.replace("-", " "),
        key.replace(".", " "),
        key.replace(".", "-"),
    ):
        if alt in by_lower:
            return alt
    compact = _compact_key(key)
    if compact in by_compact:
        # Recuperar la clave lower del índice principal si existe
        for low in by_lower:
            if _compact_key(low) == compact:
                return low
        return compact
    return None


def has_pkmn_info(name_pkmn: str) -> bool:
    """True solo si hay entrada real en champions_info_stats (tras aliases)."""
    return _lookup_pkmn(name_pkmn) is not None


@lru_cache(maxsize=1024)
def getPkmnInfo(name_pkmn: str):
    """
    Stats Champions: (tipo1, tipo2, hp, atk, def, spa, spd, spe) o None si miss.

    None es falsy — usar `if info:` / `has_pkmn_info`. No devolver tupla de ceros
    en miss (rompía `if getPkmnInfo(cand)` en resolvers de mega).
    """
    try:
        key = (name_pkmn or "").strip().lower()
        if not key:
            return None
        pkmn_info = _lookup_pkmn(name_pkmn)
        if pkmn_info:
            return (
                pkmn_info.get("tipo1"),
                pkmn_info.get("tipo2"),
                pkmn_info.get("hp", 0),
                pkmn_info.get("ataque", 0),
                pkmn_info.get("defensa", 0),
                pkmn_info.get("ataque_especial", 0),
                pkmn_info.get("defensa_especial", 0),
                pkmn_info.get("velocidad", 0),
            )
        warn_key = _alias_key(key)
        if warn_key not in _MISS_WARNED and key not in _MISS_WARNED:
            _MISS_WARNED.add(key)
            _MISS_WARNED.add(warn_key)
            print(f"[AVISO] Pokémon '{name_pkmn}' no encontrado en champions_info_stats.json")
    except Exception as error:
        print(f"[ERROR] Falló la lectura de champions_info_stats.json para '{name_pkmn}': {error}")

    return None
