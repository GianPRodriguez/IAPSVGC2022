import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple


_STATS_PATH = Path(__file__).resolve().parent / "JSONs" / "champions_info_stats.json"
_MISS_WARNED: set = set()


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


def _lookup_pkmn(name_pkmn: str) -> Optional[dict]:
    by_lower, by_compact = _stats_indexes()
    key = (name_pkmn or "").strip().lower()
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


@lru_cache(maxsize=1024)
def getPkmnInfo(name_pkmn: str):
    try:
        key = (name_pkmn or "").strip().lower()
        if not key:
            return None, None, 0, 0, 0, 0, 0, 0
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
        if key not in _MISS_WARNED:
            _MISS_WARNED.add(key)
            print(f"[AVISO] Pokémon '{name_pkmn}' no encontrado en champions_info_stats.json")
    except Exception as error:
        print(f"[ERROR] Falló la lectura de champions_info_stats.json para '{name_pkmn}': {error}")

    return None, None, 0, 0, 0, 0, 0, 0
