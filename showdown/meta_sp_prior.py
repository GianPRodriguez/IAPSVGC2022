"""
Prior suave de Stat Points a partir del compendio MetaVGC (OTS scrapeados).

Solo se usa para desempate / point-estimate entre candidatos YA compatibles
con las observaciones de combate. Nunca poda StatBelief.candidates.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from showdown.stat_inference import STAT_KEYS
from teams.team_converter import normalize_name

_JSON_PATH = (
    Path(__file__).resolve().parent / 'inference_sim' / 'metavgc_regmc_teams.json'
)

# species -> nature -> Counter de tuples SP (hp,atk,def,spa,spd,spe)
_SpreadIndex = Dict[str, Dict[str, Counter]]
# species -> nature -> stat -> Counter[sp]
_MarginalIndex = Dict[str, Dict[str, Dict[str, Counter]]]

_loaded_path: Optional[Path] = None
_spread_index: _SpreadIndex = {}
_marginal_index: _MarginalIndex = {}
_species_marginal: Dict[str, Dict[str, Counter]] = {}
_global_marginal: Dict[str, Counter] = {k: Counter() for k in STAT_KEYS}
_sample_count: int = 0


def _norm(s: Optional[str]) -> str:
    return normalize_name(s or '') or ''


def _sp_tuple(sp: dict) -> Tuple[int, int, int, int, int, int]:
    return tuple(int(sp.get(k, 0) or 0) for k in STAT_KEYS)  # type: ignore[return-value]


def _reset_indexes() -> None:
    global _spread_index, _marginal_index, _species_marginal, _global_marginal
    global _sample_count, _loaded_path
    _spread_index = defaultdict(lambda: defaultdict(Counter))
    _marginal_index = defaultdict(lambda: defaultdict(lambda: {k: Counter() for k in STAT_KEYS}))
    _species_marginal = defaultdict(lambda: {k: Counter() for k in STAT_KEYS})
    _global_marginal = {k: Counter() for k in STAT_KEYS}
    _sample_count = 0
    _loaded_path = None


def load_meta_sp_prior(path: Optional[Path] = None, *, force: bool = False) -> int:
    """Carga/recarga el JSON MetaVGC. Devuelve nº de mons indexados."""
    global _loaded_path, _sample_count
    p = Path(path) if path is not None else _JSON_PATH
    if (
        not force
        and _loaded_path is not None
        and _loaded_path == p
        and _sample_count > 0
    ):
        return _sample_count
    _reset_indexes()
    _loaded_path = p
    if not p.is_file():
        return 0
    try:
        raw = json.loads(p.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return 0
    if not isinstance(raw, list):
        return 0

    n = 0
    for team in raw:
        for mon in (team.get('mons') or []):
            species = _norm(mon.get('species'))
            nature = _norm(mon.get('nature')) or 'serious'
            sp = mon.get('sp') or {}
            if not species:
                continue
            tup = _sp_tuple(sp)
            _spread_index[species][nature][tup] += 1
            for i, stat in enumerate(STAT_KEYS):
                val = tup[i]
                _marginal_index[species][nature][stat][val] += 1
                _species_marginal[species][stat][val] += 1
                _global_marginal[stat][val] += 1
            n += 1
    _sample_count = n
    return n


def prior_sample_count() -> int:
    load_meta_sp_prior()
    return _sample_count


def _compatible_spreads(
    species: str,
    nature: str,
    candidates: Dict[str, Set[int]],
) -> List[Tuple[Tuple[int, ...], int]]:
    """Spreads observados (species, nature) que caben en todos los sets candidatos."""
    load_meta_sp_prior()
    counter = _spread_index.get(species, {}).get(nature)
    if not counter:
        return []
    out: List[Tuple[Tuple[int, ...], int]] = []
    for tup, count in counter.items():
        ok = True
        for i, stat in enumerate(STAT_KEYS):
            allowed = candidates.get(stat)
            if allowed is not None and tup[i] not in allowed:
                ok = False
                break
        if ok:
            out.append((tup, count))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


def _mode_in_set(counter: Counter, allowed: Sequence[int]) -> Optional[int]:
    if not allowed:
        return None
    best_sp = None
    best_w = -1
    for sp in allowed:
        w = int(counter.get(sp, 0))
        if w > best_w or (w == best_w and (best_sp is None or sp < best_sp)):
            best_w = w
            best_sp = sp
    if best_sp is None or best_w <= 0:
        return None
    return best_sp


def _marginal_for(species: str, nature: str, stat: str) -> Counter:
    load_meta_sp_prior()
    by_nat = _marginal_index.get(species, {}).get(nature, {}).get(stat)
    if by_nat:
        return by_nat
    by_sp = _species_marginal.get(species, {}).get(stat)
    if by_sp:
        return by_sp
    return _global_marginal.get(stat) or Counter()


def pick_sp_from_candidates(
    *,
    species: Optional[str],
    nature: Optional[str],
    candidates: Iterable[int],
    stat: str,
    all_candidates: Optional[Dict[str, Set[int]]] = None,
) -> Optional[int]:
    """
    Elige un SP entre candidates según el prior meta.

    Si all_candidates cubre los 6 stats, intenta primero un spread conjunto
    observado compatible; si no hay, cae al modo marginal por stat.
    """
    allowed = sorted({int(x) for x in candidates})
    if not allowed:
        return None
    if len(allowed) == 1:
        return allowed[0]

    sid = _norm(species)
    nid = _norm(nature) or 'serious'
    load_meta_sp_prior()
    if _sample_count <= 0:
        return None

    if all_candidates is not None and sid:
        joint = _compatible_spreads(sid, nid, all_candidates)
        if joint:
            idx = STAT_KEYS.index(stat)
            return int(joint[0][0][idx])

    picked = _mode_in_set(_marginal_for(sid, nid, stat), allowed)
    return picked


def preferred_sp_map(
    *,
    species: Optional[str],
    nature: Optional[str],
    candidates: Dict[str, Set[int]],
) -> Dict[str, int]:
    """SP preferidos por stat (joint si cabe; si no, marginal + mediana fallback)."""
    sid = _norm(species)
    nid = _norm(nature) or 'serious'
    load_meta_sp_prior()
    out: Dict[str, int] = {}

    if sid and _sample_count > 0:
        joint = _compatible_spreads(sid, nid, candidates)
        if joint:
            tup = joint[0][0]
            return {stat: int(tup[i]) for i, stat in enumerate(STAT_KEYS)}

    for stat in STAT_KEYS:
        cands = sorted(candidates.get(stat) or [])
        if not cands:
            continue
        picked = pick_sp_from_candidates(
            species=sid,
            nature=nid,
            candidates=cands,
            stat=stat,
            all_candidates=None,
        )
        if picked is None:
            picked = cands[len(cands) // 2]
        out[stat] = int(picked)
    return out
