"""512D Team Preview observation encoder.

Contract: docs/team_preview_masked_ppo_spec.md §3.
Calc / coverage cached at encode time (env.reset); never inside combat step loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from showdown.battle_bots.greedy_solo.type_chart import (
    TYPES,
    normalize_type,
    type_effectiveness,
)
from showdown.battle_bots.expectimax_bayesian.team_preview import (
    FAKE_OUT_SPECIES,
    REDIRECT_SPECIES,
    SPEED_CONTROL_SPECIES,
    TAILWIND_SPECIES,
    TRICK_ROOM_SPECIES,
    _nid,
)
from showdown.battle_bots.greedy_solo.species_names import resolve_species_name
from API.PokemonData import getPkmnInfo

OBS_DIM = 512
MON_DIM = 35
N_POOL = 6
EPS = 1e-6
TYPE_DIM = 18
STAT_KEYS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')


@dataclass
class PreviewMonFeat:
    species: str
    types: Tuple[Optional[str], ...] = ()
    stats: Optional[dict] = None  # hp..spe absolute
    max_hp: Optional[int] = None
    item: Optional[str] = None
    ability: Optional[str] = None
    moves: Tuple[str, ...] = ()
    tera_type: Optional[str] = None


@dataclass
class PreviewObsBundle:
    our: List[PreviewMonFeat] = field(default_factory=list)
    foe: List[PreviewMonFeat] = field(default_factory=list)
    # Optional precomputed d_max matrix (6,6); else type-heuristic
    dmax: Optional[np.ndarray] = None


def _dex_stats_types(species: str) -> Tuple[dict, Tuple[Optional[str], ...], int]:
    name = resolve_species_name(species)
    info = getPkmnInfo(name)
    if not info:
        return {k: 100 for k in STAT_KEYS}, tuple(), 100
    t1, t2, hp, atk, df, spa, spd, spe = info
    types = tuple(t for t in (normalize_type(t1), normalize_type(t2)) if t)
    stats = {
        'hp': int(hp or 100),
        'atk': int(atk or 100),
        'def': int(df or 100),
        'spa': int(spa or 100),
        'spd': int(spd or 100),
        'spe': int(spe or 100),
    }
    # Rough L50 maxHP from base (formula without IV/EV): floor((2*base)*50/100)+50+10
    max_hp = int((2 * stats['hp'] * 50) // 100) + 60
    return stats, types, max_hp


def mon_feat_from_species(species: str, **kwargs) -> PreviewMonFeat:
    stats, types, max_hp = _dex_stats_types(species)
    return PreviewMonFeat(
        species=species,
        types=types or kwargs.get('types', ()),
        stats=kwargs.get('stats') or stats,
        max_hp=kwargs.get('max_hp') or max_hp,
        item=kwargs.get('item'),
        ability=kwargs.get('ability'),
        moves=tuple(kwargs.get('moves') or ()),
        tera_type=kwargs.get('tera_type'),
    )


def _encode_mon(mon: Optional[PreviewMonFeat]) -> np.ndarray:
    v = np.zeros(MON_DIM, dtype=np.float32)
    if mon is None:
        return v
    stats = mon.stats or {k: 100 for k in STAT_KEYS}
    for i, k in enumerate(STAT_KEYS):
        v[i] = float(stats.get(k, 100)) / 255.0
    # types multi-hot 18
    for t in mon.types or ():
        nt = normalize_type(t)
        if nt and nt in TYPES:
            v[6 + TYPES.index(nt)] = 1.0
    # item/ability flags (6)
    item = (mon.item or '').lower().replace(' ', '').replace('-', '')
    abi = (mon.ability or '').lower().replace(' ', '').replace('-', '')
    if 'choice' in item:
        v[24] = 1.0
    if 'focussash' in item:
        v[25] = 1.0
    if any(x in item for x in ('lifeorb', 'expertbelt', 'boosterenergy')):
        v[26] = 1.0
    if any(x in abi for x in ('levitate', 'flashfire', 'wellbakedbody', 'waterabsorb')):
        v[27] = 1.0
    nid = _nid(mon.species)
    if nid in SPEED_CONTROL_SPECIES:
        v[28] = 1.0
    if nid in REDIRECT_SPECIES or nid in TAILWIND_SPECIES or nid in TRICK_ROOM_SPECIES:
        v[29] = 1.0
    # move-role / meta flags (5) — species priors if moves unknown
    if nid in FAKE_OUT_SPECIES:
        v[30] = 1.0  # priority-ish
    if nid in REDIRECT_SPECIES:
        v[31] = 1.0  # support
    if nid in TAILWIND_SPECIES or nid in TRICK_ROOM_SPECIES:
        v[32] = 1.0  # setup/speed
    if any('pivot' in m.lower() or m.lower() in ('uturn', 'voltswitch', 'flipturn') for m in mon.moves):
        v[33] = 1.0
    if any(m.lower() in ('heatwave', 'rockslide', 'blizzard', 'earthquake') for m in mon.moves):
        v[34] = 1.0  # spread
    elif nid in ('tinglu', 'rillaboom', 'landorustherian'):
        v[34] = 0.5
    return v


def _heuristic_dmax(attacker: PreviewMonFeat, defender: PreviewMonFeat) -> float:
    """Local proxy when @smogon/calc unavailable: STAB-ish × type × Atk/SpA."""
    defs = defender.types or (None,)
    best = 0.0
    atk_stats = attacker.stats or {}
    atk_pow = max(float(atk_stats.get('atk', 100)), float(atk_stats.get('spa', 100)))
    for t in attacker.types or ():
        m = type_effectiveness(t, defs)
        # Nominal damage proxy ~ power 80
        dmg = 80.0 * m * (atk_pow / 100.0)
        if dmg > best:
            best = dmg
    if best <= 0:
        best = 1.0
    return best


def coverage_matrix(our: Sequence[PreviewMonFeat], foe: Sequence[PreviewMonFeat],
                    dmax: Optional[np.ndarray] = None) -> np.ndarray:
    mat = np.zeros((N_POOL, N_POOL), dtype=np.float32)
    for i in range(N_POOL):
        att = our[i] if i < len(our) else None
        for j in range(N_POOL):
            de = foe[j] if j < len(foe) else None
            if att is None or de is None:
                mat[i, j] = math.log(EPS)
                continue
            if dmax is not None:
                d = float(dmax[i, j])
            else:
                d = _heuristic_dmax(att, de)
            hp = float(de.max_hp or (de.stats or {}).get('hp', 100) or 100)
            mat[i, j] = math.log(d / max(hp, 1.0) + EPS)
    return mat


def encode_preview_obs(bundle: PreviewObsBundle) -> np.ndarray:
    """Build float32[512] = P1(210)+P2(210)+M(36)+ctx(56)."""
    out = np.zeros(OBS_DIM, dtype=np.float32)
    for i in range(N_POOL):
        mon = bundle.our[i] if i < len(bundle.our) else None
        out[i * MON_DIM:(i + 1) * MON_DIM] = _encode_mon(mon)
    base = N_POOL * MON_DIM  # 210
    for i in range(N_POOL):
        mon = bundle.foe[i] if i < len(bundle.foe) else None
        out[base + i * MON_DIM: base + (i + 1) * MON_DIM] = _encode_mon(mon)
    base2 = base + N_POOL * MON_DIM  # 420
    m = coverage_matrix(bundle.our, bundle.foe, bundle.dmax)
    out[base2: base2 + 36] = m.reshape(-1)
    # ctx[0:9] weather/terrain = 0; rest padding zeros (47D)
    return out
