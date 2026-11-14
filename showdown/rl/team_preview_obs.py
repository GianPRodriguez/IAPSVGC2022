"""Observación de Team Preview v2 — OTS completo (docs/team_preview_masked_ppo_spec.md §4.3).

En Team Preview (Reg M-C, Open Team Sheets) se conoce:

- **Equipo propio:** todo — especie, item, habilidad, naturaleza, SP y los cuatro
  movimientos → stats finales exactos a nivel 50.
- **Equipo rival:** la hoja abierta — especie, item, habilidad, naturaleza y
  movimientos. Los SP **no** son públicos: sus stats van como rango [SP=0, SP=32].

La observación codifica toda esa información de forma aprendible:

``obs = 6 × bloque_propio + 6 × bloque_rival + D_us→foe(36) + D_foe→us(36) + S(36)``

Cada bloque de Pokémon = rasgos semánticos (stats, tipos, forma mega, item,
habilidad, movimientos agregados) + one-hot de identidad (especie, item,
habilidad) con vocabulario congelado del pool MetaVGC
(``showdown/rl/data/preview_vocab.json``; índice 0 = desconocido).

Las matrices de pares se calculan una vez por episodio (``reset``):

- ``D_us→foe[i, j]``: daño esperado del mejor movimiento de nuestro i sobre el
  rival j, como fracción de su HP (fórmula de daño nivel 50, STAB, tipos,
  inmunidades por habilidad, items de daño, spread, precisión).
- ``D_foe→us[j, i]``: lo mismo en sentido contrario.
- ``S[i, j]``: +1 si i siempre es más rápido que j (rangos de stats, Choice
  Scarf), −1 si siempre es más lento, y un valor intermedio si depende de SP.

Los datos salen del snapshot congelado del dex Champions de Showdown
(``API/JSONs/champions_dex_snapshot.json``, ver ``scripts/export_champions_dex.js``).

El rival se reordena de forma canónica (por especie), de modo que la
observación no depende del orden en que llega su hoja. El orden propio se
conserva porque las acciones (``/team ####``) indexan esas posiciones.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from showdown.battle_bots.greedy_solo.type_chart import (
    TYPES,
    normalize_type,
    type_effectiveness,
)

OBS_VERSION = 2
N_POOL = 6
EPS = 1e-6
STAT_KEYS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
LEVEL = 50
IV = 31
SP_MAX = 32
N_TYPES = len(TYPES)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEX_PATH = _REPO_ROOT / 'API' / 'JSONs' / 'champions_dex_snapshot.json'
VOCAB_PATH = Path(__file__).resolve().parent / 'data' / 'preview_vocab.json'


def to_id(text: Any) -> str:
    return re.sub(r'[^a-z0-9]', '', str(text or '').lower())


# --------------------------------------------------------------------------- #
# Datos: dex Champions y vocabulario congelado
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _dex() -> Dict[str, Dict[str, Any]]:
    with DEX_PATH.open(encoding='utf-8') as fh:
        return json.load(fh)


_SPECIES_ALIASES = {
    'basculegionm': 'basculegion',
    'basculegionmale': 'basculegion',
    'basculegionfemale': 'basculegionf',
    'floetteeternalmega': 'floettemega',
}


def species_entry(name: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    sid = to_id(name)
    sid = _SPECIES_ALIASES.get(sid, sid)
    table = _dex()['species']
    if sid in table:
        return sid, table[sid]
    if not sid or sid == 'unknown':
        return None
    try:  # nombres raros de scrapers / poke-env
        from showdown.battle_bots.greedy_solo.species_names import resolve_species_name
        alt = to_id(resolve_species_name(str(name)))
    except Exception:
        return None
    alt = _SPECIES_ALIASES.get(alt, alt)
    if alt in table:
        return alt, table[alt]
    return None


def mega_species_id(species_id: str, item_id: str) -> Optional[str]:
    """Especie mega si ``item`` es la piedra de esta especie (Mega on en Reg M-C)."""
    item = _dex()['items'].get(item_id)
    if not item or not item.get('megaStone'):
        return None
    sp = _dex()['species'].get(species_id) or {}
    names = {to_id(sp.get('name')), to_id(sp.get('baseSpecies'))}
    for base, mega in item['megaStone'].items():
        if to_id(base) in names:
            mid = to_id(mega)
            if mid in _dex()['species']:
                return mid
    return None


@lru_cache(maxsize=1)
def load_vocab() -> Dict[str, List[str]]:
    if not VOCAB_PATH.is_file():
        raise FileNotFoundError(
            f'falta {VOCAB_PATH}; generarlo con '
            '`python -m showdown.rl.team_preview_obs --build-vocab`'
        )
    with VOCAB_PATH.open(encoding='utf-8') as fh:
        return json.load(fh)


def build_preview_vocab(teams: Sequence[Any]) -> Dict[str, List[str]]:
    """Vocabulario de identidad desde un pool de equipos (TeamMeta)."""
    species, items, abilities = set(), set(), set()
    for team in teams:
        for mon in getattr(team, 'mons', None) or []:
            ent = species_entry(mon.species)
            sid = ent[0] if ent else to_id(mon.species)
            species.add(sid)
            iid = to_id(mon.item)
            if iid:
                items.add(iid)
            if to_id(mon.ability):
                abilities.add(to_id(mon.ability))
            mid = mega_species_id(sid, iid) if iid else None
            if mid:
                for abil in (_dex()['species'][mid].get('abilities') or {}).values():
                    abilities.add(to_id(abil))
    return {
        'species': sorted(species),
        'items': sorted(items),
        'abilities': sorted(abilities),
    }


try:
    _VOCAB = load_vocab()
except FileNotFoundError:  # solo durante `--build-vocab` la primera vez
    _VOCAB = {'species': [], 'items': [], 'abilities': []}
    load_vocab.cache_clear()
_VIDX = {k: {name: i + 1 for i, name in enumerate(_VOCAB[k])}
         for k in ('species', 'items', 'abilities')}
V_SPECIES = len(_VOCAB['species']) + 1
V_ITEMS = len(_VOCAB['items']) + 1
V_ABILITIES = len(_VOCAB['abilities']) + 1


# --------------------------------------------------------------------------- #
# Tablas de items / habilidades / naturalezas
# --------------------------------------------------------------------------- #

_NATURES: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    'adamant': ('atk', 'spa'), 'bold': ('def', 'atk'), 'brave': ('atk', 'spe'),
    'calm': ('spd', 'atk'), 'careful': ('spd', 'spa'), 'gentle': ('spd', 'def'),
    'hasty': ('spe', 'def'), 'impish': ('def', 'spa'), 'jolly': ('spe', 'spa'),
    'lax': ('def', 'spd'), 'lonely': ('atk', 'def'), 'mild': ('spa', 'def'),
    'modest': ('spa', 'atk'), 'naive': ('spe', 'spd'), 'naughty': ('atk', 'spd'),
    'quiet': ('spa', 'spe'), 'rash': ('spa', 'spd'), 'relaxed': ('def', 'spe'),
    'sassy': ('spd', 'spe'), 'timid': ('spe', 'atk'),
    'bashful': (None, None), 'docile': (None, None), 'hardy': (None, None),
    'quirky': (None, None), 'serious': (None, None),
}

TYPE_BOOST_ITEMS = {
    'miracleseed': 'Grass', 'charcoal': 'Fire', 'mysticwater': 'Water',
    'fairyfeather': 'Fairy', 'blackglasses': 'Dark', 'magnet': 'Electric',
    'nevermeltice': 'Ice', 'dragonfang': 'Dragon', 'sharpbeak': 'Flying',
    'softsand': 'Ground', 'hardstone': 'Rock', 'metalcoat': 'Steel',
    'poisonbarb': 'Poison', 'spelltag': 'Ghost', 'twistedspoon': 'Psychic',
    'silverpowder': 'Bug', 'blackbelt': 'Fighting', 'silkscarf': 'Normal',
}
RESIST_BERRIES = {
    'occaberry': 'Fire', 'passhoberry': 'Water', 'wacanberry': 'Electric',
    'rindoberry': 'Grass', 'yacheberry': 'Ice', 'chopleberry': 'Fighting',
    'kebiaberry': 'Poison', 'shucaberry': 'Ground', 'cobaberry': 'Flying',
    'payapaberry': 'Psychic', 'tangaberry': 'Bug', 'chartiberry': 'Rock',
    'kasibberry': 'Ghost', 'habanberry': 'Dragon', 'colburberry': 'Dark',
    'babiriberry': 'Steel', 'roseliberry': 'Fairy', 'chilanberry': 'Normal',
}
HEAL_BERRIES = {'sitrusberry', 'figyberry', 'wikiberry', 'magoberry', 'aguavberry',
                'iapapaberry', 'oranberry'}
TERRAIN_SEEDS = {'grassyseed', 'psychicseed', 'electricseed', 'mistyseed'}
WEATHER_ROCKS = {'damprock', 'heatrock', 'smoothrock', 'icyrock'}

ITEM_FLAGS = (
    'choiceband', 'choicespecs', 'choicescarf', 'focussash', 'lifeorb',
    'expertbelt', 'typeboost', 'resistberry', 'healberry', 'lumberry',
    'leftovers', 'terrainseed', 'whiteherb', 'mentalherb', 'rockyhelmet',
    'lightclay', 'weatherrock', 'assaultvest', 'switch_item', 'megastone',
    'protective', 'other_item',
)
_SWITCH_ITEMS = {'ejectbutton', 'ejectpack', 'redcard'}
_PROTECTIVE_ITEMS = {'safetygoggles', 'covertcloak', 'clearamulet', 'abilityshield'}

ATE_ABILITIES = {'aerilate': 'Flying', 'pixilate': 'Fairy', 'refrigerate': 'Ice',
                 'galvanize': 'Electric', 'dragonize': 'Dragon', 'normalize': 'Normal'}
IMMUNITY_ABILITIES = {
    'levitate': 'Ground', 'eartheater': 'Ground', 'flashfire': 'Fire',
    'wellbakedbody': 'Fire', 'waterabsorb': 'Water', 'stormdrain': 'Water',
    'dryskin': 'Water', 'voltabsorb': 'Electric', 'lightningrod': 'Electric',
    'motordrive': 'Electric', 'sapsipper': 'Grass',
}
_WEATHER_AB = {'drizzle': 'rain', 'primordialsea': 'rain', 'drought': 'sun',
               'orichalcumpulse': 'sun', 'desolateland': 'sun', 'sandstream': 'sand',
               'snowwarning': 'snow'}
_TERRAIN_AB = {'grassysurge': 'grassy', 'seedsower': 'grassy', 'psychicsurge': 'psychic',
               'electricsurge': 'electric', 'hadronengine': 'electric', 'mistysurge': 'misty'}
_POWER_AB = {'toughclaws', 'sheerforce', 'technician', 'hugepower', 'purepower', 'strongjaw',
             'ironfist', 'megalauncher', 'sharpness', 'punkrock', 'steelworker', 'transistor',
             'waterbubble', 'gorillatactics', 'hustle', 'adaptability', 'solarpower',
             'sandforce', 'skilllink', 'rockypayload', 'dragonsmaw', 'supremeoverlord',
             'protosynthesis', 'quarkdrive', 'analytic', 'tintedlens', 'sniper', 'moxie'}
_SPEED_AB = {'unburden', 'swiftswim', 'chlorophyll', 'sandrush', 'slushrush', 'surgesurfer',
             'speedboost', 'quickfeet'}
_PUNISH_INTIM = {'defiant', 'competitive', 'rattled', 'guarddog'}
_INTIM_IMMUNE = {'innerfocus', 'oblivious', 'owntempo', 'scrappy', 'clearbody', 'whitesmoke',
                 'fullmetalbody', 'hypercutter', 'mirrorarmor', 'guarddog'}
_STATUS_SHIELD = {'goodasgold', 'magicbounce', 'purifyingsalt', 'comatose'}
_PRIORITY_BLOCK = {'armortail', 'dazzling', 'queenlymajesty'}
_AURA = {'fairyaura', 'darkaura'}
_BULK_AB = {'multiscale', 'filter', 'solidrock', 'prismarmor', 'fluffy', 'icescales',
            'thickfat', 'furcoat', 'shadowshield', 'heatproof', 'marvelscale'}

ABILITY_FLAGS = (
    'intimidate', 'rain', 'sun', 'sand', 'snow', 'grassy', 'psychic', 'electric', 'misty',
    'immunity', 'redirect', 'prankster', 'ate', 'power', 'speed', 'punish_intimidate',
    'intimidate_immune', 'status_shield', 'priority_block', 'aura', 'bulk', 'regenerator',
)

MOVE_UTILITY_FLAGS = (
    'protect', 'fakeout', 'tailwind', 'trickroom', 'redirect', 'helpinghand',
    'wideguard', 'quickguard', 'speed_drop', 'offense_drop', 'sleep', 'other_status',
    'taunt', 'encore', 'disable', 'pivot', 'setup', 'recovery', 'screens',
    'weather_move', 'terrain_move', 'perishsong', 'knockoff', 'flinch', 'crit',
    'drawback', 'allyswitch', 'ally_support',
)
_SPREAD_TARGETS = {'allAdjacentFoes', 'allAdjacent'}
_SCREENS = {'reflect', 'lightscreen', 'auroraveil'}
_ALLY_SUPPORT = {'decorate', 'coaching', 'howl', 'afteryou', 'instruct', 'lifedew',
                 'pollenpuff', 'healpulse', 'aromaticmist'}


def _ability_flags(abil: str) -> Tuple[np.ndarray, Optional[str]]:
    f = np.zeros(len(ABILITY_FLAGS), dtype=np.float32)
    idx = {k: i for i, k in enumerate(ABILITY_FLAGS)}
    if not abil:
        return f, None
    if abil == 'intimidate':
        f[idx['intimidate']] = 1.0
    if abil in _WEATHER_AB:
        f[idx[_WEATHER_AB[abil]]] = 1.0
    if abil in _TERRAIN_AB:
        f[idx[_TERRAIN_AB[abil]]] = 1.0
    atype: Optional[str] = None
    if abil in IMMUNITY_ABILITIES:
        f[idx['immunity']] = 1.0
        atype = IMMUNITY_ABILITIES[abil]
    if abil in ('stormdrain', 'lightningrod'):
        f[idx['redirect']] = 1.0
    if abil == 'prankster':
        f[idx['prankster']] = 1.0
    if abil in ATE_ABILITIES:
        f[idx['ate']] = 1.0
        atype = ATE_ABILITIES[abil]
    if abil in _POWER_AB:
        f[idx['power']] = 1.0
    if abil in _SPEED_AB:
        f[idx['speed']] = 1.0
    if abil in _PUNISH_INTIM:
        f[idx['punish_intimidate']] = 1.0
    if abil in _INTIM_IMMUNE:
        f[idx['intimidate_immune']] = 1.0
    if abil in _STATUS_SHIELD:
        f[idx['status_shield']] = 1.0
    if abil in _PRIORITY_BLOCK:
        f[idx['priority_block']] = 1.0
    if abil in _AURA:
        f[idx['aura']] = 1.0
    if abil in _BULK_AB:
        f[idx['bulk']] = 1.0
    if abil == 'regenerator':
        f[idx['regenerator']] = 1.0
    return f, atype


def _item_flags(item: str) -> Tuple[np.ndarray, Optional[str]]:
    f = np.zeros(len(ITEM_FLAGS), dtype=np.float32)
    idx = {k: i for i, k in enumerate(ITEM_FLAGS)}
    if not item:
        return f, None
    itype: Optional[str] = None
    entry = _dex()['items'].get(item) or {}
    if item in idx and item not in ('typeboost', 'resistberry', 'healberry', 'megastone'):
        f[idx[item]] = 1.0
    elif item in TYPE_BOOST_ITEMS:
        f[idx['typeboost']] = 1.0
        itype = TYPE_BOOST_ITEMS[item]
    elif item in RESIST_BERRIES:
        f[idx['resistberry']] = 1.0
        itype = RESIST_BERRIES[item]
    elif item in HEAL_BERRIES:
        f[idx['healberry']] = 1.0
    elif item in TERRAIN_SEEDS:
        f[idx['terrainseed']] = 1.0
    elif item in WEATHER_ROCKS:
        f[idx['weatherrock']] = 1.0
    elif item in _SWITCH_ITEMS:
        f[idx['switch_item']] = 1.0
    elif item in _PROTECTIVE_ITEMS:
        f[idx['protective']] = 1.0
    elif entry.get('megaStone'):
        f[idx['megastone']] = 1.0
    else:
        f[idx['other_item']] = 1.0
    return f, itype


def _type_onehot(types: Sequence[Optional[str]]) -> np.ndarray:
    v = np.zeros(N_TYPES, dtype=np.float32)
    for t in types:
        nt = normalize_type(t)
        if nt and nt in TYPES:
            v[TYPES.index(nt)] = 1.0
    return v


def _stat(base: int, sp: int, nat: float, key: str) -> int:
    inner = (2 * int(base) + IV + int(sp)) * LEVEL // 100
    if key == 'hp':
        return inner + LEVEL + 10
    return int((inner + 5) * nat)


def _nature_mult(nature: Optional[str], key: str) -> Optional[float]:
    """1.1 / 0.9 / 1.0 si la naturaleza es conocida; None si no."""
    if key == 'hp':
        return 1.0
    nat = _NATURES.get(to_id(nature))
    if nat is None:
        return None
    plus, minus = nat
    if plus == key:
        return 1.1
    if minus == key:
        return 0.9
    return 1.0


# --------------------------------------------------------------------------- #
# Datos de entrada
# --------------------------------------------------------------------------- #

@dataclass
class PreviewMonFeat:
    """Lo conocido de un Pokémon en preview.

    ``stats`` = stats finales exactos (hp..spe) cuando se conocen (equipo propio);
    ``None`` para el rival (SP ocultos → rango).
    """

    species: str
    item: Optional[str] = None
    ability: Optional[str] = None
    nature: Optional[str] = None
    moves: Tuple[str, ...] = ()
    stats: Optional[Dict[str, int]] = None
    tera_type: Optional[str] = None  # Tera off en Reg M-C: no se codifica

    def key(self) -> Tuple:
        st = tuple(int(self.stats[k]) for k in STAT_KEYS) if self.stats else None
        return (
            to_id(self.species), to_id(self.item), to_id(self.ability),
            to_id(self.nature), tuple(sorted(to_id(m) for m in self.moves if m)), st,
        )


@dataclass
class PreviewObsBundle:
    our: List[PreviewMonFeat] = field(default_factory=list)
    foe: List[PreviewMonFeat] = field(default_factory=list)
    # Opcional: matriz (6,6) de daño us→foe precomputada (fracción de HP).
    dmax: Optional[np.ndarray] = None


def mon_feat_from_species(species: str, **kwargs) -> PreviewMonFeat:
    """Solo especie (+ campos opcionales). Útil cuando no hay hoja abierta."""
    return PreviewMonFeat(
        species=str(species),
        item=kwargs.get('item'),
        ability=kwargs.get('ability'),
        nature=kwargs.get('nature'),
        moves=tuple(kwargs.get('moves') or ()),
        stats=kwargs.get('stats'),
    )


def stats_from_sp(species: str, sp: Dict[str, int], nature: Optional[str]) -> Optional[Dict[str, int]]:
    ent = species_entry(species)
    if ent is None:
        return None
    base = ent[1]['baseStats']
    out = {}
    for k in STAT_KEYS:
        nat = _nature_mult(nature, k)
        out[k] = _stat(base[k], int(sp.get(k, 0) or 0), 1.0 if nat is None else nat, k)
    return out


def mon_feat_from_monset(mon: Any, *, public_only: bool) -> PreviewMonFeat:
    """Desde ``team_factory.MonSet``. ``public_only`` = hoja rival (sin SP)."""
    stats = None
    if not public_only:
        stats = stats_from_sp(mon.species, dict(getattr(mon, 'sp', None) or {}), mon.nature)
    return PreviewMonFeat(
        species=str(mon.species),
        item=getattr(mon, 'item', None) or None,
        ability=getattr(mon, 'ability', None) or None,
        nature=getattr(mon, 'nature', None) or None,
        moves=tuple(getattr(mon, 'moves', None) or ()),
        stats=stats,
    )


def _pe_item(mon: Any) -> Optional[str]:
    item = getattr(mon, 'item', None)
    if not item or item == 'unknown_item':
        return None
    return str(item)


def mon_feat_from_pokeenv(mon: Any, *, own: bool) -> PreviewMonFeat:
    """Desde ``poke_env.battle.Pokemon`` en preview.

    Propio: viene del teambuilder (SP en ``evs`` + naturaleza) → stats exactos.
    Rival: hoja abierta (``showteam``); sus ``evs`` se IGNORAN (SP no públicos).
    """
    species = str(getattr(mon, 'species', '') or 'unknown')
    moves = tuple(str(m) for m in (getattr(mon, 'moves', None) or {}).keys())
    nature = getattr(mon, 'nature', None)
    stats = None
    if own:
        evs = getattr(mon, 'evs', None)
        if evs and len(evs) == 6 and nature:
            sp = {k: int(v or 0) for k, v in zip(STAT_KEYS, evs)}
            stats = stats_from_sp(species, sp, nature)
        if stats is None:
            st = dict(getattr(mon, 'stats', None) or {})
            hp = getattr(mon, 'max_hp', None)
            if hp and all(st.get(k) for k in STAT_KEYS[1:]):
                stats = {'hp': int(hp), **{k: int(st[k]) for k in STAT_KEYS[1:]}}
    return PreviewMonFeat(
        species=species,
        item=_pe_item(mon),
        ability=getattr(mon, 'ability', None) or None,
        nature=nature or None,
        moves=moves,
        stats=stats,
    )


def mon_feat_from_request_row(row: Dict[str, Any]) -> PreviewMonFeat:
    """Desde una fila ``side.pokemon`` del |request| de Showdown (equipo propio)."""
    details = str(row.get('details') or '')
    species = details.partition(',')[0] or 'unknown'
    stats = None
    raw = dict(row.get('stats') or {})
    cond = str(row.get('condition') or '')
    try:
        max_hp = int(cond.split('/')[1].split()[0]) if '/' in cond else None
    except (ValueError, IndexError):
        max_hp = None
    if max_hp and all(raw.get(k) for k in STAT_KEYS[1:]):
        stats = {'hp': max_hp, **{k: int(raw[k]) for k in STAT_KEYS[1:]}}
    return PreviewMonFeat(
        species=species,
        item=row.get('item') or None,
        ability=row.get('baseAbility') or row.get('ability') or None,
        moves=tuple(row.get('moves') or ()),
        stats=stats,
    )


# --------------------------------------------------------------------------- #
# Perfil derivado por Pokémon (cacheado)
# --------------------------------------------------------------------------- #

@dataclass
class _MoveInfo:
    mid: str
    type: str
    category: str  # Physical / Special
    power: float  # BP efectivo (multihit, -ate)
    accuracy: float
    priority: int
    spread: bool
    offensive_stat: str  # atk / spa / def
    use_target_atk: bool
    defensive_stat: str  # def / spd
    fixed_damage: Optional[int]


@dataclass
class _Profile:
    present: bool
    sem: np.ndarray
    species_idx: int
    item_idx: int
    ability_idx: Tuple[int, ...]
    types: Tuple[str, ...]
    smin: Dict[str, float]
    smax: Dict[str, float]
    item: str
    ability: str
    damaging: List[_MoveInfo]


SEM_DIM = (
    1                       # present
    + 1                     # stats exactos conocidos
    + 6 + 6 + 6             # base (forma efectiva), stat min, stat max
    + N_TYPES               # tipos forma efectiva
    + 1 + 6 + N_TYPES       # puede megaevolucionar, base y tipos pre-mega
    + len(ITEM_FLAGS) + N_TYPES
    + len(ABILITY_FLAGS) + N_TYPES
    + N_TYPES + 4           # mejor poder por tipo; phys/spec/spread/prioridad
    + len(MOVE_UTILITY_FLAGS)
)
MON_DIM = SEM_DIM + V_SPECIES + V_ITEMS + V_ABILITIES
PAIR_DIM = 3 * N_POOL * N_POOL
OBS_DIM = 2 * N_POOL * MON_DIM + PAIR_DIM


def _move_infos(moves: Sequence[str], ability: str, types: Sequence[str]
                ) -> Tuple[List[_MoveInfo], np.ndarray, np.ndarray]:
    """Movimientos dañinos + agregados (poder por tipo, 4 mejores) + utilidades."""
    table = _dex()['moves']
    util = np.zeros(len(MOVE_UTILITY_FLAGS), dtype=np.float32)
    uidx = {k: i for i, k in enumerate(MOVE_UTILITY_FLAGS)}
    by_type = np.zeros(N_TYPES, dtype=np.float32)
    best = np.zeros(4, dtype=np.float32)  # phys, spec, spread, priority
    out: List[_MoveInfo] = []
    ate = ATE_ABILITIES.get(ability)
    for raw in moves:
        mid = to_id(raw)
        m = table.get(mid)
        if not m:
            continue
        prio = int(m.get('priority') or 0)
        if ability == 'prankster' and m.get('category') == 'Status':
            prio += 1
        secs = m.get('secondaries') or []
        # --- utilidades ------------------------------------------------------
        if m.get('stallingMove'):
            util[uidx['protect']] = 1.0
        if mid in ('fakeout', 'firstimpression') or mid == 'upperhand':
            util[uidx['fakeout']] = 1.0
        side = m.get('sideCondition')
        if side == 'tailwind':
            util[uidx['tailwind']] = 1.0
        if side == 'wideguard':
            util[uidx['wideguard']] = 1.0
        if side == 'quickguard':
            util[uidx['quickguard']] = 1.0
        if side in _SCREENS:
            util[uidx['screens']] = 1.0
        if m.get('pseudoWeather') == 'trickroom':
            util[uidx['trickroom']] = 1.0
        vol = m.get('volatileStatus')
        if vol in ('followme', 'ragepowder', 'spotlight'):
            util[uidx['redirect']] = 1.0
        if vol == 'helpinghand':
            util[uidx['helpinghand']] = 1.0
        if vol == 'taunt':
            util[uidx['taunt']] = 1.0
        if vol == 'encore':
            util[uidx['encore']] = 1.0
        if vol in ('disable', 'imprison', 'torment') or mid == 'imprison':
            util[uidx['disable']] = 1.0
        target_boosts = dict(m.get('boosts') or {}) if m.get('target') not in ('self', 'allySide', 'adjacentAlly', 'allies') else {}
        for s in secs:
            if (s.get('chance') or 0) >= 50 and s.get('boosts'):
                for k, v in s['boosts'].items():
                    target_boosts[k] = target_boosts.get(k, 0) + v
            if s.get('volatileStatus') == 'flinch' and mid not in ('fakeout',):
                util[uidx['flinch']] = 1.0
        if target_boosts.get('spe', 0) < 0:
            util[uidx['speed_drop']] = 1.0
        if target_boosts.get('atk', 0) < 0 or target_boosts.get('spa', 0) < 0:
            util[uidx['offense_drop']] = 1.0
        status = m.get('status')
        if status == 'slp':
            util[uidx['sleep']] = 1.0
        elif status:
            util[uidx['other_status']] = 1.0
        if m.get('selfSwitch'):
            util[uidx['pivot']] = 1.0
        self_boosts = dict(m.get('selfBoosts') or {})
        if m.get('target') == 'self' and m.get('boosts'):
            self_boosts.update(m['boosts'])
        if any(v > 0 for v in self_boosts.values()):
            util[uidx['setup']] = 1.0
        if any(v < 0 for v in self_boosts.values()) or m.get('recoil'):
            util[uidx['drawback']] = 1.0
        if m.get('heal') or m.get('drain'):
            util[uidx['recovery']] = 1.0
        if m.get('weather'):
            util[uidx['weather_move']] = 1.0
        if m.get('terrain'):
            util[uidx['terrain_move']] = 1.0
        if mid == 'perishsong':
            util[uidx['perishsong']] = 1.0
        if mid == 'knockoff':
            util[uidx['knockoff']] = 1.0
        if (m.get('critRatio') or 1) >= 2 or m.get('willCrit'):
            util[uidx['crit']] = 1.0
        if mid == 'allyswitch':
            util[uidx['allyswitch']] = 1.0
        if mid in _ALLY_SUPPORT:
            util[uidx['ally_support']] = 1.0
        # --- daño ------------------------------------------------------------
        cat = m.get('category')
        if cat not in ('Physical', 'Special'):
            continue
        mtype = str(m.get('type') or 'Normal')
        bp = float(m.get('basePower') or 0)
        fixed = m.get('damage')
        fixed_dmg = LEVEL if fixed == 'level' else (int(fixed) if isinstance(fixed, (int, float)) else None)
        if bp <= 0 and fixed_dmg is None:
            bp = 60.0  # poder variable (Low Kick, Heavy Slam, …): valor típico
        if ate and mtype == 'Normal':
            mtype = ate
            bp *= 1.2
        hits = m.get('multihit')
        if isinstance(hits, list):
            bp *= 5.0 if ability == 'skilllink' else 3.1
        elif isinstance(hits, (int, float)):
            bp *= float(hits)
        acc_raw = m.get('accuracy')
        acc = 1.0 if (acc_raw is None or acc_raw >= 100) else float(acc_raw) / 100.0
        if ability == 'noguard':
            acc = 1.0
        spread = m.get('target') in _SPREAD_TARGETS
        off = str(m.get('overrideOffensiveStat') or ('atk' if cat == 'Physical' else 'spa'))
        dfs = str(m.get('overrideDefensiveStat') or ('def' if cat == 'Physical' else 'spd'))
        info = _MoveInfo(
            mid=mid, type=mtype, category=cat, power=bp, accuracy=acc, priority=prio,
            spread=spread, offensive_stat=off,
            use_target_atk=m.get('overrideOffensivePokemon') == 'target',
            defensive_stat=dfs, fixed_damage=fixed_dmg,
        )
        out.append(info)
        stab = (2.0 if ability == 'adaptability' else 1.5) if mtype in types else 1.0
        eff_power = bp * stab * acc
        nt = normalize_type(mtype)
        if nt in TYPES:
            ti = TYPES.index(nt)
            by_type[ti] = max(by_type[ti], eff_power / 250.0)
        if cat == 'Physical':
            best[0] = max(best[0], eff_power / 250.0)
        else:
            best[1] = max(best[1], eff_power / 250.0)
        if spread:
            best[2] = max(best[2], 0.75 * eff_power / 250.0)
        if prio > 0:
            best[3] = max(best[3], eff_power / 250.0)
    return out, np.concatenate([by_type, best]), util


@lru_cache(maxsize=8192)
def _profile_cached(key: Tuple) -> _Profile:
    species, item, ability, nature, moves, exact = key
    ent = species_entry(species)
    if ent is None:
        return _Profile(
            present=False, sem=np.zeros(SEM_DIM, dtype=np.float32), species_idx=0,
            item_idx=0, ability_idx=(), types=(), smin={k: 100.0 for k in STAT_KEYS},
            smax={k: 100.0 for k in STAT_KEYS}, item=item, ability=ability, damaging=[],
        )
    sid, sp = ent
    base0 = sp['baseStats']
    types0 = tuple(sp.get('types') or ())
    mid = mega_species_id(sid, item) if item else None
    eff = _dex()['species'][mid] if mid else sp
    base1 = eff['baseStats']
    types1 = tuple(eff.get('types') or ())
    eff_ability = to_id((eff.get('abilities') or {}).get('0')) if mid else ability
    eff_ability = eff_ability or ability

    # --- rangos de stats (forma efectiva) ------------------------------------
    smin: Dict[str, float] = {}
    smax: Dict[str, float] = {}
    if exact is not None:
        for k, v in zip(STAT_KEYS, exact):
            # mega: el stat interno sube Δbase (2Δb·50/100); naturaleza ≈ 1.
            v2 = float(v) + float(base1[k] - base0[k])
            smin[k] = smax[k] = v2
    else:
        for k in STAT_KEYS:
            nat = _nature_mult(nature, k)
            lo = 0.9 if nat is None else nat
            hi = 1.1 if nat is None else nat
            smin[k] = float(_stat(base1[k], 0, lo, k))
            smax[k] = float(_stat(base1[k], SP_MAX, hi, k))

    ifl, itype = _item_flags(item)
    afl, atype = _ability_flags(eff_ability)
    dmg_moves, agg, util = _move_infos(moves, eff_ability, types1)

    parts = [
        np.asarray([1.0, 1.0 if exact is not None else 0.0], dtype=np.float32),
        np.asarray([base1[k] / 255.0 for k in STAT_KEYS], dtype=np.float32),
        np.asarray([smin[k] / 250.0 for k in STAT_KEYS], dtype=np.float32),
        np.asarray([smax[k] / 250.0 for k in STAT_KEYS], dtype=np.float32),
        _type_onehot(types1),
        np.asarray([1.0 if mid else 0.0], dtype=np.float32),
        np.asarray([base0[k] / 255.0 for k in STAT_KEYS], dtype=np.float32),
        _type_onehot(types0),
        ifl, _type_onehot([itype] if itype else []),
        afl, _type_onehot([atype] if atype else []),
        agg, util,
    ]
    sem = np.concatenate(parts).astype(np.float32)
    assert sem.shape[0] == SEM_DIM, (sem.shape, SEM_DIM)
    abil_idx = tuple(sorted({_VIDX['abilities'].get(a, 0) for a in (ability, eff_ability) if a}))
    return _Profile(
        present=True, sem=sem,
        species_idx=_VIDX['species'].get(sid, 0),
        item_idx=_VIDX['items'].get(item, 0) if item else 0,
        ability_idx=abil_idx,
        types=types1, smin=smin, smax=smax, item=item, ability=eff_ability,
        damaging=dmg_moves,
    )


def mon_profile(mon: Optional[PreviewMonFeat]) -> Optional[_Profile]:
    if mon is None:
        return None
    return _profile_cached(mon.key())


_OFF_CAN_MEGA = 2 + 18 + N_TYPES
_OFF_ABIL = _OFF_CAN_MEGA + 1 + 6 + N_TYPES + len(ITEM_FLAGS) + N_TYPES
_OFF_AGG = SEM_DIM - len(MOVE_UTILITY_FLAGS) - (N_TYPES + 4)
_OFF_UTIL = SEM_DIM - len(MOVE_UTILITY_FLAGS)
ROLE_NAMES = (
    'fakeout', 'tailwind', 'trickroom', 'redirect', 'intimidate', 'speed_drop',
    'spread', 'priority', 'protect', 'sleep', 'helpinghand', 'wideguard', 'mega',
)


def role_flags(mon: Optional[PreviewMonFeat]) -> Dict[str, bool]:
    """Roles interpretables de un Pokémon (para métricas de estrategia en la tesis)."""
    prof = mon_profile(mon)
    if prof is None or not prof.present:
        return {r: False for r in ROLE_NAMES}
    sem = prof.sem
    u = {k: bool(sem[_OFF_UTIL + i] > 0.5) for i, k in enumerate(MOVE_UTILITY_FLAGS)}
    a = {k: bool(sem[_OFF_ABIL + i] > 0.5) for i, k in enumerate(ABILITY_FLAGS)}
    agg = sem[_OFF_AGG:_OFF_UTIL]
    return {
        'fakeout': u['fakeout'],
        'tailwind': u['tailwind'],
        'trickroom': u['trickroom'],
        'redirect': u['redirect'] or a['redirect'],
        'intimidate': a['intimidate'],
        'speed_drop': u['speed_drop'],
        'spread': bool(agg[N_TYPES + 2] > 0),
        'priority': bool(agg[N_TYPES + 3] > 0),
        'protect': u['protect'],
        'sleep': u['sleep'],
        'helpinghand': u['helpinghand'],
        'wideguard': u['wideguard'],
        'mega': bool(sem[_OFF_CAN_MEGA] > 0.5),
    }


def _encode_mon(mon: Optional[PreviewMonFeat]) -> np.ndarray:
    v = np.zeros(MON_DIM, dtype=np.float32)
    prof = mon_profile(mon)
    if prof is None or not prof.present:
        return v
    v[:SEM_DIM] = prof.sem
    o = SEM_DIM
    v[o + prof.species_idx] = 1.0
    o += V_SPECIES
    if prof.item:
        v[o + prof.item_idx] = 1.0
    o += V_ITEMS
    for ai in prof.ability_idx:
        v[o + ai] = 1.0
    return v


# --------------------------------------------------------------------------- #
# Matrices de pares
# --------------------------------------------------------------------------- #

def _mid(lo: Dict[str, float], hi: Dict[str, float], k: str) -> float:
    return 0.5 * (lo[k] + hi[k])


def expected_damage_frac(att: _Profile, dfn: _Profile) -> float:
    """Daño esperado del mejor movimiento de ``att`` sobre ``dfn`` (fracción de HP)."""
    if not att.present or not dfn.present or not att.damaging:
        return 0.0
    hp = max(1.0, _mid(dfn.smin, dfn.smax, 'hp'))
    immune_type = IMMUNITY_ABILITIES.get(dfn.ability)
    best = 0.0
    for mv in att.damaging:
        if immune_type and normalize_type(mv.type) == immune_type and att.ability != 'moldbreaker':
            continue
        eff = type_effectiveness(mv.type, dfn.types)
        if eff <= 0:
            continue
        if mv.fixed_damage is not None:
            dmg = float(mv.fixed_damage)
        else:
            if mv.use_target_atk:
                a = _mid(dfn.smin, dfn.smax, 'atk')
            else:
                a = _mid(att.smin, att.smax, mv.offensive_stat)
                if mv.offensive_stat == 'atk' and att.ability in ('hugepower', 'purepower'):
                    a *= 2.0
                if mv.category == 'Physical' and att.item == 'choiceband':
                    a *= 1.5
                if mv.category == 'Special' and att.item == 'choicespecs':
                    a *= 1.5
            d = max(1.0, _mid(dfn.smin, dfn.smax, mv.defensive_stat))
            if mv.defensive_stat == 'spd' and dfn.item == 'assaultvest':
                d *= 1.5
            base = math.floor(math.floor(22.0 * mv.power * a / d) / 50.0) + 2.0
            mod = 0.925  # media del factor aleatorio 0.85–1.00
            if mv.spread:
                mod *= 0.75
            if mv.type in att.types:
                mod *= 2.0 if att.ability == 'adaptability' else 1.5
            mod *= eff
            if att.item == 'lifeorb':
                mod *= 1.3
            elif att.item == 'expertbelt' and eff > 1.0:
                mod *= 1.2
            elif TYPE_BOOST_ITEMS.get(att.item) == normalize_type(mv.type):
                mod *= 1.2
            if eff > 1.0 and RESIST_BERRIES.get(dfn.item) == normalize_type(mv.type):
                mod *= 0.5
            if dfn.ability in ('multiscale', 'shadowshield'):
                mod *= 0.5
            elif dfn.ability in ('filter', 'solidrock', 'prismarmor') and eff > 1.0:
                mod *= 0.75
            dmg = base * mod
        best = max(best, dmg * mv.accuracy / hp)
    return float(min(best, 2.0))


def _speed_range(p: _Profile) -> Tuple[float, float]:
    mult = 1.5 if p.item == 'choicescarf' else 1.0
    return p.smin['spe'] * mult, p.smax['spe'] * mult


def speed_relation(a: _Profile, b: _Profile) -> float:
    """+1 siempre más rápido, −1 siempre más lento, (−0.5, 0.5) si depende."""
    if not a.present or not b.present:
        return 0.0
    alo, ahi = _speed_range(a)
    blo, bhi = _speed_range(b)
    if alo > bhi:
        return 1.0
    if ahi < blo:
        return -1.0
    am, bm = 0.5 * (alo + ahi), 0.5 * (blo + bhi)
    return float(0.5 * (am - bm) / max(am, bm, 1.0))


def pair_matrices(our: Sequence[PreviewMonFeat], foe: Sequence[PreviewMonFeat]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(D_us→foe, D_foe→us, S) con forma (6,6); índices [nuestro i, rival j]."""
    po = [mon_profile(our[i]) if i < len(our) else None for i in range(N_POOL)]
    pf = [mon_profile(foe[j]) if j < len(foe) else None for j in range(N_POOL)]
    d_of = np.zeros((N_POOL, N_POOL), dtype=np.float32)
    d_fo = np.zeros((N_POOL, N_POOL), dtype=np.float32)
    s = np.zeros((N_POOL, N_POOL), dtype=np.float32)
    for i in range(N_POOL):
        for j in range(N_POOL):
            a, b = po[i], pf[j]
            if a is None or b is None:
                continue
            d_of[i, j] = expected_damage_frac(a, b)
            d_fo[i, j] = expected_damage_frac(b, a)
            s[i, j] = speed_relation(a, b)
    return d_of, d_fo, s


def coverage_matrix(our: Sequence[PreviewMonFeat], foe: Sequence[PreviewMonFeat],
                    dmax: Optional[np.ndarray] = None) -> np.ndarray:
    """log(daño esperado us→foe + ε), (6,6). Lo usa el proxy del backend mock."""
    d = np.asarray(dmax, dtype=np.float32) if dmax is not None else pair_matrices(our, foe)[0]
    return np.log(d + EPS).astype(np.float32)


def canonical_foe_order(foe: Sequence[PreviewMonFeat]) -> List[PreviewMonFeat]:
    """Orden del rival independiente de cómo llegó su hoja (por especie, item)."""
    def _k(m: PreviewMonFeat):
        prof = mon_profile(m)
        present = bool(prof and prof.present)
        return (0 if present else 1, to_id(m.species), to_id(m.item))
    return sorted(list(foe), key=_k)


def encode_preview_obs(bundle: PreviewObsBundle) -> np.ndarray:
    """float32[OBS_DIM] = 6 bloques propios + 6 rivales (orden canónico) + matrices."""
    out = np.zeros(OBS_DIM, dtype=np.float32)
    our = list(bundle.our)[:N_POOL]
    foe = canonical_foe_order(list(bundle.foe)[:N_POOL])
    for i in range(N_POOL):
        mon = our[i] if i < len(our) else None
        out[i * MON_DIM:(i + 1) * MON_DIM] = _encode_mon(mon)
    base = N_POOL * MON_DIM
    for j in range(N_POOL):
        mon = foe[j] if j < len(foe) else None
        out[base + j * MON_DIM: base + (j + 1) * MON_DIM] = _encode_mon(mon)
    base2 = 2 * N_POOL * MON_DIM
    d_of, d_fo, s = pair_matrices(our, foe)
    if bundle.dmax is not None and bundle.foe:
        # dmax viene en el orden original del rival: reordenar columnas.
        order = [list(bundle.foe).index(m) for m in foe]
        d_of = np.asarray(bundle.dmax, dtype=np.float32)[:, order]
    out[base2: base2 + 36] = d_of.reshape(-1) / 2.0
    out[base2 + 36: base2 + 72] = d_fo.reshape(-1) / 2.0
    out[base2 + 72: base2 + 108] = s.reshape(-1)
    return out


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description='Utilidades de la observación de Team Preview')
    ap.add_argument('--build-vocab', action='store_true',
                    help='Regenera showdown/rl/data/preview_vocab.json desde el pool MetaVGC')
    args = ap.parse_args(argv)
    if args.build_vocab:
        from showdown.inference_sim.team_factory import active_teams
        vocab = build_preview_vocab(list(active_teams()))
        VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        VOCAB_PATH.write_text(json.dumps(vocab, indent=1) + '\n', encoding='utf-8')
        print(f'OK {VOCAB_PATH} ' + ' '.join(f'{k}={len(v)}' for k, v in vocab.items()))
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
