"""Equipos Reg M-C basados en MetaVGC + empaquetado Showdown (EVs = Stat Points)."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from showdown.stat_inference import STAT_KEYS, champions_stat
from teams.team_converter import normalize_name

# Fuente: https://metavgc.com/teams/featured/regulation-mc

# Pastes / MetaVGC a veces usan -M; poke-env dex = basculegion (no basculegionm).
_SPECIES_EXPORT_ALIASES = {
    'basculegion-m': 'Basculegion',
    'basculegion-male': 'Basculegion',
    'basculegionm': 'Basculegion',
    'basculegion-female': 'Basculegion-F',
}

# MetaVGC / pastes: nombre colloquial → id Showdown (p.ej. mysticfire ≠ mysticalfire).
_MOVE_EXPORT_ALIASES = {
    'mystic fire': 'Mystical Fire',
    'mysticfire': 'Mystical Fire',
}

_REGMC_WARNED = False
_REGMC_REPORTED: set = set()


def canonicalize_species(species: str) -> str:
    """Nombre de especie seguro para export Showdown / poke-env / Champions."""
    raw = (species or '').strip()
    if not raw:
        return raw
    low = raw.lower()
    if low in _SPECIES_EXPORT_ALIASES:
        return _SPECIES_EXPORT_ALIASES[low]
    compact = ''.join(ch for ch in low if ch.isalnum())
    for src, dst in _SPECIES_EXPORT_ALIASES.items():
        if ''.join(ch for ch in src if ch.isalnum()) == compact:
            return dst
    return raw


def canonicalize_move(move: str) -> str:
    """Nombre de move seguro para |utm| / validación Showdown."""
    raw = (move or '').strip()
    if not raw:
        return raw
    low = raw.lower()
    if low in _MOVE_EXPORT_ALIASES:
        return _MOVE_EXPORT_ALIASES[low]
    compact = ''.join(ch for ch in low if ch.isalnum())
    for src, dst in _MOVE_EXPORT_ALIASES.items():
        if ''.join(ch for ch in src if ch.isalnum()) == compact:
            return dst
    return raw


@dataclass
class MonSet:
    species: str
    item: str
    ability: str
    nature: str
    moves: List[str]
    sp: Dict[str, int]
    gender: str = ''
    tera: str = ''

    def species_id(self) -> str:
        return normalize_name(self.species)

    def finals(self) -> Dict[str, int]:
        from API.PokemonData import getPkmnInfo

        info = getPkmnInfo(self.species)
        if not info:
            bases = {k: 1 for k in STAT_KEYS}
        else:
            bases = {
                'hp': info[2], 'atk': info[3], 'def': info[4],
                'spa': info[5], 'spd': info[6], 'spe': info[7],
            }
        return {
            k: champions_stat(int(bases[k] or 1), int(self.sp.get(k, 0)), self.nature, k)
            for k in STAT_KEYS
        }


@dataclass
class TeamMeta:
    name: str
    source_url: str
    mons: List[MonSet]
    true_sp: Dict[str, Dict[str, int]] = field(default_factory=dict)
    true_final: Dict[str, Dict[str, int]] = field(default_factory=dict)
    natures: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for mon in self.mons:
            mon.species = canonicalize_species(mon.species)
            mon.moves = [canonicalize_move(m) for m in mon.moves]
        self.true_sp = {}
        self.true_final = {}
        self.natures = {}
        for mon in self.mons:
            sid = mon.species_id()
            self.true_sp[sid] = {k: int(mon.sp.get(k, 0)) for k in STAT_KEYS}
            self.true_final[sid] = mon.finals()
            self.natures[sid] = mon.nature


def _sp(hp=0, atk=0, df=0, spa=0, spd=0, spe=0) -> Dict[str, int]:
    return {'hp': hp, 'atk': atk, 'def': df, 'spa': spa, 'spd': spd, 'spe': spe}


def _export_mon(mon: MonSet) -> str:
    # Sin línea Gender: (poke-env la malparsea). Basculegion base = macho.
    species = canonicalize_species(mon.species)
    lines = [f"{species} @ {mon.item}"]
    lines.append(f"Ability: {mon.ability}")
    lines.append('Level: 50')
    if mon.tera:
        lines.append(f"Tera Type: {mon.tera}")
    labels = {'hp': 'HP', 'atk': 'Atk', 'def': 'Def', 'spa': 'SpA', 'spd': 'SpD', 'spe': 'Spe'}
    ev_parts = []
    for k in STAT_KEYS:
        v = int(mon.sp.get(k, 0) or 0)
        if v:
            ev_parts.append(f"{v} {labels[k]}")
    if ev_parts:
        lines.append(f"EVs: {' / '.join(ev_parts)}")
    lines.append(f"{mon.nature} Nature")
    for move in mon.moves:
        lines.append(f"- {canonicalize_move(move)}")
    return '\n'.join(lines)


def to_showdown_export(meta: TeamMeta) -> str:
    return '\n\n'.join(_export_mon(m) for m in meta.mons)


SABLEYEVGC = TeamMeta(
    name='SableyeVGC Kickoff Cup #1',
    source_url=(
        'https://metavgc.com/teams/'
        'salamence-mega-rillaboom-by-sableyevgc-at-r-vgc-regulation-m-c-kickoff-cup-place-1st'
    ),
    mons=[
        MonSet(
            'Salamence', 'Salamencite', 'Intimidate', 'Naive',
            ['Hyper Voice', 'Double-Edge', 'Tailwind', 'Protect'],
            _sp(hp=1, atk=21, spa=15, spe=29),
        ),
        MonSet(
            'Rillaboom', 'Sitrus Berry', 'Grassy Surge', 'Adamant',
            ['Wood Hammer', 'Grassy Glide', 'U-turn', 'Fake Out'],
            _sp(hp=18, atk=25, df=5, spd=10, spe=8),
        ),
        MonSet(
            'Sneasler', 'Grassy Seed', 'Unburden', 'Adamant',
            ['Dire Claw', 'Close Combat', 'Fake Out', 'Protect'],
            _sp(hp=16, atk=20, spe=30),
        ),
        MonSet(
            'Arcanine-Hisui', 'Focus Sash', 'Rock Head', 'Jolly',
            ['Flare Blitz', 'Head Smash', 'Extreme Speed', 'Protect'],
            _sp(hp=2, atk=32, spe=32),
        ),
        MonSet(
            'Kingambit', 'Chople Berry', 'Defiant', 'Adamant',
            ['Kowtow Cleave', 'Iron Head', 'Low Kick', 'Sucker Punch'],
            _sp(hp=32, atk=25, df=2, spd=6, spe=1),
        ),
        MonSet(
            'Basculegion', 'Mystic Water', 'Adaptability', 'Adamant',
            ['Wave Crash', 'Last Respects', 'Aqua Jet', 'Protect'],
            _sp(hp=4, atk=23, df=13, spe=26),
            gender='M',
        ),
    ],
)

ROBIN_VGC = TeamMeta(
    name='Robin_VGC Rookidee Qualifier #1',
    source_url=(
        'https://metavgc.com/teams/'
        'salamence-mega-floette-mega-by-robin-vgc-at-reg-m-c-qualifier-for-rookideevgc-grand-tourney-place-1st'
    ),
    mons=[
        MonSet(
            'Salamence', 'Salamencite', 'Intimidate', 'Jolly',
            ['Double-Edge', 'Dragon Dance', 'Roost', 'Protect'],
            _sp(hp=2, atk=32, spe=32),
        ),
        MonSet(
            'Floette-Eternal', 'Floettite', 'Flower Veil', 'Modest',
            ['Moonblast', 'Dazzling Gleam', 'Calm Mind', 'Protect'],
            _sp(hp=2, spa=32, spe=32),
        ),
        MonSet(
            'Sneasler', 'Focus Sash', 'Unburden', 'Jolly',
            ['Dire Claw', 'Close Combat', 'Fake Out', 'Protect'],
            _sp(hp=2, atk=32, spe=32),
        ),
        MonSet(
            'Kingambit', 'Life Orb', 'Defiant', 'Adamant',
            ['Kowtow Cleave', 'Swords Dance', 'Sucker Punch', 'Protect'],
            _sp(hp=32, atk=32, df=2),
        ),
        MonSet(
            'Incineroar', 'Sitrus Berry', 'Intimidate', 'Careful',
            ['Flare Blitz', 'Parting Shot', 'Helping Hand', 'Fake Out'],
            _sp(hp=32, atk=2, spd=32),
        ),
        MonSet(
            'Sinistcha', 'Colbur Berry', 'Hospitality', 'Bold',
            ['Matcha Gotcha', 'Rage Powder', 'Trick Room', 'Protect'],
            _sp(hp=32, atk=2, df=32),
        ),
    ],
)

ROID_BLANCAFLOR = TeamMeta(
    name='Roid Blancaflor Mid-Week Smoke #1',
    source_url=(
        'https://metavgc.com/teams/'
        'basculegion-male-farigiraf-by-roid-blancaflor-at-mid-week-smoke-10-place-1st'
    ),
    mons=[
        MonSet(
            'Basculegion', 'Choice Scarf', 'Adaptability', 'Jolly',
            ['Aqua Jet', 'Last Respects', 'Wave Crash', 'Flip Turn'],
            _sp(hp=2, atk=32, spe=32),
            gender='M',
        ),
        MonSet(
            'Farigiraf', 'Sitrus Berry', 'Armor Tail', 'Modest',
            ['Psychic', 'Helping Hand', 'Trick Room', 'Imprison'],
            _sp(hp=29, df=14, spd=23),
        ),
        MonSet(
            'Blaziken', 'Life Orb', 'Speed Boost', 'Rash',
            ['Flare Blitz', 'Close Combat', 'Protect', 'Heat Wave'],
            _sp(hp=2, atk=32, spe=32),
        ),
        MonSet(
            'Whimsicott', 'Focus Sash', 'Prankster', 'Timid',
            ['Moonblast', 'Energy Ball', 'Tailwind', 'Protect'],
            _sp(hp=2, spa=32, spe=32),
        ),
        MonSet(
            'Floette-Eternal', 'Floettite', 'Flower Veil', 'Timid',
            ['Moonblast', 'Dazzling Gleam', 'Light of Ruin', 'Protect'],
            _sp(hp=2, spa=32, spe=32),
        ),
        MonSet(
            'Salamence', 'Yache Berry', 'Intimidate', 'Timid',
            ['Hurricane', 'Flamethrower', 'Tailwind', 'Protect'],
            _sp(hp=2, spa=32, spe=32),
        ),
    ],
)

METAVGC_TEAMS: List[TeamMeta] = [SABLEYEVGC, ROBIN_VGC, ROID_BLANCAFLOR]

_JSON_PATH = Path(__file__).resolve().parent / 'metavgc_regmc_teams.json'


def _mon_from_dict(d: dict) -> MonSet:
    sp = d.get('sp') or {}
    return MonSet(
        species=canonicalize_species(d['species']),
        item=d['item'],
        ability=d.get('ability') or '',
        nature=d.get('nature') or 'Serious',
        moves=list(d.get('moves') or []),
        sp={
            'hp': int(sp.get('hp', 0) or 0),
            'atk': int(sp.get('atk', 0) or 0),
            'def': int(sp.get('def', 0) or 0),
            'spa': int(sp.get('spa', 0) or 0),
            'spd': int(sp.get('spd', 0) or 0),
            'spe': int(sp.get('spe', 0) or 0),
        },
        gender=d.get('gender') or '',
        tera=d.get('tera') or '',
    )


def _item_id(item: str) -> str:
    return ''.join(ch for ch in (item or '').lower() if ch.isalnum())


def team_passes_item_clause(meta: TeamMeta) -> bool:
    """Reg M-C / VGC: como máximo 1 de cada ítem (Item Clause)."""
    seen = set()
    for mon in meta.mons or []:
        iid = _item_id(getattr(mon, 'item', '') or '')
        if not iid:
            continue
        if iid in seen:
            return False
        seen.add(iid)
    return True


def load_scraped_teams(path: Optional[Path] = None) -> List[TeamMeta]:
    p = path or _JSON_PATH
    if not p.is_file():
        return []
    raw = json.loads(p.read_text(encoding='utf-8'))
    out: List[TeamMeta] = []
    for row in raw:
        try:
            out.append(TeamMeta(
                name=row.get('name') or row.get('slug') or 'MetaVGC',
                source_url=row.get('source_url') or '',
                mons=[_mon_from_dict(m) for m in row.get('mons') or []],
            ))
        except Exception as exc:
            print(f'[team_factory] skip scraped team: {exc!r}')
    return out


def active_teams() -> List[TeamMeta]:
    scraped = load_scraped_teams()
    pool = scraped if len(scraped) >= 2 else list(METAVGC_TEAMS)
    ok: List[TeamMeta] = []
    for t in pool:
        if not team_passes_item_clause(t):
            dups = []
            counts: Dict[str, int] = {}
            for mon in t.mons:
                iid = _item_id(mon.item)
                if not iid:
                    continue
                counts[iid] = counts.get(iid, 0) + 1
            dups = [k for k, n in counts.items() if n > 1]
            print(
                f'[team_factory] skip Item Clause: {t.name} '
                f'(dup items: {", ".join(dups) or "?"})'
            )
            continue
        ok.append(t)

    # Validación completa Reg M-C (learnsets, clauses, especies) vía Showdown.
    # Opt-out: REGMC_SKIP_PS_VALIDATE=1
    if os.environ.get('REGMC_SKIP_PS_VALIDATE', '').strip().lower() not in (
        '1', 'true', 'yes',
    ):
        try:
            from showdown.inference_sim.validate_regmc_teams import validate_team_metas
            ps_ok, ps_bad = validate_team_metas(ok)
            for t, problems in ps_bad:
                if t.name in _REGMC_REPORTED:
                    continue  # una vez por equipo y proceso (se llama en cada reset)
                _REGMC_REPORTED.add(t.name)
                tip = problems[0] if problems else 'invalid'
                print(f'[team_factory] skip Reg M-C invalid: {t.name} ({tip})')
            ok = ps_ok
        except Exception as exc:
            # Una vez por proceso: con N entornos se llamaria en cada reset.
            global _REGMC_WARNED
            if not _REGMC_WARNED:
                _REGMC_WARNED = True
                print(f'[team_factory] WARN Reg M-C validate skipped: {exc!r}')

    if len(ok) < 2:
        raise RuntimeError(
            'Tras filtrar Item Clause / Reg M-C quedan <2 equipos MetaVGC válidos'
        )
    return ok


def team_by_index(i: int) -> TeamMeta:
    teams = active_teams()
    return teams[i % len(teams)]


def battle_pairings(n_battles: int) -> List[tuple]:
    """Empareja equipos distintos para n batallas BO1.

    Con 2n equipos, usa pares disjuntos 0-1, 2-3, ... para maximizar cobertura.
    Si hay menos equipos, rota con offset.
    """
    teams = active_teams()
    n = len(teams)
    if n < 2:
        raise RuntimeError('Se necesitan al menos 2 equipos MetaVGC')
    pairs = []
    if n >= 2 * n_battles:
        for i in range(n_battles):
            pairs.append((teams[2 * i], teams[2 * i + 1]))
        return pairs
    for i in range(n_battles):
        a = teams[i % n]
        b = teams[(i + 1 + (i // max(n - 1, 1))) % n]
        if a.name == b.name:
            b = teams[(i + 1) % n]
        pairs.append((a, b))
    return pairs


def random_battle_pairings(n_battles: int, *, seed: Optional[int] = None) -> List[tuple]:
    """Pares aleatorios (equipos distintos) del pool activo."""
    import random as _rnd

    teams = active_teams()
    n = len(teams)
    if n < 2:
        raise RuntimeError('Se necesitan al menos 2 equipos MetaVGC')
    rng = _rnd.Random(seed)
    pairs: List[tuple] = []
    for _ in range(n_battles):
        a, b = rng.sample(teams, 2)
        pairs.append((a, b))
    return pairs


def disjoint_ab_pairings(n_teams: int, *, seed: Optional[int] = None) -> List[tuple]:
    """
    Toma N equipos (N par), los baraja y parte en pools A/B sin solape.
    Devuelve N/2 pares (A[i], B[i]) para BO3 heurístico vs random.
    """
    import random as _rnd

    if n_teams < 2 or n_teams % 2 != 0:
        raise ValueError(f'n_teams debe ser par y >= 2 (recibido {n_teams})')
    teams = list(active_teams())
    if len(teams) < n_teams:
        raise RuntimeError(
            f'Se necesitan {n_teams} equipos MetaVGC; hay {len(teams)}. '
            'Amplía metavgc_regmc_teams.json / export.'
        )
    rng = _rnd.Random(seed)
    picked = rng.sample(teams, n_teams)
    rng.shuffle(picked)
    half = n_teams // 2
    pool_a = picked[:half]
    pool_b = picked[half:]
    return list(zip(pool_a, pool_b))


def parse_sp_line(text: str) -> Dict[str, int]:
    mapping = {
        'hp': 'hp', 'atk': 'atk', 'def': 'def', 'spa': 'spa', 'spd': 'spd', 'spe': 'spe',
    }
    out = {k: 0 for k in STAT_KEYS}
    for num, name in re.findall(r'(\d+)\s*(HP|Atk|Def|SpA|SpD|Spe)', text, flags=re.I):
        out[mapping[name.lower()]] = int(num)
    return out
