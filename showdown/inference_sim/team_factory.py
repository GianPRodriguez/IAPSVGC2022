"""Equipos Reg M-C basados en MetaVGC + empaquetado Showdown (EVs = Stat Points)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from showdown.stat_inference import STAT_KEYS, champions_stat
from teams.team_converter import normalize_name

# Fuente: https://metavgc.com/teams/featured/regulation-mc


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
    lines = [f"{mon.species} @ {mon.item}"]
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
        lines.append(f"- {move}")
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


def team_by_index(i: int) -> TeamMeta:
    return METAVGC_TEAMS[i % len(METAVGC_TEAMS)]


def parse_sp_line(text: str) -> Dict[str, int]:
    mapping = {
        'hp': 'hp', 'atk': 'atk', 'def': 'def', 'spa': 'spa', 'spd': 'spd', 'spe': 'spe',
    }
    out = {k: 0 for k in STAT_KEYS}
    for num, name in re.findall(r'(\d+)\s*(HP|Atk|Def|SpA|SpD|Spe)', text, flags=re.I):
        out[mapping[name.lower()]] = int(num)
    return out
