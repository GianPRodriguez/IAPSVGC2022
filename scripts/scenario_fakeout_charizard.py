"""Escenario puntual: Incineroar Fake Out vs Charizard mega + spread/FF.

Compara greedy_joint vs expectimax_bayesian en un turno sintético.
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Optional, Tuple

from showdown.battle_bots.expectimax_bayesian.expectimax import (
    ExpectimaxConfig,
    best_expectimax_bayesian_action,
)
from showdown.battle_bots.greedy_joint.joint import best_joint_double_action
from showdown.battle_bots.greedy_solo.greedy import CalcPoke, MoveOption, ScoredAction
from showdown.damage_calc_bridge import set_damage_range_override


def _move_label(a: Optional[ScoredAction]) -> str:
    if a is None:
        return '—'
    slot = int(getattr(a, 'move_slot', 0) or 0)
    tgt = int(getattr(a, 'target', 0) or 0)
    mega = int(getattr(a, 'mega', 0) or 0)
    kind = getattr(a, 'kind', 'move')
    return f'{kind} slot={slot}->foe{tgt} mega={mega}'


def _build_scenario(*, use_override: bool) -> dict:
    """Incineroar + Rillaboom vs Charizard + Venusaur; Incin acaba de entrar."""
    incin = CalcPoke(
        species='Incineroar',
        ability='Intimidate',
        nature='Careful',
        item='Sitrus Berry',
        just_switched=True,
        stats={'hp': 202, 'atk': 130, 'def': 111, 'spa': 90, 'spd': 110, 'spe': 87},
        cur_hp=202,
        types=('Fire', 'Dark'),
    )
    rilla = CalcPoke(
        species='Rillaboom',
        ability='Grassy Surge',
        nature='Adamant',
        item='Assault Vest',
        stats={'hp': 177, 'atk': 181, 'def': 100, 'spa': 60, 'spd': 90, 'spe': 85},
        cur_hp=177,
        types=('Grass', 'Dark'),
    )
    charizard = CalcPoke(
        species='Charizard',
        ability='Blaze',
        nature='Timid',
        item='Charizardite Y',
        just_switched=True,
        stats={'hp': 153, 'atk': 76, 'def': 94, 'spa': 159, 'spd': 102, 'spe': 157},
        cur_hp=153,
        types=('Fire', 'Flying'),
    )
    sylveon = CalcPoke(
        species='Sylveon',
        ability='Pixilate',
        nature='Modest',
        item='Life Orb',
        stats={'hp': 175, 'atk': 60, 'def': 85, 'spa': 150, 'spd': 130, 'spe': 60},
        cur_hp=175,
        types=('Fairy',),
    )
    foes = [(1, charizard), (2, sylveon)]

    incin_moves = [
        MoveOption(1, 'Fake Out', 'normal', 'Physical'),
        MoveOption(2, 'Flare Blitz', 'normal', 'Physical'),
        MoveOption(3, 'Heat Wave', 'allAdjacent', 'Special'),
        MoveOption(4, 'Knock Off', 'normal', 'Physical'),
    ]
    # Set clásico Rillaboom (sin Heat Wave en Incin; spread solo si lo lleva el partner).
    rilla_moves = [
        MoveOption(1, 'Fake Out', 'normal', 'Physical'),
        MoveOption(2, 'Wood Hammer', 'normal', 'Physical'),
        MoveOption(3, 'Grassy Glide', 'normal', 'Physical'),
        MoveOption(4, 'U-turn', 'normal', 'Physical'),
    ]

    if use_override:
        # Daños didácticos: Flare Blitz casi KO Charizard; Heat Wave spread + FF fuerte al Grass.
        def fake_batch(reqs):
            out = []
            for req in reqs:
                atk = str((req.get('attacker') or {}).get('species', '')).lower()
                move = str(req.get('move') or '').lower().replace(' ', '').replace('-', '')
                defn = str((req.get('defender') or {}).get('species', '')).lower()
                is_ally = bool(req.get('isAlly') or req.get('is_ally'))
                if move == 'fakeout':
                    out.append((30, 40))
                elif move == 'heatwave':
                    if is_ally or 'rillaboom' in defn:
                        out.append((85, 100))  # KO al compañero Grass
                    elif 'venusaur' in defn:
                        out.append((70, 90))
                    elif 'charizard' in defn:
                        out.append((35, 50))
                    else:
                        out.append((40, 55))
                elif move == 'flareblitz' and 'charizard' in defn:
                    out.append((88, 100))
                elif move == 'knockoff' and 'charizard' in defn:
                    out.append((45, 55))
                elif move == 'woodhammer' and 'charizard' in defn:
                    out.append((55, 70))
                elif move == 'woodhammer' and 'sylveon' in defn:
                    out.append((55, 70))
                elif move == 'grassyglide' and 'sylveon' in defn:
                    out.append((45, 60))
                elif move == 'uturn' and 'charizard' in defn:
                    out.append((35, 45))
                else:
                    out.append((25, 35))
            return out

        set_damage_range_override(fake_batch)
    else:
        set_damage_range_override(None)

    return {
        'incin': incin,
        'rilla': rilla,
        'foes': foes,
        'incin_moves': incin_moves,
        'rilla_moves': rilla_moves,
        'use_override': use_override,
    }


async def _run_agent(name: str, scenario: dict):
    incin = scenario['incin']
    rilla = scenario['rilla']
    foes = scenario['foes']
    incin_moves = scenario['incin_moves']
    rilla_moves = scenario['rilla_moves']

    if name == 'greedy_joint':
        best_l, best_r, score = await best_joint_double_action(
            incin, rilla, incin_moves, rilla_moves, foes,
        )
        meta = type('Meta', (), {'phase': 'joint', 'diverged': False})()
        return best_l, best_r, score, meta

    cfg = ExpectimaxConfig()  # cobertura 75% por defecto (sin inyección FO)
    return await best_expectimax_bayesian_action(
        incin, rilla, incin_moves, rilla_moves, foes,
        config=cfg,
        can_mega_left=False,
        can_mega_right=False,
        foe_known_moves={
            1: ('Heat Wave', 'Air Slash', 'Solar Beam', 'Protect'),
            2: ('Hyper Voice', 'Dazzling Gleam', 'Quick Attack', 'Protect'),
        },
    )


def _slot_move(moves, scored) -> str:
    if scored is None:
        return '?'
    slot = int(getattr(scored, 'move_slot', 0) or 0)
    for m in moves:
        if int(m.slot) == slot:
            return m.name
    return f'slot{slot}'


async def main() -> int:
    if '--real-only' in sys.argv:
        modes = ('real_calc',)
    elif '--override-only' in sys.argv:
        modes = ('override',)
    else:
        modes = ('real_calc', 'override')
    results = {}

    for mode in modes:
        sc = _build_scenario(use_override=(mode == 'override'))
        try:
            row = {}
            for agent in ('greedy_joint', 'expectimax_bayesian'):
                best_l, best_r, score, meta = await _run_agent(agent, sc)
                incin_move = _slot_move(sc['incin_moves'], best_l)
                rilla_move = _slot_move(sc['rilla_moves'], best_r)
                row[agent] = {
                    'incineroar': incin_move,
                    'incin_target': int(getattr(best_l, 'target', 0) or 0),
                    'rillaboom': rilla_move,
                    'score': float(score),
                    'left': _move_label(best_l),
                    'right': _move_label(best_r),
                    'phase': getattr(meta, 'phase', 'joint'),
                    'diverged': bool(getattr(meta, 'diverged', False)),
                }
            results[mode] = row
        finally:
            set_damage_range_override(None)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
