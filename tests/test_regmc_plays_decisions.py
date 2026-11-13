"""Jugadas de metajuego de Reg M-C: ¿las elige cada agente en escenarios claros?

Cada escenario usa un daño simulado (sin calculadora real) y se evalúa con greedy_joint
y con expectimax. Los escenarios documentan la teoría del juego; si un agente no la
cumple, el test falla y eso es un hallazgo (no se debilita el test). Lo que el código no
implementa todavía va marcado ``xfail`` con el motivo.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from showdown.battle_bots.expectimax_bayesian.expectimax import (
    ExpectimaxConfig,
    best_expectimax_bayesian_action,
)
from showdown.battle_bots.greedy_joint.joint import best_joint_double_action
from showdown.battle_bots.greedy_solo.greedy import CalcPoke, MoveOption
from showdown.damage_calc_bridge import set_damage_range_override

AGENTS = ('joint', 'expectimax')


def mon(species: str, spe: int, hp: int = 160, types=('Normal',), **kw) -> CalcPoke:
    return CalcPoke(
        species=species, nature='Serious',
        stats={'hp': hp, 'atk': 120, 'def': 100, 'spa': 120, 'spd': 100, 'spe': spe},
        cur_hp=kw.pop('cur_hp', hp), types=tuple(types), **kw,
    )


STATUS_MOVES = {'tailwind', 'spore', 'protect', 'helping hand', 'follow me', 'rage powder', 'wide guard',
                'quick guard', 'trick room', 'reflect', 'light screen', 'sleep powder', 'taunt', 'encore'}


def moves(*names: str) -> List[MoveOption]:
    return [MoveOption(i + 1, n, 'normal', 'Status' if n.lower() in STATUS_MOVES else 'Physical')
            for i, n in enumerate(names)]


def make_damage(table: Dict[Tuple[str, str, str], Tuple[int, int]],
                default: Tuple[int, int] = (40, 48)):
    """Daño por (atacante, movimiento, defensor) en minúsculas; el resto usa ``default``."""
    def fake_batch(reqs):
        out = []
        for r in reqs:
            key = (str(r['attacker'].get('species', '')).lower(), str(r['move']).lower(),
                   str(r['defender'].get('species', '')).lower())
            out.append(table.get(key, default))
        return out
    return fake_batch


def decide(agent: str, left: CalcPoke, right: CalcPoke, lmoves: Sequence[MoveOption],
           rmoves: Sequence[MoveOption], foes: Sequence[Tuple[int, CalcPoke]], damage,
           foe_moves: Optional[Dict[int, Tuple[str, ...]]] = None):
    """Devuelve ((movimiento izq, objetivo), (movimiento der, objetivo))."""
    set_damage_range_override(damage)
    try:
        if agent == 'joint':
            bl, br, _ = asyncio.run(best_joint_double_action(left, right, lmoves, rmoves, foes))
        else:
            cfg = ExpectimaxConfig(deadline_ms=8000.0, phase1_budget_ms=3000.0, phase2_end_ms=7800.0,
                                   k_profiles=2, max_worlds=2, max_our_pairs=40)
            bl, br, _, _ = asyncio.run(best_expectimax_bayesian_action(
                left, right, lmoves, rmoves, foes, config=cfg,
                foe_known_moves=foe_moves or {1: ('Tackle',), 2: ('Tackle',)}, reserves=[]))
    finally:
        set_damage_range_override(None)

    def name(action, mvs):
        if action is None:
            return None
        return next((m.name for m in mvs if m.slot == action.move_slot), None), action.target
    return name(bl, lmoves), name(br, rmoves)


# --- Fake Out ----------------------------------------------------------------
@pytest.mark.parametrize('agent', AGENTS)
def test_fake_out_on_first_turn_against_the_biggest_threat(agent):
    left = mon('Incineroar', 80, types=('Fire', 'Dark'), just_switched=True)
    right = mon('Rillaboom', 85, types=('Grass',))
    foes = [(1, mon('Garchomp', 140, types=('Dragon', 'Ground'))),
            (2, mon('Pelipper', 60, types=('Water', 'Flying')))]
    dmg = make_damage({
        ('garchomp', 'earthquake', 'rillaboom'): (150, 170),  # amenaza que mataría al compañero
        ('garchomp', 'earthquake', 'incineroar'): (60, 70),
    })
    (lm, lt), (rm, rt) = decide(agent, left, right, moves('Fake Out', 'Flare Blitz'),
                                moves('Wood Hammer', 'Protect'), foes, dmg,
                                {1: ('Earthquake', 'Dragon Claw'), 2: ('Hurricane',)})
    assert lm == 'Fake Out' and lt == 1, (agent, lm, lt, rm, rt)


@pytest.mark.parametrize('agent', AGENTS)
def test_fake_out_not_chosen_after_the_first_turn(agent):
    left = mon('Incineroar', 80, types=('Fire', 'Dark'), just_switched=False)
    right = mon('Rillaboom', 85, types=('Grass',))
    foes = [(1, mon('Garchomp', 140)), (2, mon('Pelipper', 60))]
    (lm, _), (rm, _) = decide(agent, left, right, moves('Fake Out', 'Flare Blitz'),
                              moves('Wood Hammer', 'Protect'), foes, make_damage({}))
    assert lm != 'Fake Out', (agent, lm, rm)


# --- Sueño -------------------------------------------------------------------
@pytest.mark.parametrize('agent', AGENTS)
def test_sleep_move_on_a_vulnerable_threat_when_no_ko_is_available(agent):
    left = mon('Amoonguss', 50, types=('Grass', 'Poison'))
    right = mon('Rillaboom', 85, types=('Grass',))
    foes = [(1, mon('Garchomp', 140, types=('Dragon', 'Ground'))),
            (2, mon('Pelipper', 60, types=('Water', 'Flying')))]
    dmg = make_damage({}, default=(30, 36))  # ningún KO a la vista
    (lm, lt), _ = decide(agent, left, right, moves('Spore', 'Pollen Puff'),
                         moves('Wood Hammer', 'Protect'), foes, dmg,
                         {1: ('Earthquake',), 2: ('Hurricane',)})
    assert lm == 'Spore' and lt in (1, 2), (agent, lm, lt)


@pytest.mark.parametrize('agent', AGENTS)
def test_sleep_move_not_used_when_every_target_is_immune(agent):
    left = mon('Amoonguss', 50, types=('Grass', 'Poison'))
    right = mon('Rillaboom', 85, types=('Grass',))
    foes = [(1, mon('Ferrothorn', 40, types=('Grass', 'Steel'))),
            (2, mon('Snorlax', 30, types=('Normal',), status='slp'))]
    (lm, _), (rm, _) = decide(agent, left, right, moves('Spore', 'Pollen Puff'),
                              moves('Wood Hammer', 'Protect'), foes, make_damage({}, default=(30, 36)))
    assert lm != 'Spore', (agent, lm, rm)


# --- Tailwind / Trick Room ---------------------------------------------------
@pytest.mark.parametrize('agent', AGENTS)
def test_tailwind_when_the_team_is_outsped_by_both_foes(agent):
    left = mon('Whimsicott', 90, types=('Grass', 'Fairy'))
    right = mon('Rillaboom', 85, types=('Grass',))
    foes = [(1, mon('Garchomp', 140)), (2, mon('Dragapult', 160))]
    dmg = make_damage({}, default=(55, 65))
    (lm, _), (rm, _) = decide(agent, left, right, moves('Tailwind', 'Moonblast'),
                              moves('Wood Hammer', 'Protect'), foes, dmg,
                              {1: ('Earthquake',), 2: ('Shadow Ball',)})
    assert 'Tailwind' in (lm, rm), (agent, lm, rm)


@pytest.mark.parametrize('agent', AGENTS)
def test_no_tailwind_when_the_team_already_outspeeds_everything(agent):
    left = mon('Whimsicott', 200, types=('Grass', 'Fairy'))
    right = mon('Dragapult', 210, types=('Dragon', 'Ghost'))
    foes = [(1, mon('Ferrothorn', 20)), (2, mon('Snorlax', 30))]
    dmg = make_damage({}, default=(70, 80))
    (lm, _), (rm, _) = decide(agent, left, right, moves('Tailwind', 'Moonblast'),
                              moves('Shadow Ball', 'Protect'), foes, dmg)
    assert 'Tailwind' not in (lm, rm), (agent, lm, rm)


# --- Helping Hand ------------------------------------------------------------
@pytest.mark.parametrize('agent', AGENTS)
def test_helping_hand_when_it_turns_the_partner_hit_into_a_ko(agent):
    left = mon('Indeedee-F', 70, types=('Psychic', 'Normal'))
    right = mon('Rillaboom', 85, types=('Grass',))
    foes = [(1, mon('Garchomp', 140, hp=160, cur_hp=160)), (2, mon('Pelipper', 60))]
    # 115-125 HP no mata (160), con Helping Hand ×1.5 (≥172) sí
    dmg = make_damage({('rillaboom', 'wood hammer', 'garchomp'): (115, 125)}, default=(20, 24))
    (lm, _), (rm, rt) = decide(agent, left, right, moves('Helping Hand', 'Psychic'),
                               moves('Wood Hammer', 'Protect'), foes, dmg)
    assert lm == 'Helping Hand' and rm == 'Wood Hammer' and rt == 1, (agent, lm, rm, rt)


# --- Redirección --------------------------------------------------------------
@pytest.mark.parametrize('agent', AGENTS)
def test_follow_me_to_cover_a_partner_that_secures_a_ko(agent):
    left = mon('Clefairy', 40, types=('Fairy',))
    right = mon('Rillaboom', 85, types=('Grass',))
    foes = [(1, mon('Pelipper', 60, cur_hp=40)), (2, mon('Garchomp', 140))]
    dmg = make_damage({('rillaboom', 'wood hammer', 'pelipper'): (90, 100),
                       ('garchomp', 'earthquake', 'rillaboom'): (120, 140)})
    (lm, _), (rm, rt) = decide(agent, left, right, moves('Follow Me', 'Moonblast'),
                               moves('Wood Hammer', 'Protect'), foes, dmg,
                               {1: ('Hurricane',), 2: ('Earthquake',)})
    assert lm == 'Follow Me' and rm == 'Wood Hammer', (agent, lm, rm)


# --- Lo que el código aún no implementa (documentado) -------------------------
@pytest.mark.xfail(reason='Wide Guard / Quick Guard no se usan: el bot no tiene lógica propia '
                          'para bloquear movimientos de área o de prioridad', strict=False)
@pytest.mark.parametrize('agent', AGENTS)
def test_wide_guard_against_two_spread_attackers(agent):
    left = mon('Whimsicott', 90, types=('Grass', 'Fairy'))
    right = mon('Rillaboom', 85, types=('Grass',))
    foes = [(1, mon('Garchomp', 140)), (2, mon('Dragapult', 160))]
    dmg = make_damage({
        ('garchomp', 'earthquake', 'whimsicott'): (170, 190), ('garchomp', 'earthquake', 'rillaboom'): (170, 190),
        ('dragapult', 'hyper voice', 'whimsicott'): (170, 190), ('dragapult', 'hyper voice', 'rillaboom'): (170, 190),
        ('whimsicott', 'moonblast', 'garchomp'): (70, 80), ('whimsicott', 'moonblast', 'dragapult'): (70, 80),
        ('rillaboom', 'wood hammer', 'garchomp'): (70, 80), ('rillaboom', 'wood hammer', 'dragapult'): (70, 80),
    }, default=(10, 12))
    (lm, _), (rm, _) = decide(agent, left, right, moves('Wide Guard', 'Moonblast'),
                              moves('Wood Hammer', 'Protect'), foes, dmg,
                              {1: ('Earthquake',), 2: ('Hyper Voice',)})
    assert 'Wide Guard' in (lm, rm), (agent, lm, rm)
