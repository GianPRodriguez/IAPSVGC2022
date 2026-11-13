"""Spe abierta en belief → PE de carrera; Atk abierta → max amenaza."""
from __future__ import annotations

from showdown.battle_bots.greedy_solo.greedy import (
    CalcPoke,
    _SPE_OPEN_POINT_ESTIMATE,
    apply_belief_stats,
)
from showdown.stat_inference import StatBelief, champions_stat


def test_open_spe_uses_point_estimate_not_max():
    assert _SPE_OPEN_POINT_ESTIMATE is True
    # >8 candidatos → is_narrow False
    belief = StatBelief(
        'pelipper',
        {'hp': 60, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 65},
        'modest',
    )
    belief.candidates['spe'] = set(range(0, 33))
    belief.candidates['atk'] = set(range(0, 33))
    poke = CalcPoke(species='Pelipper', stats={
        'hp': 100, 'atk': 50, 'def': 100, 'spa': 95, 'spd': 70, 'spe': 65,
    })
    apply_belief_stats(poke, belief)
    pref = belief.preferred_sp('spe')
    assert pref is not None
    expected_spe = champions_stat(65, int(pref), 'modest', 'spe')
    max_spe = champions_stat(65, 32, 'modest', 'spe')
    assert poke.stats is not None
    assert int(poke.stats['spe']) == int(expected_spe)
    assert int(poke.stats['spe']) < int(max_spe)
    # Atk abierta sigue en max SP
    expected_atk = champions_stat(50, 32, 'modest', 'atk')
    assert int(poke.stats['atk']) == int(expected_atk)
