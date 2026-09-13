"""Prior suave MetaVGC para point-estimate de Stat Points."""
from __future__ import annotations

from showdown import meta_sp_prior as msp
from showdown.stat_inference import StatBelief, champions_stat, reset_sp_limits


def test_meta_prior_loads_compendium():
    n = msp.load_meta_sp_prior(force=True)
    assert n >= 100, f'expected >=100 mons indexed, got {n}'


def test_preferred_sp_stays_inside_candidates():
    reset_sp_limits()
    msp.load_meta_sp_prior(force=True)
    belief = StatBelief(
        'salamence',
        {'hp': 95, 'atk': 135, 'def': 80, 'spa': 110, 'spd': 80, 'spe': 100},
        'jolly',
    )
    # Conjunto amplio: el prior solo elige dentro del set.
    for stat in ('hp', 'atk', 'def', 'spa', 'spd', 'spe'):
        sp = belief.preferred_sp(stat)
        assert sp is not None
        assert sp in belief.candidates[stat]


def test_preferred_sp_respects_narrow_set():
    reset_sp_limits()
    msp.load_meta_sp_prior(force=True)
    belief = StatBelief(
        'salamence',
        {'hp': 95, 'atk': 135, 'def': 80, 'spa': 110, 'spd': 80, 'spe': 100},
        'jolly',
    )
    belief.candidates['spe'] = {28, 29, 30}
    sp = belief.preferred_sp('spe')
    assert sp in {28, 29, 30}


def test_point_estimate_uses_legal_champions_stat():
    reset_sp_limits()
    msp.load_meta_sp_prior(force=True)
    belief = StatBelief(
        'rillaboom',
        {'hp': 100, 'atk': 125, 'def': 90, 'spa': 60, 'spd': 70, 'spe': 85},
        'adamant',
    )
    est = belief.point_estimate_stats()
    assert 'spe' in est
    assert est['spe'] == champions_stat(
        85, belief.preferred_sp('spe'), 'adamant', 'spe'
    )


def test_prior_never_mutates_candidates():
    reset_sp_limits()
    msp.load_meta_sp_prior(force=True)
    belief = StatBelief(
        'incineroar',
        {'hp': 95, 'atk': 115, 'def': 90, 'spa': 80, 'spd': 90, 'spe': 60},
        'careful',
    )
    before = {k: set(v) for k, v in belief.candidates.items()}
    _ = belief.preferred_sp('spe')
    _ = belief.point_estimate_stats()
    assert belief.candidates == before
