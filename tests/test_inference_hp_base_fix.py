"""Regresión: HPB % del rival no debe corromper la base HP de StatBelief."""
import sys
sys.path.insert(0, '.')

from showdown import stat_inference as si
from showdown.battle import Pokemon


def test_opponent_percent_hp_does_not_corrupt_belief_base():
    si.reset_sp_limits()
    beliefs = si.TeamStatBeliefs()
    p = Pokemon('salamence', -1, True, 'salamencite', 'intimidate', None, None, None, None, 'jolly')
    # Tras switch/damage PS, HPB_battle_base queda en 100 (%)
    p.HPB_battle_base = 100
    p.HPA_battle_actual = 80
    belief = beliefs.belief_for(p)
    assert belief is not None
    assert belief.base_stats['hp'] == 95  # base de especie Champions
    low, high = belief.stat_range('hp')
    assert low == si.champions_stat(95, 0, 'jolly', 'hp')
    assert high == si.champions_stat(95, 32, 'jolly', 'hp')
    si.reset_sp_limits()
