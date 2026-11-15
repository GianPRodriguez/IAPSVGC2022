"""Expectimax Fase 2b: Protect condicional del rival en la ofensa."""
import sys

sys.path.insert(0, '.')


def test_foe_protect_probability_conditional():
    """Fase 2b: P(Protect) solo si lo lleva, no lo usó el turno anterior; más alta bajo amenaza de KO."""
    from showdown.battle_bots.expectimax_bayesian.expectimax import (
        ExpectimaxConfig, foe_protect_probability,
    )
    from showdown.battle_bots.greedy_solo.greedy import CalcPoke

    cfg = ExpectimaxConfig(opp_protect_base=0.1, opp_protect_threat=0.3, opp_protect_ko_threshold=0.5)
    foe = CalcPoke(species='incineroar')
    assert foe_protect_probability(foe, ['Fake Out', 'Protect'], 0.2, cfg) == 0.1
    assert foe_protect_probability(foe, ['Fake Out', 'Protect'], 0.8, cfg) == 0.3
    assert foe_protect_probability(foe, ['Fake Out', 'Knock Off'], 0.8, cfg) == 0.0
    foe2 = CalcPoke(species='incineroar', protect_counter=1)
    assert foe_protect_probability(foe2, ['Protect'], 0.8, cfg) == 0.0
    assert foe_protect_probability(foe, ['Protect'], 0.8, ExpectimaxConfig()) == 0.0  # off por defecto
