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


def test_iso_switch_weight_default_off_and_overridable(monkeypatch):
    """Fase 2c: switch_v entra en V isomorfo solo con v_iso_switch_weight > 0."""
    from showdown.battle_bots.expectimax_bayesian.expectimax import (
        ExpectimaxConfig, load_expectimax_config,
    )

    assert ExpectimaxConfig().v_iso_switch_weight == 0.0
    monkeypatch.setenv('EXPECTIMAX_OVERRIDES',
                       'enable_voluntary_switch=1;max_switch_hybrids=8;v_iso_switch_weight=1.0')
    cfg = load_expectimax_config()
    assert cfg.enable_voluntary_switch is True and cfg.max_switch_hybrids == 8
    assert cfg.v_iso_switch_weight == 1.0


def test_switch_cost_knobs_default_off():
    """Fase 2c: costo de cambio y penalización anti-rebote, apagados por defecto."""
    from showdown.battle_bots.expectimax_bayesian.expectimax import ExpectimaxConfig, ReserveOption
    from showdown.battle_bots.greedy_solo.greedy import CalcPoke

    cfg = ExpectimaxConfig()
    assert cfg.switch_cost == 0.0 and cfg.switch_recent_penalty == 0.0
    r = ReserveOption(slot=3, poke=CalcPoke(species='incineroar'))
    assert r.recently_out is False
    assert ReserveOption(slot=3, poke=CalcPoke(species='incineroar'), recently_out=True).recently_out
