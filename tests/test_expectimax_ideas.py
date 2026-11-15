"""Piezas nuevas para las pruebas de mejora de expectimax (ideas 1, 2 y 4)."""
from showdown.battle_bots.expectimax_bayesian.expectimax import ExpectimaxConfig, _tempo_credit
from showdown.battle_bots.greedy_joint.joint import JointAtom
from showdown.battle_bots.greedy_solo.greedy import CalcPoke


def mon(species, spe, **kw):
    return CalcPoke(species=species, stats={'hp': 200, 'atk': 100, 'def': 100, 'spa': 100,
                                            'spd': 100, 'spe': spe}, cur_hp=200, **kw)


def atom(tid, name='Wood Hammer', utility='damage', pct=60.0):
    return JointAtom(move_slot=1, target=tid, utility=utility, move_name=name,
                     damage_pct={tid: pct}, max_roll={tid: 100}, ko_tids=set())


def credit(ours_spe, foe_spe, pk=1.0, threat=500.0, **kw):
    left_atk, foe = mon('us', ours_spe), mon('foe', foe_spe)
    return _tempo_credit(
        atom(1), None, left_atk, None, [(1, foe)], {1: pk}, {1: threat},
        trick_room=kw.get('trick_room', False), our_tailwind=kw.get('tw', False),
        opp_tailwind=False, terrain=None)


def test_tempo_credit_needs_ko_probability_and_acting_first():
    assert credit(150, 100) == 500.0                 # le matamos antes de que se mueva
    assert credit(150, 100, pk=0.4) == 200.0          # proporcional a P(KO)
    assert credit(80, 100) == 0.0                     # él actúa antes: su ataque ocurre
    assert credit(150, 100, threat=0.0) == 0.0        # sin amenaza no hay nada que anular


def test_tempo_credit_respects_trick_room_and_tailwind():
    assert credit(150, 100, trick_room=True) == 0.0   # Trick Room: el lento va primero
    assert credit(80, 100, tw=True) == 500.0          # Tailwind propio nos adelanta


def test_new_config_flags_default_to_off():
    cfg = ExpectimaxConfig()
    assert cfg.v_iso_syn_weight == 0.0 and cfg.v_iso_threat_weight == 0.0


def test_stable_c235_defaults():
    # Versión estable confirmada en el servidor (semillas 8001-8017, +2.3 pts vs base).
    cfg = ExpectimaxConfig()
    assert cfg.v_tempo_weight == 1.0
    assert cfg.phase2_gate_enabled and cfg.phase2_gate_mode == 'beat'
    assert cfg.phase2_gate_margin == 40.0
    assert cfg.soft_ko_bonus == 250.0
    assert cfg.k_profiles == 4 and cfg.max_worlds == 12


def test_c235_survives_load_without_overrides(monkeypatch):
    from showdown.battle_bots.expectimax_bayesian.expectimax import load_expectimax_config
    monkeypatch.delenv('EXPECTIMAX_OVERRIDES', raising=False)
    cfg = load_expectimax_config()
    assert (cfg.v_tempo_weight, cfg.phase2_gate_mode, cfg.phase2_gate_margin,
            cfg.soft_ko_bonus, cfg.k_profiles, cfg.max_worlds) == (1.0, 'beat', 40.0, 250.0, 4, 12)


def test_per_player_sp_oracle_only_affects_that_beliefs_object():
    from showdown.stat_inference import TeamStatBeliefs
    a, b = TeamStatBeliefs(), TeamStatBeliefs()
    a.oracle_lookup = lambda sp: {'spe': 32, 'hp': 0, 'atk': 0, 'def': 0, 'spa': 0, 'spd': 0}
    from showdown.stat_inference import _apply_oracle, StatBelief
    ba = StatBelief('garchomp', {'spe': 102}, 'jolly')
    bb = StatBelief('garchomp', {'spe': 102}, 'jolly')
    _apply_oracle(ba, a.oracle_lookup)
    _apply_oracle(bb, b.oracle_lookup)
    assert ba.candidates['spe'] == {32}
    assert len(bb.candidates['spe']) > 1
