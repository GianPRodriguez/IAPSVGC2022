from showdown.inference_sim import analyze_opp_model as ana
from showdown.inference_sim import opp_model_trace as t

LINES = [
    '|turn|3',
    '|move|p1a: Mine|Flamethrower|p2a: Foe',
    '|move|p2a: Foe|Protect|p2a: Foe',
    '|move|p2b: Other|Moonblast|p1b: Mine2',
    '|move|p2b: Other|Sleep Talk|p2b: Other|[from]move: Sleep Talk',
    '|turn|4',
    '|move|p2a: Foe|Flamethrower|p1a: Mine',
]


def test_parse_foe_moves_only_the_given_turn_and_skips_called_moves():
    ev = t.parse_foe_moves(LINES, 3, 'p2', 'p1')
    assert ev == [('Foe', 'Protect', 0), ('Other', 'Moonblast', 2)]
    assert t.parse_foe_moves(LINES, 9, 'p2', 'p1') == []


def _pred(moves, merged):
    return {'moves': moves, 'merged': merged}


def test_score_event_probabilities_and_hits():
    p = _pred(['Protect', 'Flamethrower'],
              {('flamethrower', 1, 'greedy'): 0.6, ('protect', 0, 'control'): 0.3,
               ('__switch__', 0, 'switch'): 0.1})
    s = t.score_event([p], 'Flamethrower', 1)
    assert s['status'] == 'ok' and s['top1_move_hit'] and s['top1_exact_hit']
    assert abs(s['p_move'] - 0.6) < 1e-9 and abs(s['mass_control'] - 0.3) < 1e-9
    c = t.score_event([p], 'Protect', 0)
    assert not c['top1_move_hit'] and abs(c['p_exact'] - 0.3) < 1e-9


def test_score_event_handles_missing_or_ambiguous_prediction():
    p = _pred(['Protect'], {('protect', 0, 'control'): 1.0})
    assert t.score_event([p], 'Surf', 1)['status'] == 'sin_prediccion'
    assert t.score_event([p, p], 'Protect', 0)['status'] == 'ambiguo'


def test_summarize_splits_attacks_and_control():
    rows = [
        {'status': 'ok', 'control': False, 'p_move': 0.6, 'p_exact': 0.6, 'uniform_p': 0.5,
         'top1_move_hit': True, 'top1_exact_hit': True, 'mass_control': 0.2, 'mass_switch': 0.1,
         'species': 'a'},
        {'status': 'ok', 'control': True, 'p_move': 0.2, 'p_exact': 0.2, 'uniform_p': 0.5,
         'top1_move_hit': False, 'top1_exact_hit': False, 'mass_control': 0.2, 'mass_switch': 0.1,
         'species': 'a'},
        {'status': 'sin_prediccion'},
    ]
    s = ana.summarize(rows)
    assert s['con_prediccion'] == 2 and s['ataques'] == 1 and s['control'] == 1
    assert abs(s['fraccion_real_de_movimientos_de_control'] - 0.5) < 1e-9
    assert abs(s['acierto_top1_en_ataques'] - 1.0) < 1e-9


def test_expectimax_env_overrides(monkeypatch):
    import pytest
    from showdown.battle_bots.expectimax_bayesian.expectimax import load_expectimax_config
    monkeypatch.setenv('EXPECTIMAX_OVERRIDES', 'opp_policy_control_weight=0.1;opp_policy_greedy_weight=0.9')
    cfg = load_expectimax_config()
    assert abs(cfg.opp_policy_control_weight - 0.1) < 1e-9 and abs(cfg.opp_policy_greedy_weight - 0.9) < 1e-9
    monkeypatch.setenv('EXPECTIMAX_OVERRIDES', 'no_existe=1')
    with pytest.raises(ValueError):
        load_expectimax_config()


def _dist(opp_moves, hit_map, **kw):
    from showdown.battle_bots.expectimax_bayesian.expectimax import _rebuild_opp_dists_from_hits
    out = _rebuild_opp_dists_from_hits(
        opp_moves, hit_map, left_atk=object(), right_atk=None, w_greedy=1.0, w_control=0.0,
        use_real_damage=True, w_switch=0.0, split_targets=False, **kw)
    return {a.move_name.lower().replace(' ', ''): p for a, p in out['OL']}


def test_opp_dist_slot_collision_between_foes_is_avoided_with_unique_slots():
    from showdown.battle_bots.expectimax_bayesian.opp_policy import move_options_from_names
    hit = {('F1', 'flamethrower', 'OL'): (40, 50), ('F2', 'surf', 'OL'): (10, 20)}
    # Mismo slot 1 para los dos rivales: se pisan el daño y salen igual de probables.
    collided = move_options_from_names(['Flamethrower']) + move_options_from_names(['Surf'])
    d0 = _dist(collided, hit)
    assert abs(d0['flamethrower'] - d0['surf']) < 1e-9
    unique = move_options_from_names(['Flamethrower'], start_slot=1) + move_options_from_names(['Surf'], start_slot=5)
    d1 = _dist(unique, hit)
    assert d1['flamethrower'] > 2.5 * d1['surf']


def test_opp_dist_fakeout_first_turn_gets_target_mass():
    from showdown.battle_bots.expectimax_bayesian.opp_policy import move_options_from_names
    hit = {('F1', 'fakeout', 'OL'): (10, 12), ('F1', 'closecombat', 'OL'): (80, 90)}
    moves = move_options_from_names(['Fake Out', 'Close Combat'])
    base = _dist(moves, hit)
    boosted = _dist(moves, hit, fakeout_first_p=0.7)
    assert base['fakeout'] < 0.2
    assert abs(boosted['fakeout'] - 0.7) < 0.02
