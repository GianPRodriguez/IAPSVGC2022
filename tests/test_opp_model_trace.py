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
