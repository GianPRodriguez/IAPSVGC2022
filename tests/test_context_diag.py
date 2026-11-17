"""Decisiones según el plan del rival (showdown/rl/context_diag.py, memorizacion_equipos.md §12)."""
from types import SimpleNamespace as NS

from showdown.rl import context_diag as CD


def _battle(msgs, role='p1'):
    return NS(_replay_data=[m.split('|') for m in msgs], player_role=role)


def test_count_events_by_side_and_activations():
    b = _battle(['|turn|1',
                 '|move|p2a: Indeedee|Follow Me|p2a: Indeedee',
                 '|move|p2b: Hatterene|Trick Room|p2b: Hatterene',
                 '|-fieldstart|move: Trick Room|[of] p2b: Hatterene',
                 '|move|p1a: Incineroar|Fake Out|p2a: Indeedee',
                 '|move|p1b: Whimsicott|Tailwind|p1b: Whimsicott',
                 '|-sidestart|p1: Jugador|move: Tailwind',
                 '|move|p2a: Indeedee|Fake Out|p1a: Incineroar|[from]move: Instruct'])  # llamado: no cuenta
    ev = CD.count_events(b)
    assert ev['rival'] == {'redirect': 1, 'trick_room': 1, 'trick_room_activado': 1}
    assert ev['nuestro'] == {'fake_out': 1, 'tailwind': 1, 'tailwind_activado': 1}
    # desde el lado p2 se invierten
    ev2 = CD.count_events(_battle([m for m in ['|move|p2b: Hatterene|Trick Room|p2b: Hatterene']], role='p2'))
    assert ev2['nuestro'] == {'trick_room': 1}


def test_trick_room_owner_tracks_start_and_end():
    start = ['|move|p2b: Hatterene|Trick Room|p2b: Hatterene', '|-fieldstart|move: Trick Room|[of] p2b: Hatterene']
    assert CD.trick_room_owner(_battle(start)) == 'rival'
    assert CD.trick_room_owner(_battle(start + ['|-fieldend|move: Trick Room'])) == ''
    assert CD.trick_room_owner(_battle(start, role='p2')) == 'nuestro'


def _mon(moves, mss=1, fainted=False):
    return NS(moves=tuple(moves), moves_since_switch=mss, is_active=True, fainted=fainted)


def _snap(foes, tr=False, opp_tw=False, turn=3):
    return NS(field=NS(trick_room=tr, opp_tailwind=opp_tw), foe=NS(active=foes), turn=turn)


def test_contexts():
    hat = _mon(['Trick Room', 'Dazzling Gleam'])
    ind = _mon(['Follow Me', 'Psychic'])
    inc = _mon(['Fake Out', 'Knock Off'], mss=0)
    s = _snap([hat, ind])
    assert CD.contexts(s, _battle([])) == {'rival_puede_trick_room', 'rival_redireccion'}
    s = _snap([inc, ind], opp_tw=True, turn=1)
    assert CD.contexts(s, _battle([])) == {'rival_fake_out', 'rival_redireccion', 'rival_tailwind', 'turno_1'}
    start = ['|move|p2b: Hatterene|Trick Room|p2b: Hatterene', '|-fieldstart|move: Trick Room|[of] p2b: Hatterene']
    assert 'rival_trick_room' in CD.contexts(_snap([hat, ind], tr=True), _battle(start))
    assert 'rival_puede_trick_room' not in CD.contexts(_snap([hat, ind], tr=True), _battle(start))
    assert CD.contexts(_snap([_mon(['Tackle']), _mon(['Tackle'])]), _battle([])) == {'sin_contexto'}


def test_target_flags_with_random_baseline():
    s = _snap([_mon(['Trick Room']), _mon(['Follow Me'])])
    f0, f1 = CD.target_flags(s, 0), CD.target_flags(s, 1)
    assert f0['al_que_activa_trick_room'] == 1.0 and f1['al_redirector'] == 1.0
    assert f0['esperado_trick_room'] == 0.5 and f1['esperado_redirector'] == 0.5


def test_summary_table():
    counts = {'rival_puede_trick_room': {'decisiones': 10, 'ataque_rival': 6, 'protect': 2, 'ataques_un_rival': 6,
                                         'al_que_activa_trick_room': 4, 'esperado_trick_room': 3}}
    md = '\n'.join(CD.summarize(counts))
    assert '| rival_puede_trick_room | 10 |' in md and '67 % (azar 50 %)' in md


def test_summary19(tmp_path):
    import json
    from showdown.rl import kickoff_summary19 as K
    (tmp_path / 'diag').mkdir()
    rep = {'resultados': {'greedy_joint': {'win': 3, 'loss': 1}},
           'contexto': {'rival_tailwind': {'decisiones': 4, 'protect': 2}},
           'partidas': [{'rival': 'greedy_joint', 'res': 'win',
                         'eventos': {'rival': {'tailwind_activado': 1}, 'nuestro': {}}},
                        {'rival': 'greedy_joint', 'res': 'loss', 'eventos': {'rival': {}, 'nuestro': {}}}]}
    (tmp_path / 'diag' / 'ddqn_s0.json').write_text(json.dumps(rep), encoding='utf-8')
    assert K.main([str(tmp_path)]) == 0
    md = (tmp_path / 'kickoff_summary19.md').read_text(encoding='utf-8')
    assert 'greedy_joint 75 %' in md and '| rival_tailwind | 4 |' in md
    assert '| greedy_joint | rival | 0 % | 0 % | 50 % |' in md
    assert 'Sin datos.' in md  # PPO sin archivos
