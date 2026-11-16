"""Resumen de la octava cola (memorización de equipos rivales)."""
import json
import sys
from types import SimpleNamespace

sys.path.insert(0, '.')

from showdown.rl.ddqn import kickoff_summary8 as K


def _team(key, species):
    from showdown.inference_sim.team_factory import MonSet
    return SimpleNamespace(source_url=key, name=key,
                           mons=[MonSet(species=s, item='', ability='', nature='', moves=[], sp={}) for s in species])


def test_similarity_rows_buckets_by_overlap():
    refs = [_team('r', ['A', 'B', 'C', 'D', 'E', 'F'])]
    ev = {'e5': _team('e5', ['A', 'B', 'C', 'D', 'E', 'Z']), 'e2': _team('e2', ['A', 'B', 'X', 'Y', 'W', 'Z'])}
    games = [{'rival': 'greedy_joint', 'equipo': 'e5', 'res': 'win'},
             {'rival': 'greedy_joint', 'equipo': 'e5', 'res': 'loss'},
             {'rival': 'expectimax_bayesian', 'equipo': 'e2', 'res': 'loss'},
             {'rival': 'random', 'equipo': 'e2', 'res': 'win'}]           # random no cuenta
    rows = K.similarity_rows(games, refs, ev)
    assert rows[5] == [1, 2] and rows[2] == [0, 1]


def test_summary_writes_tables(tmp_path, monkeypatch):
    d = tmp_path / 'diag'
    d.mkdir()
    res = {f: {'win': 60, 'loss': 40} for f in K.FOES}
    res_ev = {f: {'win': 40, 'loss': 60} for f in K.FOES}
    (d / 'F170_s0_own.json').write_text(json.dumps({'resultados': res}), encoding='utf-8')
    (d / 'F170_s0_eval.json').write_text(json.dumps({'resultados': res_ev, 'partidas': [
        {'rival': 'greedy_joint', 'equipo': 'e5', 'res': 'win'}]}), encoding='utf-8')
    run = tmp_path / 'F170_s0'
    run.mkdir()
    rec = {'kind': 'eval', 'step': 250000, 'mean_wr': 0.4,
           'rows': [{'foe': f, 'win_rate': 0.4, 'n_games': 50} for f in K.FOES]}
    (run / 'train_metrics.jsonl').write_text(json.dumps(rec) + '\n', encoding='utf-8')
    teams = {'e5': _team('e5', ['A', 'B', 'C', 'D', 'E', 'Z']), 'r': _team('r', ['A', 'B', 'C', 'D', 'E', 'F'])}
    monkeypatch.setattr(K, '_teams_by_key', lambda split, n=0: {'e5': teams['e5']} if split == 'eval' else {'r': teams['r']})
    K.main([str(tmp_path), str(tmp_path)])
    md = (tmp_path / 'kickoff_summary8.md').read_text(encoding='utf-8')
    assert '| 170 equipos fit, semilla 0 | 60.0 % | 40.0 % | -20.0' in md
    assert '| 170 equipos fit, semilla 0 | 250k | 40.0 % |' in md
    assert '100 % (n=1)' in md
