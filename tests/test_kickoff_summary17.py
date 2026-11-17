"""Resumen de la decimoséptima cola (showdown/rl/kickoff_summary17.py)."""
import json

from showdown.rl import kickoff_summary17 as K


def _diag(path, w):
    res = {f: {'win': w, 'loss': 200 - w} for f in K.HEUR}
    res['random'] = {'win': 199, 'loss': 1}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'resultados': res}), encoding='utf-8')


def test_rules_use_val_and_chosen_reg(tmp_path):
    q15, d = tmp_path / 'q15', tmp_path / 'q17'
    (q15 / 'diag').mkdir(parents=True)
    (q15 / 'chosen_reg.txt').write_text('--weight-decay 0.03 --dropout 0.1\n', encoding='utf-8')
    for s in (0, 1):
        _diag(q15 / 'diag' / f'R_wd03_s{s}_final_val.json', 120)   # 60 %
        _diag(q15 / 'diag' / f'REG_s{s}_final_val.json', 20)       # no debe usarse
        _diag(d / 'diag' / f'PPO1M_s{s}_final_val.json', 180)      # 90 %
        _diag(d / 'diag' / f'DDQN2M_s{s}_final_val.json', 140)     # 70 % → +10: sí
        _diag(d / 'diag' / f'PPO2M_s{s}_final_val.json', 182)      # 91 % → +1: no
        _diag(d / 'diag' / f'DSYM_s{s}_final_val.json', 124)       # 62 % → +2: no
        _diag(d / 'diag' / f'DDQN2M_s{s}_final_eval.json', 10)     # registro: no decide
    assert K.main([str(d), str(q15)]) == 0
    md = (d / 'kickoff_summary17.md').read_text(encoding='utf-8')
    rows = {l.split('|')[2].strip(): l for l in md.splitlines() if l.startswith('| A ') or l.startswith('| D ')}
    assert '| +10.0' in rows['DDQN 2M frente a DDQN 1M (REG elegida)'] and '| sí |' in rows['DDQN 2M frente a DDQN 1M (REG elegida)']
    assert '| no |' in rows['PPO 2M frente a PPO 1M (con simetría)']
    assert '| no |' in rows['DDQN 1M con simetría frente a sin ella (REG elegida)']
    assert 'R_wd03' in md


def test_missing_data(tmp_path):
    (tmp_path / 'a' / 'diag').mkdir(parents=True)
    (tmp_path / 'b').mkdir()
    assert K.main([str(tmp_path / 'a'), str(tmp_path / 'b')]) == 0
    assert 'sin datos' in (tmp_path / 'a' / 'kickoff_summary17.md').read_text(encoding='utf-8')
