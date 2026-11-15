"""Fase 2c de expectimax: conteo de cambios voluntarios, rebotes y turnos por juego."""
import sys

sys.path.insert(0, '.')

from showdown.inference_sim import analyze_switches as a


def test_bounce_and_turns(tmp_path):
    rows = [
        {'tag': 'b1', 'turn': 2, 'out': 'A', 'in': 'C'},
        {'tag': 'b1', 'turn': 3, 'out': 'C', 'in': 'A'},   # rebote: A salió hace 1 turno
        {'tag': 'b1', 'turn': 9, 'out': 'A', 'in': 'D'},   # no es rebote (D nunca salió)
    ]
    log = tmp_path / 'run.log'
    log.write_text('[1/2] LOSS g1:t4L c2=-1 g2:t12W c2=+1 | x\n', encoding='utf-8')
    text = '\n'.join(a.summarize(rows, a.game_turns([str(log)])))
    assert 'turnos medios por juego: 8.0' in text
    assert 'rebotes (vuelve a entrar quien salió hace <= 2 turnos): 1 (33.3 % de los cambios)' in text
