"""Estadística de la cola G/H contra arquetipos (protocolo §4.19). Exploración, bloque X1 = 11001–11017.

Uso: python gh_arq_stats.py <dir_raw> <dir_reports>
Brazos (300 BO3 por semilla):
  G0 / G3: G 5.2 contra greedy_joint nivel 0 / nivel 3      H0 / H3: H 2.2 ídem
  GA / GS / GF: G 5.2 + arquetipos / + serie BO3 / + hoja con campo, contra el nivel 3
  HA / HS / HF: H 2.2 ídem
Parte 1 (punto 1, descriptiva): caída nivel 0 → 3 de cada agente (McNemar pareado) y brecha H − G en cada nivel.
  Hipótesis pre-registrada sobre H: la brecha H − G es menor (menos negativa) contra el nivel 3 que contra el 0.
Parte 2 (puntos 2–4, exploración): cada extensión frente a su agente base contra el nivel 3 (McNemar).
  Regla: candidata a confirmación si la diferencia es > 0 con p < 0,05; la confirmación (bloques nuevos 11101+,
  también contra los baselines de siempre) se pre-registra aparte.
"""
from __future__ import annotations

import glob
import os
import sys
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from showdown.inference_sim.confirm_stats import binom_report, mcnemar, parse_series

BLOCK = '110'


def files(raw: str, arm: str) -> List[str]:
    return sorted(glob.glob(os.path.join(raw, f'{arm}_s{BLOCK}*.txt')))


def secs(rep: str, arm: str) -> float:
    vals = []
    for p in glob.glob(os.path.join(rep, f'{arm}_s{BLOCK}*.sec')):
        with open(p, encoding='utf-8') as fh:
            vals.append(float(fh.read().strip()))
    return sum(vals) / len(vals) if vals else float('nan')


def rate(raw, arm) -> str:
    s = parse_series(files(raw, arm))
    if not s:
        return 'sin datos'
    r = binom_report(s)
    return (f"{100 * r['rate']:.2f} % [{100 * r['ci_lo']:.2f}, {100 * r['ci_hi']:.2f}] "
            f"({r['n']} series, {r['seeds']} semillas)")


def mc(raw, a, b) -> str:
    sa, sb = parse_series(files(raw, a)), parse_series(files(raw, b))
    if not sa or not sb:
        return 'sin datos'
    r = mcnemar(sa, sb)
    return (f"{r['diff_pts']:+.2f} pts (emparejadas {r['matched']}, {a} mejor {r['a_better']}, "
            f"{b} mejor {r['b_better']}, p = {r['p']:.4g})")


def diff(raw, a, b):
    sa, sb = parse_series(files(raw, a)), parse_series(files(raw, b))
    if not sa or not sb:
        return None
    r = mcnemar(sa, sb)
    return r['diff_pts'], r['p']


def main(raw: str, rep: str) -> int:
    print('=== G y H contra greedy_joint en modo arquetipo (protocolo §4.19), X1 = 11001–11017 ===\n')
    print('--- Tasas de victoria (series BO3)')
    for arm, name in (('G0', 'G 5.2 vs nivel 0'), ('G3', 'G 5.2 vs nivel 3'), ('H0', 'H 2.2 vs nivel 0'),
                      ('H3', 'H 2.2 vs nivel 3'), ('GA', 'G + arquetipos vs nivel 3'),
                      ('GS', 'G + serie vs nivel 3'), ('GF', 'G + hoja de campo vs nivel 3'),
                      ('HA', 'H + arquetipos vs nivel 3'), ('HS', 'H + serie vs nivel 3'),
                      ('HF', 'H + hoja de campo vs nivel 3')):
        print(f'{arm:3s} {name:30s} {rate(raw, arm)}  ({secs(rep, arm):.0f} s/semilla)')
    print('\n--- Parte 1 (descriptiva): caída nivel 0 → 3 (G3 − G0, H3 − H0)')
    print(f'G: {mc(raw, "G3", "G0")}')
    print(f'H: {mc(raw, "H3", "H0")}')
    print('--- Brecha H − G en cada nivel')
    g0, g3 = diff(raw, 'H0', 'G0'), diff(raw, 'H3', 'G3')
    print(f'nivel 0: {mc(raw, "H0", "G0")}')
    print(f'nivel 3: {mc(raw, "H3", "G3")}')
    if g0 and g3:
        print(f'Hipótesis sobre H (brecha menor contra el nivel 3): {g3[0] - g0[0]:+.2f} pts de cambio -> '
              f'{"coincide" if g3[0] > g0[0] else "no coincide"} (descriptivo)')
    print('\n--- Parte 2 (exploración contra el nivel 3; candidata si > 0 y p < 0,05)')
    for a, b, name in (('GA', 'G3', 'G + arquetipos'), ('GS', 'G3', 'G + serie BO3'), ('GF', 'G3', 'G + hoja de campo'),
                       ('HA', 'H3', 'H + arquetipos'), ('HS', 'H3', 'H + serie BO3'), ('HF', 'H3', 'H + hoja de campo')):
        d = diff(raw, a, b)
        verdict = '' if d is None else ('CANDIDATA a confirmación' if (d[0] > 0 and d[1] < 0.05) else 'no pasa')
        print(f'{name:20s} {mc(raw, a, b)}  -> {verdict}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
