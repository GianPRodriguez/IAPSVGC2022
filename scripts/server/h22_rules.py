"""Reglas pre-registradas de la exploración de H 2.2 (protocolo §4.8, spec de H §4d).

Uso: python h22_rules.py <h22_summary.txt> "<fb B1 B2 B3>" "<err B1 B2 B3>" "<s B1 B2 B3>" <s R>
Lee las líneas McNemar "--- Bk frente a R" del resumen y aplica:
  superioridad (B1, B3): 0 errores, fallbacks ≤ 5 % de 5100 series, diferencia > 0 y p < 0,05;
  no inferioridad (B2): 0 errores, fallbacks ≤ 5 %, cota inferior unilateral 95 % > −1,5 pts
  (aprox. normal de la diferencia pareada) y tiempo por semilla ≤ 85 % del de R.
"""
from __future__ import annotations

import math
import re
import sys
from typing import Optional, Tuple

ARMS = ('B1', 'B2', 'B3')
NI_MARGIN = -1.5
TIME_CAP = 0.85


def mcnemar_line(txt: str, arm: str) -> Optional[Tuple[float, float, float]]:
    """(diferencia en pts, p, cota inferior unilateral 95 % en pts) o None."""
    m = re.search(rf'--- {arm} frente a R\nemparejadas=(\d+) A_mejor=(\d+) B_mejor=(\d+) '
                  rf'diferencia=([+-][\d.]+) pts p=([\d.e+-]+)', txt)
    if not m:
        return None
    n, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
    d = (a - b) / n
    se = math.sqrt(max(0.0, (a + b) / n - d * d) / n)
    return 100 * d, float(m.group(5)), 100 * (d - 1.645 * se)


def main(argv) -> int:
    txt = open(argv[1], encoding='utf-8').read()
    fb = dict(zip(ARMS, map(int, argv[2].split())))
    er = dict(zip(ARMS, map(int, argv[3].split())))
    sec = dict(zip(ARMS, map(int, argv[4].split())))
    sr = int(argv[5])
    ok = lambda c: er[c] == 0 and fb[c] <= 0.05 * 5100  # noqa: E731
    print('\n=== Reglas pre-registradas ===')
    best = None
    for c in ('B1', 'B3'):
        r = mcnemar_line(txt, c)
        passed = bool(r) and ok(c) and r[0] > 0 and r[1] < 0.05
        d, p = (r[0], r[1]) if r else (float('nan'), float('nan'))
        print(f'superioridad {c}: diferencia={d:+.2f} pts p={p:.4g} errores={er[c]} fallbacks={fb[c]} '
              f'-> {"pasa" if passed else "no pasa"}')
        if passed and (best is None or (r[0], -sec[c]) > best[0]):
            best = ((r[0], -sec[c]), c)
    r = mcnemar_line(txt, 'B2')
    ni = bool(r) and ok('B2') and r[2] > NI_MARGIN and sec['B2'] <= TIME_CAP * sr
    d, lo = (r[0], r[2]) if r else (float('nan'), float('nan'))
    print(f'no inferioridad B2: diferencia={d:+.2f} pts cota_inf_95={lo:+.2f} pts (margen {NI_MARGIN}) '
          f'tiempo={sec["B2"]} s frente a R {sr} s ({100 * sec["B2"] / max(1, sr):.0f} %, tope '
          f'{100 * TIME_CAP:.0f} %) errores={er["B2"]} fallbacks={fb["B2"]} -> {"pasa" if ni else "no pasa"}')
    print(f'\nBrazo ganador (superioridad): {best[1] if best else "ninguno"}')
    print(f'B2 no inferior y más rápido: {"sí" if ni else "no"}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
