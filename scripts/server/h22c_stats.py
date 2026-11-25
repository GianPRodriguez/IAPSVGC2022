"""Estadística pre-registrada de la confirmación de H 2.2 = B2 (protocolo §4.9, spec de H §4e).

Uso: python h22c_stats.py <dir_raw> <dir_reports>
Brazos (C17 = 10801–10817, C18 = 10901–10917): Bj/Be/Bs = B2 contra joint/expectimax/solo,
Hj = H 2.1 contra joint, Gj = G 5.2 contra joint.
  U1–U3: B2 > baseline (binomial bilateral; IC bootstrap por clúster).
  U4: B2 no inferior a H 2.1 contra joint (McNemar pareado; H0: diferencia ≤ −1,5 pts; p unilateral
      por aproximación normal; se reporta la cota inferior unilateral 95 %).
  Holm m = 4 sobre U1–U4. Condición de tiempo: s/semilla de Bj ≤ 85 % del de Hj.
  Descriptivo: B2 frente a G 5.2 contra joint (McNemar) y por bloque.
"""
from __future__ import annotations

import glob
import math
import os
import sys
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from showdown.inference_sim.confirm_stats import binom_report, holm, mcnemar, parse_series

NI_MARGIN = -0.015
TIME_CAP = 0.85
BLOCKS = {'C17': '108', 'C18': '109'}


def files(raw: str, arm: str, block: str = '10[89]') -> List[str]:
    return sorted(glob.glob(os.path.join(raw, f'{arm}_s{block}*.txt')))


def secs(rep: str, arm: str) -> float:
    vals = []
    for p in glob.glob(os.path.join(rep, f'{arm}_s10[89]*.sec')):
        with open(p, encoding='utf-8') as fh:
            vals.append(float(fh.read().strip()))
    return sum(vals) / len(vals) if vals else float('nan')


def ni_test(a, b):
    """(diferencia pts, cota inferior unilateral 95 % pts, p unilateral de H0: dif ≤ margen, r)."""
    r = mcnemar(a, b)
    n, n10, n01 = r['matched'], r['a_better'], r['b_better']
    d = (n10 - n01) / n
    se = math.sqrt(max(1e-12, ((n10 + n01) / n - d * d) / n))
    z = (d - NI_MARGIN) / se
    return 100 * d, 100 * (d - 1.645 * se), 0.5 * math.erfc(z / math.sqrt(2)), r


def fmt_binom(r) -> str:
    return (f"series={r['n']} victorias={r['wins']:.1f} tasa={100 * r['rate']:.2f}% "
            f"IC95_cluster=[{100 * r['ci_lo']:.2f}, {100 * r['ci_hi']:.2f}] p={r['p']:.4g} "
            f"semillas>50%={r['seeds_above_50']}/{r['seeds']}")


def fmt_mc(r) -> str:
    return (f"emparejadas={r['matched']} A_mejor={r['a_better']} B_mejor={r['b_better']} "
            f"diferencia={r['diff_pts']:+.2f} pts p={r['p']:.4g}")


def main(raw: str, rep: str) -> int:
    ps = []
    print('=== Confirmación de H 2.2 = B2 (H 2.1 + RM+), C17 = 10801–10817, C18 = 10901–10917 ===')
    for arm, name in (('Bj', 'U1 B2 > greedy_joint'), ('Be', 'U2 B2 > expectimax'), ('Bs', 'U3 B2 > greedy_solo')):
        r = binom_report(parse_series(files(raw, arm)))
        ps.append(r['p'])
        print(f'\n--- {name} ({arm}, {secs(rep, arm):.0f} s/semilla)\n{fmt_binom(r)}')
        for blk, pre in BLOCKS.items():
            print(f'    {blk}: {fmt_binom(binom_report(parse_series(files(raw, arm, pre))))}')
    d, lo, p4, r4 = ni_test(parse_series(files(raw, 'Bj')), parse_series(files(raw, 'Hj')))
    ps.append(p4)
    print(f'\n--- U4 B2 no inferior a H 2.1 (contra joint, margen {100 * NI_MARGIN:+.1f} pts)')
    print(f'{fmt_mc(r4)}  cota_inf_95={lo:+.2f} pts  p_NI={p4:.4g}')
    for blk, pre in BLOCKS.items():
        db, lob, pb, rb = ni_test(parse_series(files(raw, 'Bj', pre)), parse_series(files(raw, 'Hj', pre)))
        print(f'    {blk}: {fmt_mc(rb)}  cota_inf_95={lob:+.2f} pts  p_NI={pb:.4g}')
    print(f'--- Hj (H 2.1 contra joint, {secs(rep, "Hj"):.0f} s/semilla)\n'
          f'{fmt_binom(binom_report(parse_series(files(raw, "Hj"))))}')
    sb, sh = secs(rep, 'Bj'), secs(rep, 'Hj')
    time_ok = sb <= TIME_CAP * sh
    print(f'\n--- condición de tiempo: Bj {sb:.0f} s frente a Hj {sh:.0f} s = {100 * sb / sh:.0f} % '
          f'(tope {100 * TIME_CAP:.0f} %) -> {"cumple" if time_ok else "no cumple"}')
    print(f'\n--- descriptivo: B2 frente a G 5.2 (contra joint; Gj {secs(rep, "Gj"):.0f} s/semilla)')
    print(fmt_mc(mcnemar(parse_series(files(raw, 'Bj')), parse_series(files(raw, 'Gj')))))
    for blk, pre in BLOCKS.items():
        print(f'    {blk}: {fmt_mc(mcnemar(parse_series(files(raw, "Bj", pre)), parse_series(files(raw, "Gj", pre))))}')
    print(fmt_binom(binom_report(parse_series(files(raw, 'Gj')))))
    print('\n--- Holm (m = 4; U1, U2, U3, U4)')
    res = holm(ps)
    for name, (p, a, rej) in zip(('U1', 'U2', 'U3', 'U4'), res):
        print(f'{name}: p={p:.4g} p_holm={a:.4g} {"rechaza H0" if rej else "no rechaza"}')
    ok = all(rej for _p, _a, rej in res) and time_ok
    print(f'\nDecisión pre-registrada: {"B2 = H 2.2 (default del player)" if ok else "H se queda en 2.1"}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
