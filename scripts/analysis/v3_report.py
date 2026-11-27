"""Informe de PPO v3 / DDQN v2 con las reglas pre-registradas (protocolo §4.16).

Lee los JSON de ``policy_diag`` que deja ``scripts/server/v3_queue.sh`` en ``<OUT>/diag``:

- **P1** (máscara del Protect consecutivo sin reentrenar): ``p1_<fam>_s<k>_{off,on}.json``. Mismas semillas en
  los dos brazos → partidas pareadas por orden dentro de cada rival. McNemar; no inferioridad si la cota inferior
  unilateral 95 % de (on − off) > −1,5 pts. Se usan joint, solo y expectimax (random es control aparte).
- **S1** (preview condicionado sin reentrenar): ``s1_<fam>_s<k>_{off,on}.json`` (series). La partida 1 es igual;
  se compara la puntuación en las partidas 2–3 pareando por serie. Regla: +1,5 pts o más y p < 0,05 unilateral.
- **V** (memoria de serie): ``v_<fam>_{C,V}_s<k>.json`` (series, ``--seed 40000``). Primaria: tasa de victorias en
  las partidas 2–3, V − C, media de las semillas (IC t entre semillas) y prueba z de dos proporciones agrupada,
  unilateral, Holm m = 2 (PPO, DDQN). Secundarias: series ganadas y no inferioridad en la partida 1 (−1,5 pts).

    python scripts/analysis/v3_report.py <OUT>     # escribe <OUT>/v3_report.md
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')
FAMS = ('ppo', 'ddqn')
MARGIN = 0.015


def _load(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _t95(df: int) -> float:
    return {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(df, 1.96)


def mcnemar(pairs: List[Tuple[int, int]]) -> Dict[str, float]:
    """pairs = (gana_brazo_A, gana_brazo_B) por partida. Diferencia B − A pareada."""
    n = len(pairs)
    b = sum(1 for a, c in pairs if c and not a)
    c_ = sum(1 for a, c in pairs if a and not c)
    if n == 0:
        return {'n': 0}
    d = (b - c_) / n
    var = max(0.0, (b + c_) - (b - c_) ** 2 / n) / (n * n)
    se = math.sqrt(var)
    p_sup = 1.0 - _phi(d / se) if se > 0 else (0.0 if d > 0 else 1.0)
    return {'n': n, 'b': b, 'c': c_, 'diff': d, 'se': se, 'lo95_unilateral': d - 1.645 * se, 'p_superioridad': p_sup}


def p1(diag: Path) -> List[str]:
    out = ['## P1. Máscara del Protect consecutivo (sin reentrenar)', '',
           '| Familia | Partidas pareadas | on − off | Cota inferior 95 % | ¿No inferior (> −1,5)? | Situaciones / repeticiones (off) |',
           '|---|---|---|---|---|---|']
    for fam in FAMS:
        pairs, sit, rep = [], 0, 0
        for k in range(5):
            off, on = _load(diag / f'p1_{fam}_s{k}_off.json'), _load(diag / f'p1_{fam}_s{k}_on.json')
            if not off or not on:
                continue
            by_a, by_b = defaultdict(list), defaultdict(list)
            for g in off.get('partidas', []):
                by_a[g['rival']].append(g['res'] == 'win')
            for g in on.get('partidas', []):
                by_b[g['rival']].append(g['res'] == 'win')
            for foe in HEUR:
                pairs += list(zip(by_a[foe], by_b[foe]))
            ms = (off.get('mascara_protect_consecutivo') or {}).get('por_rival', {})
            for foe in HEUR:
                sit += int((ms.get(foe) or {}).get('situaciones', 0))
                rep += int((ms.get(foe) or {}).get('repitio_protect', 0))
        r = mcnemar(pairs)
        if not r.get('n'):
            out.append(f'| {fam} | 0 | – | – | – | – |')
            continue
        ok = 'sí' if r['lo95_unilateral'] > -MARGIN else 'no'
        out.append(f"| {fam} | {r['n']} | {100 * r['diff']:+.2f} pts | {100 * r['lo95_unilateral']:+.2f} pts | {ok} | "
                   f'{sit} / {rep} |')
    return out + ['']


def _series_scores(d: dict) -> Dict[Tuple[str, int], Tuple[int, int]]:
    """(rival, serie) -> (victorias, partidas) en las partidas 2–3."""
    acc: Dict[Tuple[str, int], List[int]] = defaultdict(lambda: [0, 0])
    for g in d.get('partidas', []):
        if g.get('rival') not in HEUR or int(g.get('partida', 1)) < 2:
            continue
        key = (g['rival'], int(g.get('serie', 0)))
        acc[key][0] += int(g['res'] == 'win')
        acc[key][1] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}


def s1(diag: Path) -> List[str]:
    out = ['## S1. Preview condicionado a la serie (sin reentrenar)', '',
           '| Familia | Series pareadas | on − off (partidas 2–3) | p unilateral | ¿Se adopta (≥ +1,5 y p < 0,05)? |',
           '|---|---|---|---|---|']
    for fam in FAMS:
        diffs = []
        for k in range(5):
            off, on = _load(diag / f's1_{fam}_s{k}_off.json'), _load(diag / f's1_{fam}_s{k}_on.json')
            if not off or not on:
                continue
            a, b = _series_scores(off), _series_scores(on)
            for key in set(a) & set(b):
                diffs.append(b[key][0] / b[key][1] - a[key][0] / a[key][1])
        n = len(diffs)
        if n < 2:
            out.append(f'| {fam} | {n} | – | – | – |')
            continue
        m = sum(diffs) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
        p = 1.0 - _phi(m / (sd / math.sqrt(n))) if sd > 0 else (0.0 if m > 0 else 1.0)
        ok = 'sí' if (m >= MARGIN and p < 0.05) else 'no'
        out.append(f'| {fam} | {n} | {100 * m:+.2f} pts | {p:.3g} | {ok} |')
    return out + ['']


def _rates(d: dict) -> Dict[str, Tuple[int, int]]:
    """'g1', 'g23', 'series' -> (victorias, total) contra los heurísticos."""
    r = {'g1': [0, 0], 'g23': [0, 0], 'series': [0, 0]}
    ser: Dict[Tuple[str, int], List[int]] = defaultdict(lambda: [0, 0])
    for g in d.get('partidas', []):
        if g.get('rival') not in HEUR:
            continue
        w = int(g['res'] == 'win')
        k = 'g1' if int(g.get('partida', 1)) == 1 else 'g23'
        r[k][0] += w
        r[k][1] += 1
        s = ser[(g['rival'], int(g.get('serie', 0)))]
        s[0] += w
        s[1] += 1 - w
    for w, lo in ser.values():
        r['series'][0] += int(w > lo)
        r['series'][1] += 1
    return {k: (v[0], v[1]) for k, v in r.items()}


def _ztest(w1: int, n1: int, w2: int, n2: int) -> Tuple[float, float, float]:
    """(diferencia p2 − p1, cota inferior unilateral 95 %, p unilateral de superioridad)."""
    if not n1 or not n2:
        return float('nan'), float('nan'), float('nan')
    p1_, p2_ = w1 / n1, w2 / n2
    d = p2_ - p1_
    se = math.sqrt(p1_ * (1 - p1_) / n1 + p2_ * (1 - p2_) / n2)
    pool = (w1 + w2) / (n1 + n2)
    se0 = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    p = 1.0 - _phi(d / se0) if se0 > 0 else (0.0 if d > 0 else 1.0)
    return d, d - 1.645 * se, p


def v(diag: Path) -> List[str]:
    out = ['## V. Memoria de serie (confirmación): V − C', '',
           '| Familia | Semillas | Partidas 2–3: C → V | V − C (media de semillas, IC 95 % entre semillas) | p unilateral (z agrupada) | p Holm | Partida 1: V − C (cota inf.) | Series: C → V |',
           '|---|---|---|---|---|---|---|---|']
    rows = []
    for fam in FAMS:
        per_seed, tot = [], {'C': defaultdict(lambda: [0, 0]), 'V': defaultdict(lambda: [0, 0])}
        for k in range(3):
            dc, dv = _load(diag / f'v_{fam}_C_s{k}.json'), _load(diag / f'v_{fam}_V_s{k}.json')
            if not dc or not dv:
                continue
            rc, rv = _rates(dc), _rates(dv)
            if rc['g23'][1] and rv['g23'][1]:
                per_seed.append(rv['g23'][0] / rv['g23'][1] - rc['g23'][0] / rc['g23'][1])
            for arm, rr in (('C', rc), ('V', rv)):
                for key, (w, n) in rr.items():
                    tot[arm][key][0] += w
                    tot[arm][key][1] += n
        if not per_seed:
            rows.append((fam, None))
            continue
        m = sum(per_seed) / len(per_seed)
        if len(per_seed) > 1:
            sd = math.sqrt(sum((x - m) ** 2 for x in per_seed) / (len(per_seed) - 1))
            h = _t95(len(per_seed) - 1) * sd / math.sqrt(len(per_seed))
        else:
            h = float('nan')
        C, V = tot['C'], tot['V']
        d, _lo, p = _ztest(C['g23'][0], C['g23'][1], V['g23'][0], V['g23'][1])
        d1, lo1, _p1 = _ztest(C['g1'][0], C['g1'][1], V['g1'][0], V['g1'][1])
        rows.append((fam, dict(n=len(per_seed), m=m, h=h, p=p, d1=d1, lo1=lo1,
                               c23=C['g23'], v23=V['g23'], cs=C['series'], vs=V['series'])))
    # Holm m = 2
    ps = sorted([(r[1]['p'], r[0]) for r in rows if r[1]], key=lambda x: x[0])
    holm, prev = {}, 0.0
    for i, (p, fam) in enumerate(ps):
        adj = min(1.0, max(prev, p * (len(ps) - i)))
        holm[fam] = prev = adj
    for fam, r in rows:
        if not r:
            out.append(f'| {fam} | 0 | – | – | – | – | – | – |')
            continue

        def pct(t):
            return f'{100 * t[0] / t[1]:.1f} %' if t[1] else '–'
        conf = ' **confirmada**' if holm.get(fam, 1.0) < 0.05 and r['m'] > 0 else ''
        out.append(f"| {fam} | {r['n']} | {pct(r['c23'])} → {pct(r['v23'])} | {100 * r['m']:+.2f} pts "
                   f"(± {100 * r['h']:.2f}) | {r['p']:.3g} | {holm.get(fam, float('nan')):.3g}{conf} | "
                   f"{100 * r['d1']:+.2f} ({100 * r['lo1']:+.2f}; {'no inferior' if r['lo1'] > -MARGIN else 'no demostrada'}) | "
                   f"{pct(r['cs'])} → {pct(r['vs'])} |")
    return out + ['']


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]) if argv else Path('.')
    diag = out / 'diag'
    lines = ['# PPO v3 / DDQN v2: informe (protocolo §4.16)', '',
             f'Generado por `scripts/analysis/v3_report.py` desde `{diag}`. Rivales: joint, solo y expectimax '
             '(random es control aparte). Equipos `eval`, rivales deterministas.', '']
    lines += p1(diag) + s1(diag) + v(diag)
    (out / 'v3_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
