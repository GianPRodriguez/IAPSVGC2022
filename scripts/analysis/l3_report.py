"""Informe del entrenamiento contra greedy_joint nivel 3 (protocolo §4.20): brazo L3 frente a C.

Lee los JSON de ``policy_diag`` (con ``--context``) que deja ``scripts/server/l3_queue.sh`` en ``<OUT>/diag``:
``ev_<fam>_{L3,C,prev}_s<k>.json``. Mismas semillas en todos los brazos → partidas pareadas por orden dentro de cada
rival (como P1 de §4.16).

- **T (PPO, DDQN):** L3 − C contra ``greedy_joint_arq3``, McNemar pareado sobre las 3 semillas; superioridad,
  p unilateral.
- **N (PPO, DDQN):** no inferioridad de L3 frente a C contra los 3 heurísticos (greedy_solo, greedy_joint,
  expectimax): H0 d ≤ −1,5 pts, p unilateral.
- Holm m = 4 sobre T_ppo, T_ddqn, N_ppo, N_ddqn. Descriptivo: por rival y semilla, y frente a la versión anterior.

    python scripts/analysis/l3_report.py <OUT>     # escribe <OUT>/l3_report.md
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')
L3 = 'greedy_joint_arq3'
FAMS = ('ppo', 'ddqn')
SEEDS = (0, 1, 2)
MARGIN = 0.015
ALPHA = 0.05


def _load(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def paired(diag: Path, fam: str, a: str, b: str, foes: Sequence[str]) -> Dict[str, float]:
    """Diferencia pareada b − a (victorias) sobre los rivales ``foes`` y las semillas disponibles."""
    pairs: List[Tuple[int, int]] = []
    for k in SEEDS:
        da, db = _load(diag / f'ev_{fam}_{a}_s{k}.json'), _load(diag / f'ev_{fam}_{b}_s{k}.json')
        if not da or not db:
            continue
        ga, gb = defaultdict(list), defaultdict(list)
        for g in da.get('partidas', []):
            ga[g['rival']].append(int(g['res'] == 'win'))
        for g in db.get('partidas', []):
            gb[g['rival']].append(int(g['res'] == 'win'))
        for f in foes:
            pairs += list(zip(ga[f], gb[f]))
    n = len(pairs)
    if n == 0:
        return {'n': 0}
    up = sum(1 for x, y in pairs if y and not x)
    down = sum(1 for x, y in pairs if x and not y)
    d = (up - down) / n
    se = math.sqrt(max(1e-12, (up + down) - (up - down) ** 2 / n)) / n
    return {'n': n, 'diff': d, 'se': se, 'lo95': d - 1.645 * se,
            'p_sup': 1.0 - _phi(d / se), 'p_ni': 1.0 - _phi((d + MARGIN) / se)}


def holm(pvals: Dict[str, float]) -> Dict[str, Tuple[float, bool]]:
    order = sorted(pvals, key=lambda k: pvals[k])
    m, run, stop, out = len(order), 0.0, False, {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        rej = (not stop) and pvals[k] <= ALPHA / (m - i)
        stop = stop or not rej
        out[k] = (run, rej)
    return out


def rate(diag: Path, fam: str, arm: str, k: int, foes: Sequence[str]) -> Optional[float]:
    d = _load(diag / f'ev_{fam}_{arm}_s{k}.json')
    if not d:
        return None
    res = d.get('resultados') or {}
    w = sum((res.get(f) or {}).get('win', 0) for f in foes)
    n = sum(sum((res.get(f) or {}).values()) for f in foes)
    return w / n if n else None


def build(out: Path) -> str:
    diag = out / 'diag'
    tests, rows = {}, {}
    for fam in FAMS:
        rows[f'T_{fam}'] = paired(diag, fam, 'C', 'L3', (L3,))
        rows[f'N_{fam}'] = paired(diag, fam, 'C', 'L3', HEUR)
        if rows[f'T_{fam}'].get('n'):
            tests[f'T_{fam}'] = rows[f'T_{fam}']['p_sup']
        if rows[f'N_{fam}'].get('n'):
            tests[f'N_{fam}'] = rows[f'N_{fam}']['p_ni']
    hm = holm(tests) if tests else {}
    L = ['# Entrenamiento contra greedy_joint nivel 3: L3 frente a C (protocolo §4.20)', '',
         'Partidas pareadas por rival y semilla (`eval`, 200 por rival, rivales deterministas, máscara activada).', '',
         '| Prueba | Pares | L3 − C | Cota inf. 95 % | p | p Holm | Resultado |', '|---|---|---|---|---|---|---|']
    for k, name in (('T_ppo', 'T PPO: L3 > C contra el nivel 3'), ('T_ddqn', 'T DDQN: L3 > C contra el nivel 3'),
                    ('N_ppo', 'N PPO: L3 no inferior a C contra las 3 heurísticas'),
                    ('N_ddqn', 'N DDQN: L3 no inferior a C contra las 3 heurísticas')):
        r = rows[k]
        if not r.get('n'):
            L.append(f'| {name} | 0 | – | – | – | – | sin datos |')
            continue
        p = r['p_sup'] if k.startswith('T') else r['p_ni']
        ph, rej = hm.get(k, (float('nan'), False))
        L.append(f"| {name} | {r['n']} | {100 * r['diff']:+.2f} pts | {100 * r['lo95']:+.2f} pts | {p:.3g} | {ph:.3g} | "
                 f"{'**se confirma**' if rej else 'no se confirma'} |")
    L += ['', '## Por rival y semilla (tasa de victorias)', '',
          '| Familia | Brazo | Semilla | 3 heurísticas | joint | nivel 3 |', '|---|---|---|---|---|---|']
    for fam in FAMS:
        for arm in ('prev', 'C', 'L3'):
            for k in SEEDS:
                h, j, l3 = (rate(diag, fam, arm, k, f) for f in (HEUR, ('greedy_joint',), (L3,)))
                if h is None:
                    continue
                L.append(f'| {fam} | {arm} | s{k} | {100 * h:.1f} % | {100 * j:.1f} % | {100 * l3:.1f} % |')
    L += ['', 'Decisión pre-registrada (§4.20): en cada familia donde T y N se confirman, el brazo L3 es la versión '
          'oficial (PPO v3 / DDQN v2). Si T se confirma y N no, decide el usuario.']
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0])
    txt = build(out)
    (out / 'l3_report.md').write_text(txt, encoding='utf-8')
    print(txt)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
