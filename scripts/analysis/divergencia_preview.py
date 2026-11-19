"""¿Cuánto varía y cuánto cuesta el preview no determinista? (P1b, memorizacion_equipos.md §17.3)

Lee los ``policy_diag`` con ``PREVIEW_SAMPLE=1`` (``formacion`` jugada y ``formacion_determinista``) y
compara con la referencia determinista (mismo combate, preview determinista). Por modo: victorias
contra los 3 heurísticos en `val` y Δ frente a la referencia, % de partidas con otra formación, con
otros líderes, con otros 4 elegidos, y número de formaciones distintas usadas.

    python scripts/analysis/divergencia_preview.py ~/iapsvgc_runs/p1b ~/iapsvgc_runs/p5full
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')
MODES = (('pv_T1.0', 'Preview sorteado T = 1,0 (combate determinista)'),
         ('pv_T0.5', 'Preview sorteado T = 0,5 (combate determinista)'),
         ('pv_T0.5_cb_T0.5', 'Preview T = 0,5 + combate T = 0,5 (modo contra personas)'))


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def _wins(reps):
    w = n = 0
    for r in reps:
        for f in HEUR:
            x = (r or {}).get('resultados', {}).get(f, {})
            w += x.get('win', 0)
            n += sum(x.values())
    return w, n


def _variety(reps):
    games = [g for r in reps for g in (r or {}).get('partidas', [])
             if g.get('rival') in HEUR and g.get('formacion') and g.get('formacion_determinista')]
    if not games:
        return None
    lead = lambda f: frozenset(f[:2])  # noqa: E731
    four = lambda f: frozenset(f[:4])  # noqa: E731
    n = len(games)
    return {
        'formacion': sum(lead(g['formacion']) != lead(g['formacion_determinista'])
                         or four(g['formacion']) != four(g['formacion_determinista']) for g in games) / n,
        'lideres': sum(lead(g['formacion']) != lead(g['formacion_determinista']) for g in games) / n,
        'cuatro': sum(four(g['formacion']) != four(g['formacion_determinista']) for g in games) / n,
        'distintas': len({(lead(g['formacion']), four(g['formacion'])) for g in games}),
        'n': n,
    }


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    d, ref = Path(a[0]).expanduser(), Path(a[1]).expanduser()
    rw = _wins([_load(ref / 'diag' / f'ppo_p5_s{s}_val.json') for s in (0, 1)])
    L = ['| Modo | Victorias `val` | Δ vs determinista | Otra formación | Otros líderes | Otros 4 | '
         'Formaciones distintas | Partidas |', '|---|---|---|---|---|---|---|---|',
         f'| Determinista (preview v2.1 + combate PPO P5) | '
         f'{100 * rw[0] / rw[1]:.1f} % | — | — | — | — | — | {rw[1] // 3 if rw[1] else 0} |' if rw[1] else
         '| Determinista | sin datos | | | | | | |']
    for key, name in MODES:
        reps = [_load(d / 'diag' / f'{key}_s{s}_val.json') for s in (0, 1)]
        w = _wins(reps)
        v = _variety(reps)
        if not w[1]:
            L.append(f'| {name} | sin datos | | | | | | |')
            continue
        p, q = w[0] / w[1], (rw[0] / rw[1] if rw[1] else float('nan'))
        hw = 1.96 * math.sqrt(p * (1 - p) / w[1] + (q * (1 - q) / rw[1] if rw[1] else 0))
        dl = f'{100 * (p - q):+.1f} (± {100 * hw:.1f})' if rw[1] else '—'
        vv = (f"{100 * v['formacion']:.1f} % | {100 * v['lideres']:.1f} % | {100 * v['cuatro']:.1f} % | "
              f"{v['distintas']} | {v['n']}") if v else '— | — | — | — | —'
        L.append(f'| {name} | {100 * p:.1f} % | {dl} | {vv} |')
    out = '\n'.join(L) + '\n'
    (d / 'divergencia_preview.md').write_text(out, encoding='utf-8')
    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
