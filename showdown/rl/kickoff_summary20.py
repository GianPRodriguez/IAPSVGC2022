"""Resumen de la vigésima cola: preview y combate por separado, y piloto con equipos propios al azar.

Lee ``<dir>/diag/*.json`` (policy_diag) y escribe ``<dir>/kickoff_summary20.md``
(memorizacion_equipos.md §13). Referencias: decimonovena cola (equipo propio al azar con los
agentes de equipo fijo) y colas 12/15 (equipo fijo).

    python -m showdown.rl.kickoff_summary20 ~/iapsvgc_runs/kickoff20 ~/iapsvgc_runs
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

from showdown.rl.context_diag import summarize

HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')
AGENTS = ('ddqn', 'ppo')
LABEL = {'ddqn': 'DDQN', 'ppo': 'PPO'}


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def pooled(reps: List[Optional[dict]], foes=HEUR) -> Optional[Tuple[float, int]]:
    w = n = 0
    for r in reps:
        if not r:
            continue
        res = r.get('resultados', {})
        w += sum(res.get(f, {}).get('win', 0) for f in foes)
        n += sum(sum(res.get(f, {}).values()) for f in foes)
    return (w / n, n) if n else None


def _pct(x) -> str:
    return '—' if not x else f'{100 * x[0]:.1f} %'


def _diff(a, b) -> str:
    if not a or not b:
        return '—'
    se = math.sqrt(a[0] * (1 - a[0]) / a[1] + b[0] * (1 - b[0]) / b[1])
    return f'{100 * (a[0] - b[0]):+.1f} (± {100 * se:.1f})'


def formations(reps: List[Optional[dict]]) -> Counter:
    c = Counter()
    for r in reps:
        for g in (r or {}).get('partidas', []):
            if g.get('formacion') and g['rival'] in HEUR:
                c[g['formacion']] += 1
    return c


def build(d: Path, runs: Path) -> str:
    D = d / 'diag'
    L = ['# Resumen de la vigésima cola: preview y combate por separado; piloto con equipos al azar', '']

    # ------------------------------------------------------------------ 1. preview
    L += ['## 1. Contribución del preview (tu equipo, rivales `eval`, 200 partidas por rival)', '',
          'Mismo combate (final.pt de la configuración final), tres previews: **v2** (la red), '
          '**formación fija** (la que más elige v2 con ese agente, jugada siempre) y **/team 1234** (los '
          'cuatro primeros del equipo). v2 − fija = cuánto aporta adaptar la formación a cada rival; '
          'fija − 1234 = cuánto aporta elegir bien los cuatro. 3 heurísticos, 2 semillas agrupadas.', '',
          '| Agente | Preview v2 | Formación fija | /team 1234 | v2 − fija (adaptación) | fija − 1234 (elección) |',
          '|---|---|---|---|---|---|']
    for a in AGENTS:
        v2 = pooled([_load(D / f'{a}_s{s}_v2.json') for s in (0, 1)])
        fx = pooled([_load(D / f'{a}_s{s}_moda.json') for s in (0, 1)])
        f1 = pooled([_load(D / f'{a}_s{s}_1234.json') for s in (0, 1)])
        L.append(f'| {LABEL[a]} | {_pct(v2)} | {_pct(fx)} | {_pct(f1)} | {_diff(v2, fx)} | {_diff(fx, f1)} |')
    L += ['', '### Formaciones que elige el preview v2', '',
          '| Agente | Formaciones distintas | Más frecuente (% de partidas) | Las 3 más frecuentes (%) |',
          '|---|---|---|---|']
    for a in AGENTS:
        c = formations([_load(D / f'{a}_s{s}_v2.json') for s in (0, 1)])
        n = sum(c.values())
        if not n:
            L.append(f'| {LABEL[a]} | — | — | — |')
            continue
        top = c.most_common(3)
        L.append(f'| {LABEL[a]} | {len(c)} | {top[0][0]} ({100 * top[0][1] / n:.0f} %) | '
                 f'{100 * sum(x[1] for x in top) / n:.0f} % |')
    L += ['', 'Formación como posiciones 1–6 del equipo fijo (`showdown/rl/data/ddqn_team.txt`).', '']

    # ------------------------------------------------------------------ 2. combate en su distribución
    L += ['## 2. Combate según el plan del rival, con tu equipo (en su distribución)', '',
          'Decisiones libres contra los 3 heurísticos con el preview v2 (la decimonovena cola lo midió '
          'con equipos propios al azar, fuera de distribución).', '']
    for a in AGENTS:
        ctx = defaultdict(Counter)
        for s in (0, 1):
            for c, v in ((_load(D / f'{a}_s{s}_v2.json') or {}).get('contexto') or {}).items():
                ctx[c].update(v)
        L += [f'### {LABEL[a]}', ''] + (summarize(ctx) if ctx else ['Sin datos.']) + ['']

    # ------------------------------------------------------------------ 3. piloto con equipos al azar
    L += ['## 3. Piloto: entrenar con equipo propio al azar (1M, 2 semillas)', '',
          'Entrenado con un equipo propio sorteado en cada partida (grupo `fit`); evaluado contra '
          'rivales `eval` con equipo propio al azar (`eval`) y con tu equipo fijo. Referencias: los '
          'agentes entrenados con tu equipo (equipo propio al azar: decimonovena cola; tu equipo: '
          'colas 15 y 12). 3 heurísticos.', '',
          '| Agente | Entrenado con | Equipo propio al azar | Tu equipo |', '|---|---|---|---|']
    refs = {'ddqn': ([runs / 'kickoff19' / 'diag' / f'ddqn_s{s}.json' for s in (0, 1)],
                     [runs / 'kickoff15' / 'diag' / f'R_wd03_s{s}_final_eval.json' for s in (0, 1)]),
            'ppo': ([runs / 'kickoff19' / 'diag' / f'ppo_s{s}.json' for s in (0, 1)],
                    [runs / 'kickoff12' / 'diag' / f'PPO_sym_s{s}_final_eval.json' for s in (0, 1)])}
    for a in AGENTS:
        rr = pooled([_load(p) for p in refs[a][0]])
        rf = pooled([_load(p) for p in refs[a][1]])
        pr = pooled([_load(D / f'pool_{a}_s{s}_azar.json') for s in (0, 1)])
        pf = pooled([_load(D / f'pool_{a}_s{s}_fijo.json') for s in (0, 1)])
        L.append(f'| {LABEL[a]} | tu equipo fijo | {_pct(rr)} | {_pct(rf)} |')
        L.append(f'| {LABEL[a]} | **equipo al azar** | **{_pct(pr)}** ({_diff(pr, rr)}) | **{_pct(pf)}** ({_diff(pf, rf)}) |')
    L += ['', 'Entre paréntesis: diferencia con el mismo agente entrenado con tu equipo fijo.', '']
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    d, runs = Path(args[0]).expanduser(), Path(args[1]).expanduser()
    (d / 'kickoff_summary20.md').write_text(build(d, runs), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
