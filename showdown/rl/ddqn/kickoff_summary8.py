"""Resumen de la octava cola: memorización de los equipos rivales y cómo reducirla.

    python -m showdown.rl.ddqn.kickoff_summary8 ~/iapsvgc_runs/kickoff8 ~/iapsvgc_runs

Para cada corrida (validación reservada ``val``):
- victorias con sus propios equipos de entrenamiento y con equipos ``eval`` no vistos, y la brecha;
- el mejor paso según ``val`` (equipos que la corrida no entrenó).
Además, la tasa de victorias contra equipos ``eval`` según cuántas especies comparte cada uno
con el equipo de entrenamiento más parecido (si gana más cuanto más se parece, memoriza
combinaciones concretas).
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from showdown.rl.ddqn.kickoff_summary import _eval_cells, _evals

FOES = ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')
# nombre, descripción, subconjunto de equipos (0 = todos los de fit), equipos de referencia
RUNS = (
    ('D0_mini', 'mini kick-off anterior (200 equipos train, sin val)', 0, 'train'),
    ('F170_s0', '170 equipos fit, semilla 0', 0, 'fit'),
    ('SY50_s0', '170 fit + 50 % sintéticos, semilla 0', 0, 'fit'),
    ('S50_s0', '50 equipos, semilla 0', 50, 'fit'),
    ('S100_s0', '100 equipos, semilla 0', 100, 'fit'),
    ('F170_s1', '170 equipos fit, semilla 1', 0, 'fit'),
    ('SY50_s1', '170 fit + 50 % sintéticos, semilla 1', 0, 'fit'),
)


def _wr(res: Dict[str, Dict[str, int]], foes=FOES) -> Tuple[float, int]:
    w = sum(res.get(f, {}).get('win', 0) for f in foes)
    n = sum(sum(res.get(f, {}).values()) for f in foes)
    return (w / n if n else float('nan')), n


def _cell(res, foe) -> str:
    r = res.get(foe, {})
    n = sum(r.values())
    return f"{100 * r.get('win', 0) / n:.0f} %" if n else '—'


def _load(p: Path) -> Optional[dict]:
    return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else None


def _teams_by_key(split: str, subset_n: int = 0) -> Dict[str, object]:
    from showdown.inference_sim.team_factory import active_teams
    from showdown.inference_sim.team_split import team_key
    from showdown.rl.foe_pool import subset
    teams = list(active_teams(split))
    if subset_n:
        teams = subset(teams, subset_n)
    return {team_key(t): t for t in teams}


def similarity_rows(games: List[dict], refs: List, eval_by_key: Dict[str, object]) -> Dict[int, List[int]]:
    """{especies en común con el equipo de referencia más parecido: [victorias, partidas]} (sin random)."""
    from showdown.rl.synth_teams import max_overlap
    cache: Dict[str, int] = {}
    out: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
    for g in games:
        if g.get('rival') == 'random' or g.get('equipo') not in eval_by_key:
            continue
        k = g['equipo']
        if k not in cache:
            cache[k] = max_overlap(eval_by_key[k], refs)
        out[cache[k]][0] += g.get('res') == 'win'
        out[cache[k]][1] += 1
    return out


def main(argv=None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    out = Path(args[0]).expanduser()
    L = ['# Resumen de la octava cola: memorización de los equipos rivales', '']
    q = out / 'queue.log'
    if q.is_file():
        L += ['```', q.read_text(encoding='utf-8', errors='replace')[-2500:], '```', '']
    L += ['Victorias sin exploración, rivales deterministas, 100 partidas por rival y grupo. '
          '"Propios" = los equipos con que entrenó esa corrida; `eval` = 100 equipos que ninguna vio.', '',
          '## 1. Brecha entre equipos propios y no vistos', '',
          '| Corrida | Propios | `eval` | Brecha | solo (propios / eval) | joint | expectimax | random |',
          '|---|---|---|---|---|---|---|---|']
    loaded = {}
    for name, desc, _n, _ref in RUNS:
        own, ev = _load(out / 'diag' / f'{name}_own.json'), _load(out / 'diag' / f'{name}_eval.json')
        loaded[name] = (own, ev)
        if not (own and ev):
            L.append(f'| {desc} | sin datos | | | | | | |')
            continue
        (wo, no), (we, ne) = _wr(own['resultados'], FOES[1:]), _wr(ev['resultados'], FOES[1:])
        se = 100 * math.sqrt(wo * (1 - wo) / max(1, no) + we * (1 - we) / max(1, ne))
        L.append(f"| {desc} | {100 * wo:.1f} % | {100 * we:.1f} % | {100 * (we - wo):+.1f} (± {se:.1f}) | "
                 + ' | '.join(f"{_cell(own['resultados'], f)} / {_cell(ev['resultados'], f)}" for f in FOES[1:])
                 + f" | {_cell(own['resultados'], 'random')} / {_cell(ev['resultados'], 'random')} |")
    L += ['', 'Propios, `eval` y brecha: solo los 3 heurísticos (300 partidas por grupo); random va aparte '
          'porque no depende del equipo rival.', '']

    L += ['## 2. Validación reservada (`val`, 30 equipos no entrenados) durante el entrenamiento', '',
          '| Corrida | Paso | Media `val` | random | solo | joint | expectimax |', '|---|---|---|---|---|---|---|']
    for name, desc, _n, _ref in RUNS[1:]:
        ev = _evals(out / name)
        if not ev:
            L.append(f'| {desc} | sin datos | | | | | |')
            continue
        for r in ev:
            L.append(f"| {desc} | {r['step'] // 1000}k | {100 * r['mean_wr']:.1f} % | " + ' | '.join(_eval_cells(r)) + ' |')
    L.append('')

    L += ['## 3. ¿Gana más contra equipos `eval` parecidos a los de entrenamiento?', '',
          'Para cada partida contra un heurístico con un equipo `eval`: especies en común entre ese '
          'equipo y el equipo de entrenamiento más parecido (6 no ocurre: la separación es por clones).', '',
          '| Corrida | 2 o menos | 3 | 4 | 5 |', '|---|---|---|---|---|']
    try:
        os.environ.setdefault('PS_TEAM_SPLIT', 'eval')
        eval_by_key = _teams_by_key('eval')
        for name, desc, n_sub, ref in RUNS:
            ev = loaded.get(name, (None, None))[1]
            if not ev or not ev.get('partidas'):
                L.append(f'| {desc} | sin datos | | | |')
                continue
            refs = list(_teams_by_key(ref, n_sub).values())
            rows = similarity_rows(ev['partidas'], refs, eval_by_key)
            merged = {'≤2': [0, 0]}
            for k, (w, n) in rows.items():
                key = '≤2' if k <= 2 else str(k)
                merged.setdefault(key, [0, 0])
                merged[key][0] += w
                merged[key][1] += n
            cells = []
            for key in ('≤2', '3', '4', '5'):
                w, n = merged.get(key, [0, 0])
                cells.append(f'{100 * w / n:.0f} % (n={n})' if n else '—')
            L.append(f'| {desc} | ' + ' | '.join(cells) + ' |')
    except Exception as exc:  # el resumen no debe fallar por esta sección
        L.append(f'| error: {exc!r} | | | | |')
    L.append('')
    L += ['Lectura: con una semilla por configuración, diferencias de brecha de menos de ~8 puntos son '
          'ruido. La comparación principal (con y sin sintéticos) tiene 2 semillas.', '']
    (out / 'kickoff_summary8.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
