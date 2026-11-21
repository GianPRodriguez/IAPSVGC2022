"""¿El efecto de v6 (terreno) frente a v5 se concentra en series con creadores de terreno?

Análisis por subgrupos de series emparejadas (mismas semillas y equipos) entre dos colas de
exploración de G. Clasifica cada serie según si alguno de los dos equipos lleva una habilidad
que crea terreno (Grassy / Psychic / Electric / Misty Surge, Hadron Engine) y hace un McNemar
por subgrupo. Si el terreno importara, la diferencia debería aparecer en el subgrupo "con
terreno" y ser ~0 en el resto.

    python scripts/analysis/terrain_subgroup.py ~/iapsvgc_runs/mcts_g_v6/raw/v6_n200_s91*.txt \
        -- ~/iapsvgc_runs/mcts_g_v5/raw/v5_n200_s91*.txt
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from showdown.inference_sim.confirm_stats import binom_two_sided_p, parse_series  # noqa: E402

TERRAIN_ABIL = {'grassysurge', 'psychicsurge', 'electricsurge', 'mistysurge', 'hadronengine'}


def _n(x) -> str:
    return re.sub(r'[^a-z0-9]', '', str(x).lower())


def load_teams():
    teams = json.loads((REPO / 'showdown' / 'inference_sim' / 'metavgc_regmc_teams.json').read_text(encoding='utf-8'))
    return [(t['name'], any(_n(m.get('ability')) in TERRAIN_ABIL for m in t.get('mons', []))) for t in teams]


def terrain_of(name: str, teams) -> str:
    """'yes' / 'no' / 'ambiguous' según los equipos cuyo nombre empieza por ``name`` (logs truncados)."""
    name = name.strip()
    if not name:
        return 'ambiguous'
    hits = {flag for full, flag in teams if full.startswith(name) or name.startswith(full)}
    if len(hits) == 1:
        return 'yes' if hits.pop() else 'no'
    return 'ambiguous'


def mcnemar(pairs):
    a = sum(1 for x, y in pairs if x > y)
    b = sum(1 for x, y in pairs if y > x)
    n = a + b
    p = binom_two_sided_p(a, n) if n else 1.0
    diff = 100.0 * (sum(x for x, _ in pairs) - sum(y for _, y in pairs)) / max(1, len(pairs))
    return len(pairs), a, b, diff, p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' not in argv:
        print(__doc__)
        return 2
    k = argv.index('--')
    a_logs, b_logs = argv[:k], argv[k + 1:]
    teams = load_teams()
    sa = {(s, x, y): r for s, x, y, r in parse_series(a_logs)}
    sb = {(s, x, y): r for s, x, y, r in parse_series(b_logs)}
    groups = {'con terreno': [], 'sin terreno': [], 'ambiguo': []}
    for key in sorted(set(sa) & set(sb)):
        _s, ta, tb = key
        fa, fb = terrain_of(ta, teams), terrain_of(tb, teams)
        if 'yes' in (fa, fb):
            g = 'con terreno'
        elif fa == 'no' and fb == 'no':
            g = 'sin terreno'
        else:
            g = 'ambiguo'
        groups[g].append((sa[key], sb[key]))
    allp = [p for v in groups.values() for p in v]
    print('Subgrupo | series emparejadas | A mejor | B mejor | diferencia (pts) | p (McNemar exacta)')
    for name, pairs in list(groups.items()) + [('total', allp)]:
        n, a, b, diff, p = mcnemar(pairs)
        print(f'{name} | {n} | {a} | {b} | {diff:+.2f} | {p:.4g}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
