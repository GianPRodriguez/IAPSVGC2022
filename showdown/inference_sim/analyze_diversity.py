"""Resume la traza de diversidad (``PS_DIVERSITY_LOG``) por agente.

    PS_DIVERSITY_LOG=logs/div python -m showdown.inference_sim.run_robust_e2e ...
    python -m showdown.inference_sim.analyze_diversity logs/div

Por agente: decisiones, % igual a greedy_solo, % igual a greedy_joint (solo expectimax),
y con el muestreo de casi-empates activo, % de decisiones con más de un candidato y % en
que se jugó un par distinto del determinista.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from typing import Dict, List


def load(prefix: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(glob.glob(prefix + '*')):
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def summarize(rows: List[dict]) -> Dict[str, Dict[str, float]]:
    acc: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        a = acc[str(r.get('agent', '?'))]
        a['n'] += 1
        for key in ('same_solo', 'same_joint'):
            v = r.get(key)
            if v is not None:
                a[key + '_n'] += 1
                a[key] += 1.0 if v else 0.0
        if int(r.get('tie_cands') or 1) > 1:
            a['tie_multi'] += 1
            a['tie_cands_sum'] += int(r['tie_cands'])
        if r.get('tie_changed'):
            a['tie_changed'] += 1
    out: Dict[str, Dict[str, float]] = {}
    for agent, a in acc.items():
        n = a['n'] or 1.0
        out[agent] = {
            'n': int(a['n']),
            'same_solo': a['same_solo'] / a['same_solo_n'] if a['same_solo_n'] else float('nan'),
            'same_joint': a['same_joint'] / a['same_joint_n'] if a['same_joint_n'] else float('nan'),
            'tie_multi': a['tie_multi'] / n,
            'tie_cands_mean': a['tie_cands_sum'] / a['tie_multi'] if a['tie_multi'] else 1.0,
            'tie_changed': a['tie_changed'] / n,
        }
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    for agent, s in sorted(summarize(load(argv[0])).items()):
        print(f"{agent}: decisiones={s['n']} igual_a_solo={100 * s['same_solo']:.1f}% "
              f"igual_a_joint={100 * s['same_joint']:.1f}% "
              f"casi_empates>1={100 * s['tie_multi']:.1f}% (media {s['tie_cands_mean']:.1f} cand.) "
              f"par_distinto_del_determinista={100 * s['tie_changed']:.1f}%")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
