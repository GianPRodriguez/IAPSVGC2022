"""Estadística confirmatoria para afirmar "A > B" (ver docs/protocolo_evaluacion.md).

Lee los logs por semilla de ``run_robust_e2e``, donde cada serie BO3 es una línea
``[k/n] WIN|LOSS|DRAW ... | equipoA vs equipoB``. Ofrece tres análisis:

- ``binom``: tasa de victorias de A contra 50 %, con prueba binomial exacta bilateral e
  IC 95 % por *cluster bootstrap*. El clúster es (semilla, par de equipos sin orden): las
  dos BO3 cruzadas de un mismo par no son independientes.
- ``mcnemar``: comparación pareada de dos configuraciones contra el mismo rival y con las
  mismas semillas. Empareja series por (semilla, equipos), cuenta las discordantes y aplica
  una prueba binomial exacta sobre ellas.
- ``holm``: corrección de Holm-Bonferroni para una familia de p-valores.

    python -m showdown.inference_sim.confirm_stats binom runs/E_J_92*.log
    python -m showdown.inference_sim.confirm_stats mcnemar --a runs/X_*.log --b runs/Y_*.log
    python -m showdown.inference_sim.confirm_stats holm 0.003 0.02 0.04
"""
from __future__ import annotations

import argparse
import math
import random
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

_LINE = re.compile(r'^\[\d+/\d+\]\s+(WIN|LOSS|DRAW)\b.*\|\s*(.+?)\s+vs\s+(.+?)\s*$')
_SEED = re.compile(r'seed=(\d+)')

Series = Tuple[str, str, str, float]  # (semilla, equipo_a, equipo_b, resultado 1/0/0.5)


def parse_series(paths: Sequence[str]) -> List[Series]:
    """Series BO3 de uno o más logs (la semilla sale de la línea BATCH o del nombre)."""
    out: List[Series] = []
    for path in paths:
        seed = None
        m = re.search(r'_(\d{4,})\.log$', str(path))
        if m:
            seed = m.group(1)
        with open(path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if seed is None and line.startswith('BATCH'):
                    ms = _SEED.search(line)
                    if ms:
                        seed = ms.group(1)
                mm = _LINE.match(line.rstrip('\n'))
                if not mm:
                    continue
                res = {'WIN': 1.0, 'LOSS': 0.0, 'DRAW': 0.5}[mm.group(1)]
                out.append((seed or '?', mm.group(2).strip(), mm.group(3).strip(), res))
    return out


def _log_binom_pmf(k: int, n: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2)


def binom_two_sided_p(k: int, n: int) -> float:
    """p exacto bilateral de H0: p = 0,5 (suma de resultados tan o más extremos)."""
    if n <= 0:
        return 1.0
    obs = _log_binom_pmf(k, n)
    total = 0.0
    for i in range(n + 1):
        lp = _log_binom_pmf(i, n)
        if lp <= obs + 1e-12:
            total += math.exp(lp)
    return min(1.0, total)


def cluster_bootstrap_ci(series: Sequence[Series], *, reps: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> Tuple[float, float]:
    """IC de la tasa de victorias remuestreando clústeres (semilla, par de equipos)."""
    clusters: Dict[Tuple[str, frozenset], List[float]] = defaultdict(list)
    for s, a, b, r in series:
        clusters[(s, frozenset((a, b)))].append(r)
    groups = list(clusters.values())
    if not groups:
        return (float('nan'), float('nan'))
    rng = random.Random(seed)
    stats = []
    for _ in range(reps):
        tot = n = 0.0
        for _ in range(len(groups)):
            g = groups[rng.randrange(len(groups))]
            tot += sum(g)
            n += len(g)
        stats.append(tot / n)
    stats.sort()
    lo = stats[int((alpha / 2) * reps)]
    hi = stats[min(reps - 1, int((1 - alpha / 2) * reps))]
    return lo, hi


def binom_report(series: Sequence[Series]) -> Dict[str, float]:
    n = len(series)
    wins = sum(r for *_, r in series)
    k = int(round(wins))
    lo, hi = cluster_bootstrap_ci(series)
    by_seed: Dict[str, List[float]] = defaultdict(list)
    for s, _a, _b, r in series:
        by_seed[s].append(r)
    above = sum(1 for v in by_seed.values() if sum(v) / len(v) > 0.5)
    return {
        'n': n, 'wins': wins, 'rate': wins / n if n else float('nan'),
        'p': binom_two_sided_p(k, n), 'ci_lo': lo, 'ci_hi': hi,
        'seeds': len(by_seed), 'seeds_above_50': above,
    }


def mcnemar(a: Sequence[Series], b: Sequence[Series]) -> Dict[str, float]:
    """Empareja series por (semilla, equipo_a, equipo_b); discordantes → binomial exacta."""
    def index(xs):
        d: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
        for s, ta, tb, r in xs:
            d[(s, ta, tb)].append(r)
        return d
    ia, ib = index(a), index(b)
    n10 = n01 = matched = 0
    for key in ia.keys() & ib.keys():
        for ra, rb in zip(ia[key], ib[key]):
            matched += 1
            if ra > rb:
                n10 += 1
            elif rb > ra:
                n01 += 1
    disc = n10 + n01
    return {
        'matched': matched, 'a_better': n10, 'b_better': n01,
        'diff_pts': 100.0 * (n10 - n01) / matched if matched else float('nan'),
        'p': binom_two_sided_p(n10, disc) if disc else 1.0,
    }


def holm(pvals: Sequence[float], alpha: float = 0.05) -> List[Tuple[float, float, bool]]:
    """Holm-Bonferroni: (p, p ajustado, rechaza H0) en el orden de entrada."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return [(pvals[i], adj[i], adj[i] <= alpha) for i in range(m)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    pb = sub.add_parser('binom')
    pb.add_argument('logs', nargs='+')
    pm = sub.add_parser('mcnemar')
    pm.add_argument('--a', nargs='+', required=True)
    pm.add_argument('--b', nargs='+', required=True)
    ph = sub.add_parser('holm')
    ph.add_argument('p', nargs='+', type=float)
    args = ap.parse_args(argv)
    if args.cmd == 'binom':
        r = binom_report(parse_series(args.logs))
        print(f"series={r['n']} victorias={r['wins']:.1f} tasa={100 * r['rate']:.2f}% "
              f"IC95_cluster=[{100 * r['ci_lo']:.2f}, {100 * r['ci_hi']:.2f}] p={r['p']:.4g} "
              f"semillas>50%={r['seeds_above_50']}/{r['seeds']}")
    elif args.cmd == 'mcnemar':
        r = mcnemar(parse_series(args.a), parse_series(args.b))
        print(f"emparejadas={r['matched']} A_mejor={r['a_better']} B_mejor={r['b_better']} "
              f"diferencia={r['diff_pts']:+.2f} pts p={r['p']:.4g}")
    else:
        for p, a, rej in holm(args.p):
            print(f'p={p:.4g} p_holm={a:.4g} {"rechaza H0" if rej else "no rechaza"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
