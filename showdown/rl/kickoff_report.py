"""Resumen del kick-off oficial de DDQN y PPO (decimosexta cola; protocolo §3c).

Lee ``<dir>/diag/{ddqn,ppo}_s{N}_{final_eval,best_eval,final_own}.json`` (policy_diag, 200
partidas por rival) y escribe ``<dir>/kickoff_summary16.md``:

- por método: media entre semillas de final.pt contra `eval` (resultado principal) con IC 95 %
  entre semillas (t de Student, la unidad de la afirmación es el método), y el IC por partidas
  agrupadas; best.pt como secundario; brecha con sus propios equipos;
- por rival, para cada método;
- DDQN frente a PPO (diferencia de medias entre semillas, IC de Welch).

Exploratorio respecto de la Conclusión formal (run_robust_e2e, BO3, protocolo §2–§3).

    python -m showdown.rl.kickoff_report ~/iapsvgc_runs/kickoff16
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')
FOES = ('random',) + HEUR
METHODS = ('ddqn', 'ppo')
SEEDS = range(5)
# t de Student al 97,5 % por grados de libertad (n − 1)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def _load(p: Path) -> Optional[Dict[str, Dict[str, int]]]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))['resultados']
    except Exception:
        return None


def rate(res: Optional[Dict], foes=HEUR) -> Optional[Tuple[float, int]]:
    if not res:
        return None
    w = sum(res.get(f, {}).get('win', 0) for f in foes)
    n = sum(sum(res.get(f, {}).values()) for f in foes)
    return (w / n, n) if n else None


def mean_ci(xs: List[float]) -> Tuple[float, float]:
    """Media e semiamplitud del IC 95 % entre semillas (t de Student)."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float('nan')
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T975.get(n - 1, 1.96) * sd / math.sqrt(n)


def _pct(x: float) -> str:
    return f'{100 * x:.1f} %'


def collect(d: Path) -> Dict[str, Dict[str, List]]:
    out: Dict[str, Dict[str, List]] = {}
    for m in METHODS:
        out[m] = {k: [] for k in ('final_eval', 'best_eval', 'final_own')}
        for s in SEEDS:
            for k in out[m]:
                r = _load(d / 'diag' / f'{m}_s{s}_{k}.json')
                if r:
                    out[m][k].append((s, r))
    return out


def build(d: Path) -> str:
    data = collect(d)
    L = ['# Kick-off oficial: DDQN y PPO (1M pasos × 5 semillas)', '',
         'Configuración final (ddqn_kickoff_checklist.md). Evaluación de cada semilla con policy_diag, '
         '200 partidas por rival, sin exploración, rivales deterministas, equipo fijo del agente. '
         'Cifras: victorias contra los 3 heurísticos (600 partidas por semilla). **Resultado principal: '
         'final.pt contra `eval`**; best.pt (elegido con `val`) es secundario (protocolo §3c.2). IC entre '
         'semillas: t de Student; IC por partidas: binomial agrupada.', '',
         '## 1. Resumen por método', '',
         '| Método | Semillas | final.pt `eval` (IC 95 % entre semillas) | IC por partidas | best.pt `eval` | '
         'final.pt propios | Brecha |', '|---|---|---|---|---|---|---|']
    means = {}
    for m in METHODS:
        fe = [rate(r) for _, r in data[m]['final_eval']]
        fe = [x for x in fe if x]
        if not fe:
            L.append(f'| {m.upper()} | 0 | sin datos | | | | |')
            continue
        mu, hw = mean_ci([x[0] for x in fe])
        means[m] = [x[0] for x in fe]
        w = sum(x[0] * x[1] for x in fe)
        n = sum(x[1] for x in fe)
        p = w / n
        hwp = 1.96 * math.sqrt(p * (1 - p) / n)
        be = [rate(r) for _, r in data[m]['best_eval']]
        be = [x[0] for x in be if x]
        ow = [rate(r) for _, r in data[m]['final_own']]
        ow = [x[0] for x in ow if x]
        bes = _pct(sum(be) / len(be)) if be else '—'
        ows = _pct(sum(ow) / len(ow)) if ow else '—'
        gap = f'{100 * (mu - sum(ow) / len(ow)):+.1f}' if ow else '—'
        L.append(f'| {m.upper()} | {len(fe)} | **{_pct(mu)}** (± {100 * hw:.1f}) | {_pct(p)} (± {100 * hwp:.1f}) | '
                 f'{bes} | {ows} | {gap} |')
    L += ['', '## 2. Por rival (final.pt contra `eval`, media entre semillas)', '',
          '| Método | random | greedy_solo | greedy_joint | expectimax |', '|---|---|---|---|---|']
    for m in METHODS:
        cells = []
        for f in FOES:
            xs = [rate(r, (f,)) for _, r in data[m]['final_eval']]
            xs = [x[0] for x in xs if x]
            if xs:
                mu, hw = mean_ci(xs)
                cells.append(f'{_pct(mu)} (± {100 * hw:.1f})' if len(xs) > 1 else _pct(mu))
            else:
                cells.append('—')
        L.append(f'| {m.upper()} | ' + ' | '.join(cells) + ' |')
    L += ['', '## 3. Por semilla', '', '| Método | Semilla | final.pt `eval` | best.pt `eval` | final.pt propios |',
          '|---|---|---|---|---|']
    for m in METHODS:
        idx = {k: dict(data[m][k]) for k in data[m]}
        for s in SEEDS:
            row = [rate(idx[k].get(s)) for k in ('final_eval', 'best_eval', 'final_own')]
            if any(row):
                L.append(f'| {m.upper()} | {s} | ' + ' | '.join(_pct(x[0]) if x else '—' for x in row) + ' |')
    arch_rows = []
    for m in METHODS:
        normal, arch = [], []
        for s in SEEDS:
            a = rate(_load(d / 'diag' / f'{m}_s{s}_final_arquetipo.json'), ('greedy_joint',))
            n = rate(_load(d / 'diag' / f'{m}_s{s}_final_eval.json'), ('greedy_joint',))
            if a and n:
                arch.append(a[0])
                normal.append(n[0])
        if arch:
            mn, _ = mean_ci(normal)
            ma, hwa = mean_ci(arch)
            arch_rows.append(f'| {m.upper()} | {len(arch)} | {_pct(mn)} | {_pct(ma)}'
                             f"{f' (± {100 * hwa:.1f})' if len(arch) > 1 else ''} | {100 * (ma - mn):+.1f} |")
    if arch_rows:
        L += ['', '## 3b. Contra greedy_joint en modo arquetipo (Tailwind / Trick Room; final.pt, `eval`)', '',
              '| Método | Semillas | joint normal | joint arquetipo (IC entre semillas) | Diferencia |',
              '|---|---|---|---|---|'] + arch_rows
    if len(means) == 2 and all(len(v) > 1 for v in means.values()):
        a, b = means['ppo'], means['ddqn']
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
        vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
        se = math.sqrt(va / len(a) + vb / len(b))
        dof = (va / len(a) + vb / len(b)) ** 2 / max(1e-12, (va / len(a)) ** 2 / (len(a) - 1)
                                                      + (vb / len(b)) ** 2 / (len(b) - 1))
        t = T975.get(max(1, min(8, int(dof))), 1.96)
        L += ['', '## 4. PPO frente a DDQN (final.pt contra `eval`)', '',
              f'Diferencia de medias entre semillas: **{100 * (ma - mb):+.1f} puntos** '
              f'(IC 95 % de Welch ± {100 * t * se:.1f}; {len(a)} y {len(b)} semillas).']
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    d = Path(args[0]).expanduser()
    (d / 'kickoff_summary16.md').write_text(build(d), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
