"""Resumen de la segunda cola previa al kick-off: preview v2, confirmación de lr y piloto de PPO.

    python -m showdown.rl.ddqn.kickoff_summary2 ~/iapsvgc_runs/kickoff2 ~/iapsvgc_runs/kickoff
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path
from typing import List, Optional

from showdown.rl.ddqn.kickoff_summary import _eval_cells, _evals, _last_learner


def _report_lines(rep: Path, keys) -> List[str]:
    if not rep.is_file():
        return ['Sin datos.']
    txt = rep.read_text(encoding='utf-8', errors='replace')
    return [l for l in txt.splitlines() if any(l.startswith(k) for k in keys)]


def section_preview(d: Path, q1: Path) -> List[str]:
    out = ['## A/B. Preview PPO v2 (90 formaciones canónicas, equipo propio fijo, 200 actualizaciones)', '',
           'Evaluación final contra el **preview heurístico** (BASE_WR = heurístico, mismo combate greedy_joint).', '',
           '| Variante | MEAN_WR | BASE_WR (heurístico) | DELTA_WR | Entropía final |', '|---|---|---|---|---|']
    for name in ('preview_v2_ent02', 'preview_v2_ent005'):
        rep = d / name / 'reporte.txt'
        log = d / f'{name}.txt'
        if not rep.is_file():
            out.append(f'| {name} | sin datos | | | |')
            continue
        txt = rep.read_text(encoding='utf-8', errors='replace')
        g = lambda k: (re.search(rf'{k}:\s*([+\-\d.]+(?: \(SE [\d.]+\))?)', txt) or [None, '?'])[1]
        ent = '?'
        if log.is_file():
            hs = re.findall(r'\bH=([\d.]+)', log.read_text(encoding='utf-8', errors='replace'))
            ent = hs[-1] if hs else '?'
        out.append(f"| {name} | {g('MEAN_WR')} | {g('BASE_WR')} | {g('DELTA_WR')} | {ent} |")
    chosen = d / 'preview_best.txt'
    if chosen.is_file():
        out += ['', f"Preview elegido para el piloto de PPO: `{chosen.read_text().strip()}`"]
    out += ['', 'Referencia (primera cola, sin v2): preview PPO 44,1 % frente a 44,9 % del fijo; '
            'probe: heurístico 50,8 %, fijo 45,3 %, al azar 42,9 %.', '']
    return out


def section_lr(d: Path, q1: Path) -> List[str]:
    out = ['## C. Confirmación de lr 1e-4 (pareada por semilla con la base lr 3e-4)', '',
           '| Semilla | Base lr 3e-4 | lr 1e-4 | Diferencia |', '|---|---|---|---|']
    base_dirs = {0: q1 / 'tune' / 'T0_base', 1: q1 / 'ablation' / 'struct_sym_s1', 2: q1 / 'ablation' / 'struct_sym_s2'}
    new_dirs = {0: q1 / 'tune' / 'T1_lr1e-4', 1: d / 'lr_confirm' / 'lr1e-4_s1', 2: d / 'lr_confirm' / 'lr1e-4_s2'}
    diffs = []
    for s in (0, 1, 2):
        b, n = _evals(base_dirs[s]), _evals(new_dirs[s])
        if not b or not n:
            fmt = lambda ev: f"{100 * ev[-1]['mean_wr']:.1f} %" if ev else '—'
            out.append(f'| {s} | {fmt(b)} | {fmt(n)} | — |')
            continue
        bv, nv = b[-1]['mean_wr'], n[-1]['mean_wr']
        diffs.append(nv - bv)
        out.append(f'| {s} | {100 * bv:.1f} % | {100 * nv:.1f} % | {100 * (nv - bv):+.1f} |')
    if diffs:
        out += ['', f'Diferencia media: {100 * statistics.mean(diffs):+.1f} puntos en {len(diffs)} semillas '
                f'({sum(x > 0 for x in diffs)} de {len(diffs)} a favor de lr 1e-4).']
    return out + ['']


def section_ppo(d: Path) -> List[str]:
    out = ['## D. Piloto de PPO (equipo fijo, 400k pasos: random 150k + solo 250k)', '']
    rep = d / 'ppo_pilot' / 'reporte.txt'
    out += ['```'] + _report_lines(rep, ('random:', 'greedy_joint:', 'expectimax_bayesian:', 'MEAN_WR',
                                         'steps_per_sec', 'elapsed', 'epsilon', 'env_steps')) + ['```', '']
    return out


def main(argv=None) -> int:
    args = list(argv or sys.argv[1:])
    d = Path(args[0]).expanduser()
    q1 = Path(args[1]).expanduser() if len(args) > 1 else d.parent / 'kickoff'
    lines = ['# Resumen de la segunda cola previa al kick-off', '']
    q = d / 'queue.log'
    if q.is_file():
        lines += ['```', q.read_text(encoding='utf-8', errors='replace')[-3000:], '```', '']
    for fn, a in ((section_preview, (d, q1)), (section_lr, (d, q1)), (section_ppo, (d,))):
        try:
            lines += fn(*a)
        except Exception as exc:
            lines += [f'Error al resumir ({fn.__name__}): {exc!r}', '']
    (d / 'kickoff_summary2.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
