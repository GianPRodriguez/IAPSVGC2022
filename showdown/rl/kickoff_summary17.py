"""Resumen de la decimoséptima cola: duración (2M frente a 1M) y simetría en DDQN sobre REG.

Elección con `val` (200 partidas por rival, final.pt, 3 heurísticos, 2 semillas agrupadas);
`eval` y los equipos propios son solo registro. Reglas fijadas en scripts/server/kickoff_queue17.sh.

    python -m showdown.rl.kickoff_summary17 ~/iapsvgc_runs/kickoff17 ~/iapsvgc_runs/kickoff15
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')
THRESHOLD = 0.04
# opciones de REG (chosen_reg.txt de la decimoquinta cola) → nombre de sus corridas allí
REG_NAMES = {'--weight-decay 0.01 --dropout 0.1': 'REG', '--weight-decay 0.03 --dropout 0.1': 'R_wd03',
             '--weight-decay 0.01 --dropout 0.2': 'R_do2', '--weight-decay 0.03 --dropout 0.2': 'R_both'}


def wins(p: Path) -> Optional[Tuple[int, int]]:
    try:
        r = json.loads(p.read_text(encoding='utf-8'))['resultados']
    except Exception:
        return None
    return (sum(r.get(f, {}).get('win', 0) for f in HEUR), sum(sum(r.get(f, {}).values()) for f in HEUR))


def pooled(paths: List[Path]) -> Optional[Tuple[float, int]]:
    xs = [x for x in (wins(p) for p in paths) if x]
    if not xs:
        return None
    n = sum(x[1] for x in xs)
    return (sum(x[0] for x in xs) / n, n) if n else None


def _pct(x: Optional[Tuple[float, int]]) -> str:
    return '—' if not x else f'{100 * x[0]:.1f} %'


def compare(new, ref) -> Tuple[str, Optional[bool]]:
    if not new or not ref:
        return 'sin datos', None
    diff = new[0] - ref[0]
    se = math.sqrt(new[0] * (1 - new[0]) / new[1] + ref[0] * (1 - ref[0]) / ref[1])
    return f'{100 * diff:+.1f} (± {100 * se:.1f})', diff > THRESHOLD


def build(d: Path, q15: Path) -> str:
    try:
        flags = (q15 / 'chosen_reg.txt').read_text(encoding='utf-8').strip()
    except OSError:
        flags = '--weight-decay 0.01 --dropout 0.1'
    reg = REG_NAMES.get(flags, 'REG')
    ref_ddqn = pooled([q15 / 'diag' / f'{reg}_s{s}_final_val.json' for s in (0, 1)])
    ref_ppo = pooled([d / 'diag' / f'PPO1M_s{s}_final_val.json' for s in (0, 1)])
    rows = [('A', 'DDQN 2M frente a DDQN 1M (REG elegida)', 'DDQN2M', ref_ddqn),
            ('A', 'PPO 2M frente a PPO 1M (con simetría)', 'PPO2M', ref_ppo),
            ('D', 'DDQN 1M con simetría frente a sin ella (REG elegida)', 'DSYM', ref_ddqn)]
    L = ['# Resumen de la decimoséptima cola: duración y simetría en DDQN', '',
         f'REG de DDQN: `{flags}` ({reg}). Elección con `val` (30 equipos, 200 partidas por rival, '
         'final.pt, 3 heurísticos, 2 semillas agrupadas); se recomienda el cambio si supera a la '
         f'referencia por más de {100 * THRESHOLD:.0f} puntos. `eval` y propios: solo registro.', '',
         '| Pregunta | Comparación | Nuevo `val` | Referencia `val` | Diferencia | ¿Se recomienda? | Nuevo `eval` (registro) | Nuevo propios (registro) |',
         '|---|---|---|---|---|---|---|---|']
    for q, desc, name, ref in rows:
        new = pooled([d / 'diag' / f'{name}_s{s}_final_val.json' for s in (0, 1)])
        ev = pooled([d / 'diag' / f'{name}_s{s}_final_eval.json' for s in (0, 1)])
        ow = pooled([d / 'diag' / f'{name}_s{s}_final_own.json' for s in (0, 1)])
        txt, ok = compare(new, ref)
        verdict = '—' if ok is None else ('sí' if ok else 'no')
        L.append(f'| {q} | {desc} | {_pct(new)} | {_pct(ref)} | {txt} | {verdict} | {_pct(ev)} | {_pct(ow)} |')
    L += ['', 'Nota: PPO está cerca del techo en `val` (~93–95 %); una diferencia pequeña en `val` no '
          'descarta mejoras contra rivales más fuertes.', '', '## Curvas de `val` durante el entrenamiento', '',
          '| Corrida | Pasos (k) → media `val` (incluye random) |', '|---|---|']
    for name in ('DDQN2M_s0', 'DDQN2M_s1', 'PPO2M_s0', 'PPO2M_s1', 'DSYM_s0', 'DSYM_s1'):
        mf = d / name / 'train_metrics.jsonl'
        if not mf.is_file():
            continue
        pts = []
        for line in mf.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('kind') == 'eval':
                pts.append(f"{r['step'] // 1000}: {100 * r['mean_wr']:.1f}")
        L.append(f"| {name} | {', '.join(pts)} |")
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    d, q15 = Path(args[0]).expanduser(), Path(args[1]).expanduser()
    (d / 'kickoff_summary17.md').write_text(build(d, q15), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
