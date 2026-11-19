"""Tabla 2 × 2 (observación × preview) para PPO y DDQN (memorizacion_equipos.md §18.3).

Celdas (contra los 3 heurísticos, semillas 0 y 1, partidas agrupadas), para `val` y `eval`:
observación anterior / P5 × preview v2 / v2.1. Efecto del preview con cada observación y efecto de la
observación con cada preview, con IC 95 %.

    python scripts/analysis/ablacion_p5_preview.py ~/iapsvgc_runs/ablacion_p5 ~/iapsvgc_runs/p5full \
        ~/iapsvgc_runs/ppo_v1 ~/iapsvgc_runs/kickoff15 ~/iapsvgc_runs/ppo_v2
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from showdown.rl.kickoff_summary24 import _load, diff, fd, pct, pooled  # noqa: E402


def main(argv=None) -> int:
    a = [Path(x).expanduser() for x in (sys.argv[1:] if argv is None else argv)]
    out, q26, v1, q15, v2 = a
    D = out / 'diag'
    g = lambda d, n: _load(d / f'{n}.json')  # noqa: E731
    cells = {
        'PPO': {
            ('vieja', 'v2'): {'val': [g(v2 / 'diag', f'v1_s{s}_val') for s in (0, 1)],
                              'eval': [g(v1 / 'diag', f'ppo_s{s}_final_eval') for s in (0, 1)]},
            ('vieja', 'v2.1'): {sp: [g(D, f'A_ppo_v1_pv21_s{s}_{sp}') for s in (0, 1)] for sp in ('val', 'eval')},
            ('P5', 'v2'): {sp: [g(D, f'B_ppo_p5_pv2_s{s}_{sp}') for s in (0, 1)] for sp in ('val', 'eval')},
            ('P5', 'v2.1'): {sp: [g(q26 / 'diag', f'ppo_p5_s{s}_{sp}') for s in (0, 1)] for sp in ('val', 'eval')},
        },
        'DDQN': {
            ('vieja', 'v2'): {'val': [g(q15 / 'diag', f'R_wd03_s{s}_final_val') for s in (0, 1)],
                              'eval': [g(q26 / 'diag', f'ddqn_ref_s{s}_eval') for s in (0, 1)]},
            ('vieja', 'v2.1'): {sp: [g(D, f'C_ddqn_ref_pv21_s{s}_{sp}') for s in (0, 1)] for sp in ('val', 'eval')},
            ('P5', 'v2'): {sp: [g(D, f'D_ddqn_p5_pv2_s{s}_{sp}') for s in (0, 1)] for sp in ('val', 'eval')},
            ('P5', 'v2.1'): {sp: [g(q26 / 'diag', f'ddqn_p5_s{s}_{sp}') for s in (0, 1)] for sp in ('val', 'eval')},
        },
    }
    L = ['# Observación × preview (ablación, vigesimoséptima ejecución)', '']
    for agent, c in cells.items():
        for sp in ('val', 'eval'):
            p = {k: pooled(v[sp]) for k, v in c.items()}
            L += [f'## {agent}, `{sp}`', '',
                  '| Observación \ preview | v2 | v2.1 | Efecto del preview (v2.1 − v2) |', '|---|---|---|---|',
                  f"| anterior | {pct(p[('vieja', 'v2')])} | {pct(p[('vieja', 'v2.1')])} | "
                  f"{fd(diff(p[('vieja', 'v2.1')], p[('vieja', 'v2')]))} |",
                  f"| P5 | {pct(p[('P5', 'v2')])} | {pct(p[('P5', 'v2.1')])} | "
                  f"{fd(diff(p[('P5', 'v2.1')], p[('P5', 'v2')]))} |",
                  f"| Efecto de P5 (P5 − anterior) | {fd(diff(p[('P5', 'v2')], p[('vieja', 'v2')]))} | "
                  f"{fd(diff(p[('P5', 'v2.1')], p[('vieja', 'v2.1')]))} | |", '']
    md = '\n'.join(L) + '\n'
    (out / 'ablacion.md').write_text(md, encoding='utf-8')
    print(md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
