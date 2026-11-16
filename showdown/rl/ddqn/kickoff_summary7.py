"""Resumen de la séptima cola: teorías sobre ataques al aliado / cambios voluntarios y preview.

    python -m showdown.rl.ddqn.kickoff_summary7 ~/iapsvgc_runs/kickoff7 ~/iapsvgc_runs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

from showdown.rl.ddqn.kickoff_summary import _eval_cells, _evals

FOES = ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')


def _diag_row(name: str, p: Path) -> str:
    if not p.is_file():
        return f'| {name} | sin datos | | | | |'
    d = json.loads(p.read_text(encoding='utf-8'))
    t = d['total']
    n = max(1, int(t.get('decisiones', 0)))
    res = d.get('resultados', {})
    w = sum(res.get(f, {}).get('win', 0) for f in FOES)
    g = sum(sum(res.get(f, {}).values()) for f in FOES) or 1
    return (f"| {name} | {100 * t.get('ataque_aliado', 0) / n:.1f} % | {100 * t.get('cambio_voluntario', 0) / n:.1f} % | "
            f"{100 * t.get('protect', 0) / n:.1f} % | {100 * t.get('pass_con_opciones', 0) / n:.1f} % | {100 * w / g:.0f} % |")


HEAD = ['| Checkpoint | Ataque al aliado | Cambio voluntario | Protect | Pass con opciones | Victorias (4 rivales) |',
        '|---|---|---|---|---|---|']


def main(argv=None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    out, r = Path(args[0]).expanduser(), Path(args[1]).expanduser()
    dd = out / 'diag'
    L = ['# Resumen de la séptima cola: ataques al aliado, cambios voluntarios y preview', '']
    q = out / 'queue.log'
    if q.is_file():
        L += ['```', q.read_text(encoding='utf-8', errors='replace')[-2500:], '```', '']
    L += ['## D1. ¿Bajan con el entrenamiento?', ''] + HEAD
    for n in ('D1_T1_200k', 'D1_T1_400k', 'D1_mini_250000', 'D1_mini_500000', 'D1_mini_750000', 'D1_mini_1000000'):
        L.append(_diag_row(n, dd / f'{n}.json'))
    L += ['', '## D2. ¿Tiene que ver la simetría? (checkpoints de la ablación, 400k pasos)', ''] + HEAD
    for s in (0, 1, 2):
        L.append(_diag_row(f'con simetría s{s}', dd / f'D2_sym_s{s}.json'))
        L.append(_diag_row(f'sin simetría s{s}', dd / f'D2_nosym_s{s}.json'))
    L += ['', '## D3. Preview v1 frente a v2 con el mismo DDQN (mejor checkpoint de lr 1e-4)', ''] + HEAD
    L += [_diag_row('preview v1', dd / 'D3_preview_v1.json'), _diag_row('preview v2', dd / 'D3_preview_v2.json')]
    L += ['', '## D4. Preview v2 con expectimax en el combate (pareado contra el preview heurístico)', '']
    p = out / 'preview_check_expectimax' / 'preview_check.json'
    if p.is_file():
        d = json.loads(p.read_text(encoding='utf-8'))
        L += ['| Rivales | Preview v2 | Heurístico | Diferencia (± EE) |', '|---|---|---|---|']
        for key, name in (('split_train', '`train`'), ('split_eval', '`eval`'), ('formacion_fija', 'formación fija')):
            v = d.get(key)
            if v:
                L.append(f"| {name} | {100 * v['mean_wr']:.1f} % | {100 * v['base_wr']:.1f} % | "
                         f"{100 * v['delta_wr']:+.1f} (± {100 * v['delta_se']:.1f}) |")
    else:
        L.append('Sin datos.')
    L += ['', '## P5 / P6. Pilotos A/B (configuración de T0_base: lr 3e-4, preview v1, 400k pasos, semilla 0)', ''] + HEAD
    L += [_diag_row('T0_base (referencia)', dd / 'T0_base_final.json'),
          _diag_row('P5 recompensa simétrica', dd / 'P5_final.json'),
          _diag_row('P6 máscara de daño al aliado', dd / 'P6_final.json')]
    L += ['', '| Corrida | Validación final (media) | random | solo | joint | expectimax |', '|---|---|---|---|---|---|']
    for name, ck in (('T0_base', r / 'kickoff' / 'tune' / 'T0_base'), ('P5', out / 'P5_reward_hp_self'),
                     ('P6', out / 'P6_mask_ally_damage')):
        ev = _evals(ck)
        L.append(f"| {name} | {100 * ev[-1]['mean_wr']:.1f} % | " + ' | '.join(_eval_cells(ev[-1])) + ' |'
                 if ev else f'| {name} | sin datos | | | | |')
    (out / 'kickoff_summary7.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
