"""Resumen de la tercera cola: diagnóstico de la política, lr 1e-4 + lote 512, mini kick-off y carga.

    python -m showdown.rl.ddqn.kickoff_summary3 ~/iapsvgc_runs/kickoff3 ~/iapsvgc_runs/kickoff ~/iapsvgc_runs/kickoff2
    python -m showdown.rl.ddqn.kickoff_summary3 --choose ...   # elige lr/lote/preview del mini kick-off
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from showdown.rl.ddqn.kickoff_summary import _eval_cells, _evals, _last_learner


def _final(ck: Path) -> Optional[float]:
    ev = _evals(ck)
    return float(ev[-1]['mean_wr']) if ev else None


def choose(out: Path, q1: Path, q2: Path) -> str:
    """Reglas fijadas antes de ver los resultados:

    - lr 1e-4 si su diferencia media con la base (semillas pareadas 0–2) es > 0; si no, 3e-4.
    - lote 512 solo si "lr 1e-4 + lote 512" supera a "lr 1e-4" en la semilla 0; si no, 256.
    - preview: el mejor v2 de la cola 2 si existe; si no, el de la cola 1.
    """
    pairs = [(q1 / 'tune' / 'T0_base', q1 / 'tune' / 'T1_lr1e-4'),
             (q1 / 'ablation' / 'struct_sym_s1', q2 / 'lr_confirm' / 'lr1e-4_s1'),
             (q1 / 'ablation' / 'struct_sym_s2', q2 / 'lr_confirm' / 'lr1e-4_s2')]
    diffs = [(_final(n) - _final(b)) for b, n in pairs if _final(b) is not None and _final(n) is not None]
    lr = '1e-4' if diffs and statistics.mean(diffs) > 0 else '3e-4'
    combo, t1 = _final(out / 'combo_lr1e-4_b512'), _final(q1 / 'tune' / 'T1_lr1e-4')
    batch = '512' if (lr == '1e-4' and combo is not None and t1 is not None and combo > t1) else '256'
    pv = ''
    best = q2 / 'preview_best.txt'
    if best.is_file() and Path(best.read_text().strip()).is_file():
        pv = best.read_text().strip()
    elif (q1 / 'preview_used.txt').is_file() and Path((q1 / 'preview_used.txt').read_text().strip()).is_file():
        pv = (q1 / 'preview_used.txt').read_text().strip()
    return f'lr={lr} batch={batch} preview={pv}'


def section_diag(out: Path) -> List[str]:
    from showdown.rl.ddqn.policy_diag import format_table
    p = out / 'policy_diag.json'
    lines = ['## 4. Desglose de las jugadas sin exploración (mejor checkpoint de lr 1e-4)', '']
    if not p.is_file():
        return lines + ['Sin datos.', '']
    rep = json.loads(p.read_text(encoding='utf-8'))
    return lines + [f"Checkpoint: `{rep['checkpoint']}`; {rep['games_per_foe']} partidas por rival.", '',
                    format_table(rep), '',
                    'Referencia: greedy_joint nunca cambia voluntariamente; expectimax v1.9 cambia ~0,1 veces '
                    'por partida.', '']


def section_combo(out: Path, q1: Path) -> List[str]:
    lines = ['## 2. lr 1e-4 + lote 512 (semilla 0, 400k pasos)', '',
             '| Config | Media final | random | solo | joint | expectimax |', '|---|---|---|---|---|---|']
    for name, ck in (('base lr 3e-4', q1 / 'tune' / 'T0_base'), ('lr 1e-4', q1 / 'tune' / 'T1_lr1e-4'),
                     ('lote 512', q1 / 'tune' / 'T7_batch512'), ('lr 1e-4 + lote 512', out / 'combo_lr1e-4_b512')):
        ev = _evals(ck)
        if not ev:
            lines.append(f'| {name} | sin datos | | | | |')
            continue
        lines.append(f"| {name} | {100 * ev[-1]['mean_wr']:.1f} % | " + ' | '.join(_eval_cells(ev[-1])) + ' |')
    return lines + ['']


def section_mini(out: Path) -> List[str]:
    lines = ['## 1. Mini kick-off: 1M pasos con el currículo completo (random → solo → joint → expectimax)', '']
    ch = out / 'mini_choice.txt'
    if ch.is_file():
        lines += [f"Configuración (reglas fijadas de antemano): `{ch.read_text().strip()}`", '']
    ev = _evals(out / 'mini_kickoff')
    if not ev:
        return lines + ['Sin datos.', '']
    lines += ['| Paso | Media | random | solo | joint | expectimax |', '|---|---|---|---|---|---|']
    for e in ev:
        lines.append(f"| {e['step'] // 1000}k | {100 * e['mean_wr']:.1f} % | " + ' | '.join(_eval_cells(e)) + ' |')
    ln = _last_learner(out / 'mini_kickoff')
    lines += ['', f"Q medio final {ln.get('mean_q', float('nan')):.2f}, error TD {ln.get('td_error', float('nan')):.3f}, "
              f"ε final {ln.get('eps', float('nan')):.2f}.", '']
    return lines


def section_load(out: Path) -> List[str]:
    p = out / 'load.csv'
    lines = ['## 5. Carga del servidor durante la cola', '']
    if not p.is_file():
        return lines + ['Sin datos.', '']
    acc = defaultdict(list)
    with open(p, encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            try:
                acc[row['etapa']].append((float(row['cpu_ocupada']), float(row['cpu_otros_usuarios'])))
            except (ValueError, KeyError):
                pass
    lines += ['| Etapa | Muestras | CPU ocupada media | CPU de otros usuarios (media / máx., % de un núcleo) |',
              '|---|---|---|---|']
    for st, vals in acc.items():
        lines.append(f"| {st} | {len(vals)} | {statistics.mean(v[0] for v in vals):.0f} % | "
                     f"{statistics.mean(v[1] for v in vals):.0f} / {max(v[1] for v in vals):.0f} |")
    return lines + ['']


def main(argv=None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args and args[0] == '--choose':
        out, q1, q2 = (Path(a).expanduser() for a in args[1:4])
        print(choose(out, q1, q2))
        return 0
    out, q1, q2 = (Path(a).expanduser() for a in args[:3])
    lines = ['# Resumen de la tercera cola previa al kick-off', '']
    q = out / 'queue.log'
    if q.is_file():
        lines += ['```', q.read_text(encoding='utf-8', errors='replace')[-3000:], '```', '']
    for fn, a in ((section_diag, (out,)), (section_combo, (out, q1)), (section_mini, (out,)),
                  (section_load, (out,))):
        try:
            lines += fn(*a)
        except Exception as exc:
            lines += [f'Error al resumir ({fn.__name__}): {exc!r}', '']
    (out / 'kickoff_summary3.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
