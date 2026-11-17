"""Resumen de la decimoctava cola: valores Q por hueco en DDQN (memorizacion_equipos.md §10).

Elección con `val` (200 partidas por rival, final.pt, 3 heurísticos, 2 semillas agrupadas) frente
a la cabeza conjunta con la REG elegida (decimoquinta cola); `eval` y propios son solo registro.
Regla: se adopta la mejor variante factorizada si supera a la conjunta por más de 4 puntos.

    python -m showdown.rl.kickoff_summary18 ~/iapsvgc_runs/kickoff18 ~/iapsvgc_runs/kickoff15
"""
from __future__ import annotations

import sys
from pathlib import Path

from showdown.rl.kickoff_summary17 import REG_NAMES, THRESHOLD, _pct, compare, pooled

VARIANTS = (('QF0', 'factorizada aditiva (rango 0)'), ('QF8', 'factorizada con interacción de rango 8'))


def build(d: Path, q15: Path) -> str:
    try:
        flags = (q15 / 'chosen_reg.txt').read_text(encoding='utf-8').strip()
    except OSError:
        flags = '--weight-decay 0.01 --dropout 0.1'
    reg = REG_NAMES.get(flags, 'REG')
    ref = pooled([q15 / 'diag' / f'{reg}_s{s}_final_val.json' for s in (0, 1)])
    L = ['# Resumen de la decimoctava cola: valores Q por hueco en DDQN', '',
         f'DDQN 1M, configuración por defecto + REG `{flags}`. Referencia: cabeza conjunta ({reg}, '
         'decimoquinta cola). Elección con `val` (200 partidas por rival, final.pt, 3 heurísticos, 2 '
         f'semillas agrupadas): se adopta la mejor variante si supera a la referencia por más de '
         f'{100 * THRESHOLD:.0f} puntos. `eval` y propios: solo registro.', '',
         '| Variante | `val` | Referencia `val` | Diferencia | Supera el umbral | `eval` (registro) | Propios (registro) |',
         '|---|---|---|---|---|---|---|']
    best, best_diff = None, None
    for name, desc in VARIANTS:
        new = pooled([d / 'diag' / f'{name}_s{s}_final_val.json' for s in (0, 1)])
        ev = pooled([d / 'diag' / f'{name}_s{s}_final_eval.json' for s in (0, 1)])
        ow = pooled([d / 'diag' / f'{name}_s{s}_final_own.json' for s in (0, 1)])
        txt, ok = compare(new, ref)
        if new and ref and (best_diff is None or new[0] - ref[0] > best_diff):
            best, best_diff = desc, new[0] - ref[0]
        L.append(f"| {desc} | {_pct(new)} | {_pct(ref)} | {txt} | {'—' if ok is None else ('sí' if ok else 'no')} | "
                 f'{_pct(ev)} | {_pct(ow)} |')
    if best is None:
        L += ['', '**Elección:** sin datos suficientes; se mantiene la cabeza conjunta.']
    elif best_diff > THRESHOLD:
        L += ['', f'**Elección:** {best} ({100 * best_diff:+.1f} puntos en `val`). Decisión final del usuario.']
    else:
        L += ['', f'**Elección:** se mantiene la cabeza conjunta (la mejor variante, {best}, queda a '
              f'{100 * best_diff:+.1f} puntos en `val`).']
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    d, q15 = Path(args[0]).expanduser(), Path(args[1]).expanduser()
    (d / 'kickoff_summary18.md').write_text(build(d, q15), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
