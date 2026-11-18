"""Resumen de la vigesimosegunda cola: preview entrenado con su propio combate (§15).

    python -m showdown.rl.kickoff_summary22 ~/iapsvgc_runs/kickoff22
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from showdown.rl.kickoff_summary17 import THRESHOLD, _pct, compare, pooled

AGENTS = (('ddqn', 'DDQN (REG 0,03)'), ('ppo', 'PPO (simetría)'))


def build(d: Path) -> str:
    D = d / 'diag'
    L = ['# Resumen de la vigesimosegunda cola: preview entrenado con su propio combate', '',
         'Preview nuevo por agente: PPO de preview entrenado con el combate congelado de ese agente '
         '(final.pt semilla 0) en nuestro lado y greedy_joint en el rival, solo con los 170 equipos de '
         '`fit`. Se compara con el preview v2 (entrenado con greedy_joint, 200 equipos de `train`) usando '
         'el mismo combate (final.pt, semillas 0 y 1), tu equipo, 200 partidas por rival. Elección con '
         f'`val` (ahora desconocido también para el preview nuevo): se adopta si supera a v2 por más de '
         f'{100 * THRESHOLD:.0f} puntos. `eval`: registro.', '',
         '| Agente | `val` v2 | `val` nuevo | Diferencia | ¿Se adopta? | `eval` v2 | `eval` nuevo | Diferencia `eval` |',
         '|---|---|---|---|---|---|---|---|']
    for a, title in AGENTS:
        v2v = pooled([D / f'{a}_s{s}_v2_val.json' for s in (0, 1)])
        nwv = pooled([D / f'{a}_s{s}_nuevo_val.json' for s in (0, 1)])
        v2e = pooled([d.parent / 'kickoff20' / 'diag' / f'{a}_s{s}_v2.json' for s in (0, 1)])
        nwe = pooled([D / f'{a}_s{s}_nuevo_eval.json' for s in (0, 1)])
        txt, ok = compare(nwv, v2v)
        txe, _ = compare(nwe, v2e)
        L.append(f"| {title} | {_pct(v2v)} | {_pct(nwv)} | {txt} | {'—' if ok is None else ('sí' if ok else 'no')} | "
                 f'{_pct(v2e)} | {_pct(nwe)} | {txe} |')
    L += ['', '## Entrenamiento de los previews', '']
    for a, title in AGENTS:
        rep = d / f'preview_{a}' / 'reporte.txt'
        txt = rep.read_text(encoding='utf-8', errors='replace') if rep.is_file() else ''
        m = re.search(r'MEAN_WR:\s*([\d.]+)', txt)
        live = next((l for l in txt.splitlines() if l.startswith('live:')), '')
        L.append(f"- **{title}:** MEAN_WR frente al preview del piloto = {m.group(1) if m else '—'}. {live}")
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    d = Path(args[0]).expanduser()
    (d / 'kickoff_summary22.md').write_text(build(d), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
