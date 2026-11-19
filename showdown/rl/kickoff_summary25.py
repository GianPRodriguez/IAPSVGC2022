"""Resumen de la vigesimoquinta cola: P5, corrección de habilidades en la observación (PPO y DDQN).

Decisión del usuario (18-11-2026): P5 se activa en PPO y DDQN (por defecto en los entrenamientos
nuevos). Esta cola mide su efecto, no decide: si algún agente pierde más de 2 puntos en `val`, se
avisa para revisarlo antes de la corrida final de PPO v2 y del kick-off de DDQN.

Referencias (misma configuración, observación anterior, semillas 0 y 1, final.pt):
- PPO: v1 congelado (``<v2>/diag/v1_s{0,1}_val``; `eval` del kick-off en ``<v1>/diag``).
- DDQN: REG weight decay 0,03 (decimoquinta cola, ``<q15>/diag/R_wd03_s{0,1}_final_val``; `eval`
  repetido aquí con 200 partidas por rival).

    python -m showdown.rl.kickoff_summary25 ~/iapsvgc_runs/p5 ~/iapsvgc_runs/ppo_v2 ~/iapsvgc_runs/ppo_v1 \
        ~/iapsvgc_runs/kickoff15
"""
from __future__ import annotations

import sys
from pathlib import Path

from showdown.rl.kickoff_summary24 import HEUR, _load, diff, fd, pct, pooled


def build(out: Path, v2: Path, v1: Path, q15: Path) -> str:
    D = out / 'diag'
    g = lambda d, n: _load(d / f'{n}.json')  # noqa: E731
    rows = [
        ('PPO', [g(v2 / 'diag', f'v1_s{s}_val') for s in (0, 1)], [g(D, f'ppo_p5_s{s}_val') for s in (0, 1)],
         [g(v1 / 'diag', f'ppo_s{s}_final_eval') for s in (0, 1)], [g(D, f'ppo_p5_s{s}_eval') for s in (0, 1)]),
        ('DDQN', [g(q15 / 'diag', f'R_wd03_s{s}_final_val') for s in (0, 1)], [g(D, f'ddqn_p5_s{s}_val') for s in (0, 1)],
         [g(D, f'ddqn_ref_s{s}_eval') for s in (0, 1)], [g(D, f'ddqn_p5_s{s}_eval') for s in (0, 1)]),
    ]
    L = ['# P5: corrección de habilidades en la observación (vigesimoquinta cola)', '',
         'Liquid Voice (sonido → Agua), habilidades de potencia (Tough Claws, Fairy Aura, Sharpness, Iron Fist, '
         'Technician, Sheer Force, Parental Bond…) y defensivas (Thick Fat, Heatproof, Fluffy…) en la estimación '
         'de daño. 1M pasos, semillas 0 y 1, final.pt, 200 partidas por rival, contra los 3 heurísticos. '
         'Decisión del usuario: se activa en ambos; aquí solo se mide.', '',
         '| Agente | `val` antes | `val` con P5 | Δ`val` | `eval` antes | `eval` con P5 | Δ`eval` | Aviso |',
         '|---|---|---|---|---|---|---|---|']
    for name, rv, cv, re_, ce in rows:
        a, b, c, d = pooled(rv), pooled(cv), pooled(re_), pooled(ce)
        dv = diff(b, a)
        warn = 'sin datos' if dv is None else ('**revisar: pierde más de 2 puntos**' if dv[0] < -2 else 'ok')
        L.append(f'| {name} | {pct(a)} | {pct(b)} | {fd(dv)} | {pct(c)} | {pct(d)} | {fd(diff(d, c))} | {warn} |')
    L += ['', '## Por rival (con P5, `eval`)', '', '| Agente | ' + ' | '.join(HEUR) + ' |', '|---|---|---|---|']
    for name, _rv, _cv, _re, ce in rows:
        L.append(f'| {name} | ' + ' | '.join(pct(pooled(ce, (f,))) for f in HEUR) + ' |')
    a1 = pooled([g(D, f'ppo_p5_s{s}_arq1_eval') for s in (0, 1)], ('greedy_joint',))
    h2h = pooled([g(D, f'ppo_p5_s{s}_h2h') for s in (0, 1)], ('ppo_v1',))
    L += ['', f'PPO con P5: arquetipo nivel 1 (`eval`) {pct(a1)} (v1: 77,1 %); contra PPO v1 en espejo {pct(h2h)} '
          '(50 % = igual que v1).']
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    out = Path(a[0]).expanduser()
    v2 = Path(a[1]).expanduser() if len(a) > 1 else out.parent / 'ppo_v2'
    v1 = Path(a[2]).expanduser() if len(a) > 2 else out.parent / 'ppo_v1'
    q15 = Path(a[3]).expanduser() if len(a) > 3 else out.parent / 'kickoff15'
    (out / 'kickoff_summary25.md').write_text(build(out, v2, v1, q15), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
