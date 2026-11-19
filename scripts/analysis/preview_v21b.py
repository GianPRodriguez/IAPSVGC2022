"""Preview v2.1 reentrenado sin val (memorizacion_equipos.md §18.4): elección y comparación con el v2.

Combate fijo (PPO y DDQN con P5, semillas 0 y 1). Regla fijada antes de ver resultados: se elige el
preview nuevo con mejor `val` (PPO y DDQN agrupados) y se adopta si en `val` no pierde más de 2 puntos
frente al preview v2 (que conoce los equipos de `val`). `eval` es registro.

    python scripts/analysis/preview_v21b.py ~/iapsvgc_runs/preview_v21b ~/iapsvgc_runs/ablacion_p5
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from showdown.rl.kickoff_summary24 import _load, diff, fd, pct, pooled  # noqa: E402


def main(argv=None) -> int:
    a = [Path(x).expanduser() for x in (sys.argv[1:] if argv is None else argv)]
    out, abl = a
    D, A = out / 'diag', abl / 'diag'
    g = lambda d, n: _load(d / f'{n}.json')  # noqa: E731
    ref = {sp: {'ppo': [g(A, f'B_ppo_p5_pv2_s{s}_{sp}') for s in (0, 1)],
                'ddqn': [g(A, f'D_ddqn_p5_pv2_s{s}_{sp}') for s in (0, 1)]} for sp in ('val', 'eval')}
    L = ['# Preview v2.1 reentrenado sin `val` (combate PPO y DDQN con P5)', '',
         '| Preview | MEAN_WR entr. | PPO `val` | DDQN `val` | `val` agrupado | Δ`val` vs v2 | PPO `eval` | DDQN `eval` | '
         'Δ`eval` vs v2 |', '|---|---|---|---|---|---|---|---|---|']
    rv = pooled(ref['val']['ppo'] + ref['val']['ddqn'])
    re_ = pooled(ref['eval']['ppo'] + ref['eval']['ddqn'])
    L.append(f"| v2 (congelado, conoce `val`) | — | {pct(pooled(ref['val']['ppo']))} | {pct(pooled(ref['val']['ddqn']))} | "
             f"{pct(rv)} | — | {pct(pooled(ref['eval']['ppo']))} | {pct(pooled(ref['eval']['ddqn']))} | — |")
    best = None
    for ps in (0, 1):
        c = {sp: {ag: [g(D, f'pv{ps}_{ag}_s{s}_{sp}') for s in (0, 1)] for ag in ('ppo', 'ddqn')}
             for sp in ('val', 'eval')}
        v = pooled(c['val']['ppo'] + c['val']['ddqn'])
        e = pooled(c['eval']['ppo'] + c['eval']['ddqn'])
        try:
            mw = re.search(r'MEAN_WR:\s*([\d.]+)', (out / f'preview_s{ps}' / 'reporte.txt').read_text()).group(1)
        except Exception:
            mw = '—'
        L.append(f"| v2.1 sin `val`, semilla {ps} | {mw} | {pct(pooled(c['val']['ppo']))} | "
                 f"{pct(pooled(c['val']['ddqn']))} | {pct(v)} | {fd(diff(v, rv))} | {pct(pooled(c['eval']['ppo']))} | "
                 f"{pct(pooled(c['eval']['ddqn']))} | {fd(diff(e, re_))} |")
        if v[1] and (best is None or v[0] / v[1] > best[1]):
            best = (ps, v[0] / v[1], diff(v, rv))
    if best is None:
        L += ['', '**Sin datos.**']
    else:
        ok = best[2] is not None and best[2][0] >= -2.0
        L += ['', f'Elegido por `val`: semilla {best[0]}. **{"Se adopta" if ok else "No se adopta"}** '
              f'(Δ`val` frente al v2 {fd(best[2])}; regla: ≥ −2).']
    md = '\n'.join(L) + '\n'
    (out / 'preview_v21b.md').write_text(md, encoding='utf-8')
    print(md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
