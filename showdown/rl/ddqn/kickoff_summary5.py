"""Resumen de la quinta cola: verificación del preview v2 y simetría a 1M pasos (DDQN) y en PPO.

    python -m showdown.rl.ddqn.kickoff_summary5 ~/iapsvgc_runs/kickoff5 ~/iapsvgc_runs/kickoff2 ~/iapsvgc_runs/kickoff3
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List

from showdown.rl.ddqn.kickoff_summary import _eval_cells, _evals


def _pct(x) -> str:
    return f'{100 * float(x):.1f} %'


def section_preview(out: Path) -> List[str]:
    p = out / 'preview_check' / 'preview_check.json'
    lines = ['## 1. Verificación del preview v2 (pareado contra el preview heurístico, 512 contextos)', '']
    if not p.is_file():
        return lines + ['Sin datos.', '']
    d = json.loads(p.read_text(encoding='utf-8'))
    lines += ['| Comprobación | Preview | Heurístico | Diferencia (± EE) |', '|---|---|---|---|']
    for key, name in (('split_train', 'Rivales con equipos `train` (vistos)'),
                      ('split_eval', 'Rivales con equipos `eval` (no vistos)'),
                      ('formacion_fija', 'Formación más frecuente, siempre la misma')):
        v = d.get(key)
        if v:
            lines.append(f"| {name} | {_pct(v['mean_wr'])} | {_pct(v['base_wr'])} | "
                         f"{100 * v['delta_wr']:+.1f} (± {100 * v['delta_se']:.1f}) |")
    ff = d.get('formacion_fija', {})
    lines += ['', f"Formación fija: {', '.join(ff.get('traídos', []))}. Formaciones distintas elegidas: "
              f"{d.get('formaciones_distintas_train')} (train) / {d.get('formaciones_distintas_eval')} (eval).", '',
              '| Formación (train) | Fracción |', '|---|---|']
    for f in d.get('formaciones_train', [])[:6]:
        lines.append(f"| {', '.join(f['traídos'])} | {_pct(f['frac'])} |")
    rp = d.get('replays', {})
    lines += ['', f"Replays: `{rp.get('dir')}` ({len(rp.get('contextos', []))} contextos; preview y heurístico)."]
    for c in rp.get('contextos', [])[:12]:
        lines.append(f"- {c.get('rival')}: rival {', '.join(c.get('rival_especies') or [])} → trae "
                     f"{', '.join(c.get('preview_trae') or [])}")
    return lines + ['']


def section_sym(out: Path, q3: Path) -> List[str]:
    lines = ['## 2. DDQN 1M pasos: con simetría (cola 3) frente a sin simetría', '']
    cfg = out / 'mini_nosym_config.txt'
    if cfg.is_file():
        lines += [f"Configuración: `{cfg.read_text().strip()}`", '']
    lines += ['| Paso | Con simetría: media | Sin simetría: media | Sin simetría: random / solo / joint / expectimax |',
              '|---|---|---|---|']
    a = {e['step']: e for e in _evals(q3 / 'mini_kickoff')}
    b = {e['step']: e for e in _evals(out / 'mini_kickoff_nosym')}
    for step in sorted(set(a) | set(b)):
        ea, eb = a.get(step), b.get(step)
        lines.append(f"| {step // 1000}k | {_pct(ea['mean_wr']) if ea else '—'} | {_pct(eb['mean_wr']) if eb else '—'} | "
                     + (' / '.join(_eval_cells(eb)) if eb else '—') + ' |')
    return lines + ['']


def _suite(rep: Path) -> List[str]:
    if not rep.is_file():
        return ['Sin datos.']
    return [l for l in rep.read_text(encoding='utf-8', errors='replace').splitlines()
            if re.match(r'^(random|greedy_joint|expectimax_bayesian|MEAN_WR|steps_per_sec)', l)]


def section_ppo(out: Path, q2: Path) -> List[str]:
    return (['## 3. PPO: con simetría (cola 2) frente a sin simetría (400k pasos, evaluación sin exploración)', '',
             '**Con simetría:**', '```'] + _suite(q2 / 'ppo_pilot' / 'reporte.txt') + ['```', '',
             '**Sin simetría:**', '```'] + _suite(out / 'ppo_pilot_nosym' / 'reporte.txt') + ['```', ''])


def main(argv=None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    out, q2, q3 = (Path(a).expanduser() for a in args[:3])
    lines = ['# Resumen de la quinta cola previa al kick-off', '']
    q = out / 'queue.log'
    if q.is_file():
        lines += ['```', q.read_text(encoding='utf-8', errors='replace')[-2000:], '```', '']
    for fn, a in ((section_preview, (out,)), (section_sym, (out, q3)), (section_ppo, (out, q2))):
        try:
            lines += fn(*a)
        except Exception as exc:
            lines += [f'Error al resumir ({fn.__name__}): {exc!r}', '']
    (out / 'kickoff_summary5.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
