"""Resumen de la vigesimoprimera cola: rival de arquetipos y ajuste de entropía de PPO (§14).

    python -m showdown.rl.kickoff_summary21 ~/iapsvgc_runs/kickoff21 ~/iapsvgc_runs
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from showdown.rl.context_diag import summarize
from showdown.rl.kickoff_summary17 import THRESHOLD, _pct, compare, pooled

AGENTS = (('ddqn', 'DDQN (REG 0,03)'), ('ppo', 'PPO (simetría)'))
JOINT = ('greedy_joint',)


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def _joint_rate(paths):
    w = n = 0
    for p in paths:
        r = _load(p)
        if r:
            c = r.get('resultados', {}).get('greedy_joint', {})
            w += c.get('win', 0)
            n += sum(c.values())
    return (w / n, n) if n else None


def _events(paths):
    """% de partidas contra joint en que el rival activa Trick Room / Tailwind."""
    n, tr, tw = 0, 0, 0
    for p in paths:
        for g in (_load(p) or {}).get('partidas', []):
            if g.get('rival') != 'greedy_joint':
                continue
            n += 1
            ev = (g.get('eventos') or {}).get('rival', {})
            tr += bool(ev.get('trick_room_activado'))
            tw += bool(ev.get('tailwind_activado'))
    return n, tr, tw


def build(d: Path, runs: Path) -> str:
    D = d / 'diag'
    L = ['# Resumen de la vigesimoprimera cola: rival de arquetipos y entropía de PPO', '',
         '## 1. greedy_joint normal frente a greedy_joint con modo arquetipo', '',
         'Tu equipo, rivales `eval`, final.pt, 2 semillas; el modo arquetipo (`PS_JOINT_ARCHETYPE=1`) '
         'activa Tailwind siempre que no esté puesto y Trick Room cuando su lado es más lento. Referencia '
         'normal: vigésima cola (preview v2, mismos checkpoints).', '',
         '| Agente | Victorias contra joint normal | Contra joint arquetipo | Rival activa Trick Room (normal → arquetipo) | Rival activa Tailwind (normal → arquetipo) |',
         '|---|---|---|---|---|']
    for a, title in AGENTS:
        norm = [runs / 'kickoff20' / 'diag' / f'{a}_s{s}_v2.json' for s in (0, 1)]
        arch = [D / f'{a}_s{s}_arquetipo.json' for s in (0, 1)]
        rn, ra = _joint_rate(norm), _joint_rate(arch)
        nn_, trn, twn = _events(norm)
        na, tra, twa = _events(arch)
        f = lambda k, n: f'{100 * k / n:.0f} %' if n else '—'
        L.append(f'| {title} | {_pct(rn)} | {_pct(ra)} | {f(trn, nn_)} → {f(tra, na)} | {f(twn, nn_)} → {f(twa, na)} |')
    L += ['']
    for a, title in AGENTS:
        ctx = defaultdict(Counter)
        for s in (0, 1):
            for c, v in ((_load(D / f'{a}_s{s}_arquetipo.json') or {}).get('contexto') or {}).items():
                ctx[c].update(v)
        L += [f'### {title}: decisiones según el contexto contra el rival de arquetipos', '']
        L += (summarize(ctx) if ctx else ['Sin datos.']) + ['']

    L += ['## 2. Entropía de PPO (1M, 2 semillas; elección con `val`)', '',
          f'Referencia: PPO con simetría y entropía 0,01 (por defecto). Se recomienda el cambio si supera '
          f'a la referencia por más de {100 * THRESHOLD:.0f} puntos en `val`. `eval` y propios: registro.', '',
          '| Entropía | `val` | Referencia `val` | Diferencia | ¿Se recomienda? | `eval` (registro) | Propios (registro) | Cambios voluntarios |',
          '|---|---|---|---|---|---|---|---|']
    ref = pooled([runs / 'kickoff17' / 'diag' / f'PPO1M_s{s}_final_val.json' for s in (0, 1)])
    for name, label in (('ENT003', '0,003'), ('ENT03', '0,03')):
        new = pooled([D / f'{name}_s{s}_final_val.json' for s in (0, 1)])
        ev = pooled([D / f'{name}_s{s}_final_eval.json' for s in (0, 1)])
        ow = pooled([D / f'{name}_s{s}_final_own.json' for s in (0, 1)])
        sw = dec = 0
        for s in (0, 1):
            tot = (_load(D / f'{name}_s{s}_final_eval.json') or {}).get('total', {})
            sw += tot.get('cambio_voluntario', 0)
            dec += tot.get('decisiones', 0)
        txt, ok = compare(new, ref)
        L.append(f"| {label} | {_pct(new)} | {_pct(ref)} | {txt} | {'—' if ok is None else ('sí' if ok else 'no')} | "
                 f"{_pct(ev)} | {_pct(ow)} | {f'{100 * sw / dec:.1f} %' if dec else '—'} |")
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    d, runs = Path(args[0]).expanduser(), Path(args[1]).expanduser()
    (d / 'kickoff_summary21.md').write_text(build(d, runs), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
