"""Resumen de la vigesimocuarta cola: candidatos para PPO v2 (memorizacion_equipos.md §17).

Lee ``<out>/diag/*.json`` (policy_diag, 200 partidas por rival) y la referencia PPO v1 congelada
(``<v1>/diag``: kick-off oficial) y escribe ``<out>/kickoff_summary24.md``.

Reglas de decisión (fijadas antes de ver resultados):
- P1 (sorteo): por temperatura, media entre las 5 semillas de v1 de (sorteo − determinista) en `val`.
  Se adopta para jugar contra personas si Δ ≥ −2 puntos; si ambas cumplen, la de T = 1.
- P2 / P4 (rival de arquetipos; + entropía 0,03): frente a v1 (semillas 0 y 1, partidas agrupadas),
  se adopta si Δ`val` ≥ −2 y Δ(arquetipo nivel 2, `val`) > +4.
- P3 (rival PPO v1 espejo): se adopta si Δ`val` ≥ −2 y gana > 54 % contra el PPO v1 congelado.
- Cualquiera: Δ`val` > +4 → se adopta (regla general del protocolo).
`eval` es solo registro.

    python -m showdown.rl.kickoff_summary24 ~/iapsvgc_runs/ppo_v2 ~/iapsvgc_runs/ppo_v1
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HEUR = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')
TEMPS = ('1.0', '0.5')
CANDS = (('P2', 'Rival de arquetipos (nivel 2) en el entrenamiento'),
         ('P4', 'P2 + entropía 0,03'),
         ('P3', 'Rival PPO v1 congelado (espejo)'))


def _load(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def wl(rep: Optional[dict], foes=HEUR) -> Tuple[int, int]:
    if not rep:
        return 0, 0
    res = rep.get('resultados', {})
    w = sum(res.get(f, {}).get('win', 0) for f in foes)
    n = sum(sum(res.get(f, {}).values()) for f in foes)
    return w, n


def pooled(reps: List[Optional[dict]], foes=HEUR) -> Tuple[int, int]:
    w = n = 0
    for r in reps:
        a, b = wl(r, foes)
        w, n = w + a, n + b
    return w, n


def pct(wn: Tuple[int, int]) -> str:
    w, n = wn
    return f'{100 * w / n:.1f} %' if n else '—'


def diff(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[float, float]]:
    """(Δ en puntos, semiamplitud IC 95 %) de a − b, binomial con partidas agrupadas."""
    (wa, na), (wb, nb) = a, b
    if not na or not nb:
        return None
    pa, pb = wa / na, wb / nb
    se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    return 100 * (pa - pb), 100 * 1.96 * se


def fd(d: Optional[Tuple[float, float]]) -> str:
    return f'{d[0]:+.1f} (± {d[1]:.1f})' if d else '—'


def _arch_map():
    try:
        from showdown.inference_sim.team_factory import active_teams
        from showdown.inference_sim.team_split import team_key
        from showdown.rl.team_archetypes import feats, primary
    except Exception:
        return {}
    out = {}
    for split in ('train', 'eval'):
        for t in active_teams(split):
            out[team_key(t)] = primary(feats(t))
    return out


def by_archetype(reps: List[Optional[dict]], amap: Dict[str, str]) -> Dict[str, Tuple[int, int]]:
    acc: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in reps:
        for g in (r or {}).get('partidas', []):
            if g.get('rival') not in HEUR:
                continue
            a = amap.get(g.get('equipo') or '', 'desconocido')
            acc[a][1] += 1
            acc[a][0] += g.get('res') == 'win'
    return {k: (v[0], v[1]) for k, v in acc.items()}


def build(out: Path, v1: Path) -> str:
    D, V = out / 'diag', v1 / 'diag'
    g = lambda d, name: _load(d / f'{name}.json')  # noqa: E731
    L = ['# PPO v2: candidatos (vigesimocuarta cola)', '',
         'Referencia: **PPO v1 congelado** (kick-off oficial, `final.pt`). 200 partidas por rival, equipo fijo, '
         'preview v2. Cifras contra los 3 heurísticos salvo que se indique. Diferencias con IC 95 % binomial '
         '(partidas agrupadas). Reglas de decisión en el encabezado de `showdown/rl/kickoff_summary24.py`.', '']

    # ------------------------------------------------------------------ P1
    L += ['## P1. Jugar con algo de azar (sin reentrenar; 5 semillas de v1)', '',
          '| Modo | `val` | Δ`val` vs determinista | `eval` (registro) | Δ`eval` | Arquetipo n1 (joint, `eval`) | Decisión |',
          '|---|---|---|---|---|---|---|']
    gv = pooled([g(D, f'v1_s{s}_val') for s in range(5)])
    ge = pooled([g(V, f'ppo_s{s}_final_eval') for s in range(5)])
    ga = pooled([g(V, f'ppo_s{s}_final_arquetipo') for s in range(5)], ('greedy_joint',))
    L.append(f'| Determinista (v1) | {pct(gv)} | — | {pct(ge)} | — | {pct(ga)} | referencia |')
    p1_ok = []
    for t in TEMPS:
        sv = pooled([g(D, f'p1_T{t}_s{s}_val') for s in range(5)])
        se = pooled([g(D, f'p1_T{t}_s{s}_eval') for s in range(5)])
        sa = pooled([g(D, f'p1_T{t}_s{s}_arq1') for s in range(5)], ('greedy_joint',))
        dv = diff(sv, gv)
        ok = dv is not None and dv[0] >= -2.0
        if ok:
            p1_ok.append(t)
        L.append(f'| Sorteo T = {t} | {pct(sv)} | {fd(dv)} | {pct(se)} | {fd(diff(se, ge))} | {pct(sa)} | '
                 f'{"cumple (Δ ≥ −2)" if ok else ("no cumple" if dv else "sin datos")} |')
    chosen = '1.0' if '1.0' in p1_ok else (p1_ok[0] if p1_ok else None)
    L += ['', f'**P1:** {"se adopta el sorteo con T = " + chosen + " para jugar contra personas" if chosen else "no se adopta (o sin datos)"}. '
          'El resultado oficial de v1 sigue siendo el determinista.', '']

    # ------------------------------------------------------------------ P2 / P4 / P3
    L += ['## P2, P4 y P3 (2 semillas cada una, frente a v1 semillas 0 y 1)', '',
          '| Candidato | `val` | Δ`val` | Arquetipo n2 `val` | Δ | Contra PPO v1 (espejo) | `eval` (registro) | Δ`eval` | '
          'Arquetipo n1 `eval` | Arquetipo n2 `eval` | Decisión |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    rv = pooled([g(D, f'v1_s{s}_val') for s in (0, 1)])
    ra2 = pooled([g(D, f'v1_s{s}_arq2_val') for s in (0, 1)], ('greedy_joint_arq',))
    re_ = pooled([g(V, f'ppo_s{s}_final_eval') for s in (0, 1)])
    ra1e = pooled([g(V, f'ppo_s{s}_final_arquetipo') for s in (0, 1)], ('greedy_joint',))
    ra2e = pooled([g(D, f'v1_s{s}_arq2_eval') for s in (0, 1)], ('greedy_joint_arq',))
    L.append(f'| v1 (referencia) | {pct(rv)} | — | {pct(ra2)} | — | 50 % (por construcción) | {pct(re_)} | — | '
             f'{pct(ra1e)} | {pct(ra2e)} | — |')
    adopted = []
    for key, name in CANDS:
        n = key.lower()
        cv = pooled([g(D, f'{n}_s{s}_val') for s in (0, 1)])
        ca2 = pooled([g(D, f'{n}_s{s}_arq2_val') for s in (0, 1)], ('greedy_joint_arq',))
        ch = pooled([g(D, f'{n}_s{s}_h2h') for s in (0, 1)], ('ppo_v1',))
        ce = pooled([g(D, f'{n}_s{s}_eval') for s in (0, 1)])
        ca1e = pooled([g(D, f'{n}_s{s}_arq1_eval') for s in (0, 1)], ('greedy_joint',))
        ca2e = pooled([g(D, f'{n}_s{s}_arq2_eval') for s in (0, 1)], ('greedy_joint_arq',))
        dv, da = diff(cv, rv), diff(ca2, ra2)
        if dv is None:
            dec = 'sin datos'
        elif dv[0] > 4.0:
            dec = '**se adopta** (Δ`val` > +4)'
        elif dv[0] < -2.0:
            dec = 'no (pierde en `val`)'
        elif key in ('P2', 'P4') and da is not None and da[0] > 4.0:
            dec = '**se adopta** (arquetipo > +4)'
        elif key == 'P3' and ch[1] and ch[0] / ch[1] > 0.54:
            dec = '**se adopta** (> 54 % contra v1)'
        else:
            dec = 'no (no alcanza el umbral)'
        if 'adopta' in dec:
            adopted.append(key)
        L.append(f'| {key}: {name} | {pct(cv)} | {fd(dv)} | {pct(ca2)} | {fd(da)} | {pct(ch)} | {pct(ce)} | '
                 f'{fd(diff(ce, re_))} | {pct(ca1e)} | {pct(ca2e)} | {dec} |')
    if 'P2' in adopted and 'P4' in adopted:
        L += ['', 'P2 y P4 cumplen: se elige la de mayor `val` para v2 (P4 añade entropía 0,03).']

    # ------------------------------------------------------------------ por arquetipo (eval)
    amap = _arch_map()
    if amap:
        rows = {'v1 (5 semillas)': by_archetype([g(V, f'ppo_s{s}_final_eval') for s in range(5)], amap)}
        for key, _ in CANDS:
            rows[key] = by_archetype([g(D, f'{key.lower()}_s{s}_eval') for s in (0, 1)], amap)
        archs = sorted({a for r in rows.values() for a in r}, key=lambda a: -rows['v1 (5 semillas)'].get(a, (0, 0))[1])
        L += ['', '## Por arquetipo del rival (`eval`, 3 heurísticos; §16)', '',
              '| Arquetipo | ' + ' | '.join(rows) + ' |', '|---|' + '---|' * len(rows)]
        for a in archs:
            L.append(f'| {a} | ' + ' | '.join(
                f'{pct(r[a])} (n={r[a][1]})' if a in r else '—' for r in rows.values()) + ' |')

    L += ['', '## Siguiente paso', '',
          f'Elementos adoptados: {", ".join(adopted + (["P1 (sorteo T = " + chosen + ")"] if chosen else [])) or "ninguno"}. '
          'PPO v2 = v1 + lo adoptado, entrenado con 1M × 5 semillas y evaluado como el kick-off de v1 '
          '(pendiente de la aprobación del usuario).']
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path(args[0]).expanduser()
    v1 = Path(args[1]).expanduser() if len(args) > 1 else out.parent / 'ppo_v1'
    (out / 'kickoff_summary24.md').write_text(build(out, v1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
