"""Fase 1 de expectimax: escala de los términos de V, contrafactual de la amenaza y puerta.

Lee las líneas que escribe expectimax con ``PS_VTERMS_LOG=<ruta>`` (un archivo por
proceso, ``<ruta>.<pid>``) y resume:

- media y desviación de cada término (ofensa, utilidad VGC, tempo, amenaza, sinergia)
  sobre todos los pares puntuados y sobre el par elegido;
- contrafactual: con qué frecuencia cambiaría el mejor par si V restara ``w·amenaza``;
- la puerta: cuántas veces el mejor par de phase2 ya era el de joint y cuántas se aceptó.

    PS_VTERMS_LOG=logs/vterms python -m showdown.inference_sim.run_robust_e2e ...
    python -m showdown.inference_sim.analyze_vterms logs/vterms
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from typing import Dict, List, Sequence

TERMS = ('off', 'vgc', 'tempo', 'threat', 'syn')


def load(prefix: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(glob.glob(prefix + '*')):
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return rows


def _stats(xs: Sequence[float]) -> str:
    if not xs:
        return 'n=0'
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
    return f'media {m:8.1f}  sd {sd:7.1f}  n={len(xs)}'


def _v(p: dict, rec: dict, w_threat: float) -> float:
    return p['off'] + p['vgc'] + rec['w_tempo'] * p['tempo'] - w_threat * p['threat']


def _argmax(pairs: Sequence[dict], rec: dict, w_threat: float) -> int:
    return max(range(len(pairs)), key=lambda i: _v(pairs[i], rec, w_threat))


def summarize(rows: Sequence[dict], weights=(0.05, 0.1, 0.2, 0.3, 0.5, 1.0)) -> List[str]:
    out = [f'decisiones: {len(rows)}']
    allp = [p for r in rows for p in r['pairs']]
    best = [p for r in rows for p in r['pairs'] if p['best']]
    out.append('términos de V sobre todos los pares puntuados:')
    for t in TERMS:
        out.append(f'  {t:7s} {_stats([p[t] for p in allp])}')
    out.append('términos de V en el mejor par de phase2:')
    for t in TERMS:
        out.append(f'  {t:7s} {_stats([p[t] for p in best])}')
    # Escala relativa: rango de cada término DENTRO de una decisión (lo que decide).
    out.append('rango medio dentro de cada decisión (max - min entre pares):')
    for t in TERMS:
        spans = [max(p[t] for p in r['pairs']) - min(p[t] for p in r['pairs'])
                 for r in rows if len(r['pairs']) > 1]
        out.append(f'  {t:7s} {_stats(spans)}')
    # Contrafactual de la amenaza.
    multi = [r for r in rows if len(r['pairs']) > 1]
    out.append('contrafactual: % de decisiones donde cambia el mejor par si V resta w*amenaza:')
    for w in weights:
        changed = sum(1 for r in multi if _argmax(r['pairs'], r, w) != _argmax(r['pairs'], r, 0.0))
        out.append(f'  w={w:<4}  {100.0 * changed / max(1, len(multi)):5.1f} %')
    # Puerta.
    n = len(rows)
    best_is_p1 = sum(1 for r in rows if any(p['best'] and p['p1'] for p in r['pairs']))
    accepted = sum(1 for r in rows if r['accepted'])
    accepted_div = sum(1 for r in rows if r['accepted']
                       and not any(p['best'] and p['p1'] for p in r['pairs']))
    deltas = [r['best_v'] - r['p1_v'] for r in rows if r.get('p1_v') is not None]
    out.append('puerta (phase2 frente a joint):')
    out.append(f'  mejor par de phase2 = par de joint: {100.0 * best_is_p1 / max(1, n):5.1f} %')
    out.append(f'  phase2 aceptado: {100.0 * accepted / max(1, n):5.1f} %   '
               f'aceptado y distinto de joint: {100.0 * accepted_div / max(1, n):5.1f} %')
    out.append(f'  V(mejor) - V(joint): {_stats(deltas)}')
    kinds: Dict[str, int] = {}
    for p in best:
        for k in p.get('kinds', []):
            kinds[k] = kinds.get(k, 0) + 1
    out.append('tipos de acción en el mejor par: ' + ', '.join(
        f'{k}={v}' for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('prefix', help='Prefijo de PS_VTERMS_LOG (se leen <prefijo>.*)')
    args = ap.parse_args(argv)
    for line in summarize(load(args.prefix)):
        print(line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
