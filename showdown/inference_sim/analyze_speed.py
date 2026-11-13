"""Mide la poda por orden de velocidad (``observe_speed_order``) con verdad terreno.

Entrada: JSONL de ``calibration.py`` (``PS_CALIB_LOG``); solo se leen las filas
``kind == 'speed'``. Para cada observación se sabe qué SP de Spe quedaban antes y
después de aplicarla y cuál es el verdadero.

    riesgo : fracción de observaciones que dejaron fuera el SP verdadero
    poda   : fracción de los SP candidatos que la observación eliminó
    final  : por (serie, especie), cuántos SP de Spe quedan al terminar

    python -m showdown.inference_sim.analyze_speed logs/calib.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from showdown.inference_sim.analyze_evidence import wilson_upper


def load(path: Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('kind') == 'speed' and r.get('before') is not None:
                rows.append(r)
    return rows


def _flags(r: dict) -> str:
    tags = []
    if r.get('trick_room'):
        tags.append('trick_room')
    if r.get('opp_item'):
        tags.append('item')
    if r.get('opp_para'):
        tags.append('para')
    if r.get('opp_tailwind'):
        tags.append('tailwind')
    if r.get('opp_boost'):
        tags.append('boost')
    return '+'.join(tags) or 'plano'


def summarize(rows: Sequence[dict]) -> dict:
    n = len(rows)
    lost = [r for r in rows if r['true_in_before'] and not r['true_in_after']]
    changed = [r for r in rows if len(r['after']) < len(r['before'])]
    prune = [1 - len(r['after']) / len(r['before']) for r in rows if r['before']]
    return {
        'observaciones': n,
        'con_poda': len(changed),
        'verdadero_excluido': len(lost),
        'riesgo': len(lost) / n if n else 0.0,
        'riesgo_ub95': wilson_upper(len(lost), n),
        'poda_media': sum(prune) / len(prune) if prune else 0.0,
    }


def by_group(rows: Sequence[dict], keyf) -> Dict[str, dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        groups[keyf(r)].append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def final_sizes(rows: Sequence[dict]) -> dict:
    """Tamaño final del conjunto de SP de Spe por (pid, serie, especie)."""
    last: Dict[Tuple, dict] = {}
    initial = 0
    for r in sorted(rows, key=lambda x: (x.get('pid'), x.get('series'), x.get('seq', 0))):
        initial = max(initial, len(r['before']))
        last[(r.get('pid'), r.get('series'), r['species'])] = r
    sizes = sorted(len(r['after']) for r in last.values())
    if not sizes:
        return {'casos': 0}

    def q(p: float) -> int:
        return sizes[min(len(sizes) - 1, int(p * (len(sizes) - 1)))]

    return {
        'casos': len(sizes), 'sp_finales_p50': q(0.5), 'p90': q(0.9),
        'resueltos_<=3_sp': sum(1 for s in sizes if s <= 3) / len(sizes),
        'sin_poda': sum(1 for s in sizes if s >= initial) / len(sizes),
    }


def format_report(rows: Sequence[dict], examples: int = 5) -> str:
    out = ['TOTAL ' + json.dumps(summarize(rows), ensure_ascii=False), 'POR CONTEXTO']
    for k, v in by_group(rows, _flags).items():
        out.append(f'  {k:22s} obs {v["observaciones"]:6d} poda {v["poda_media"]:.3f} '
                   f'excluye_verdadero {v["verdadero_excluido"]} (riesgo {v["riesgo"]:.4f}, '
                   f'UB95 {v["riesgo_ub95"]:.4f})')
    out.append('POR DIRECCIÓN')
    for k, v in by_group(rows, lambda r: 'rival_primero' if r['opponent_first'] else 'rival_despues').items():
        out.append(f'  {k:22s} obs {v["observaciones"]:6d} poda {v["poda_media"]:.3f} '
                   f'excluye_verdadero {v["verdadero_excluido"]}')
    out.append('TAMAÑO FINAL ' + json.dumps(final_sizes(rows), ensure_ascii=False))
    bad = [r for r in rows if r['true_in_before'] and not r['true_in_after']][:examples]
    for r in bad:
        out.append(f'  FALLO {r["species"]} sp_verdadero={r["true_spe_sp"]} {_flags(r)} '
                   f'rival_primero={r["opponent_first"]} vel_propia={r["user_speed"]} '
                   f'antes={len(r["before"])} despues={len(r["after"])}')
    return chr(10).join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Poda por orden de velocidad con verdad terreno')
    ap.add_argument('log', nargs='+')
    args = ap.parse_args(list(argv) if argv is not None else None)
    rows: List[dict] = []
    for p in args.log:
        rows.extend(load(Path(p)))
    print(format_report(rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
