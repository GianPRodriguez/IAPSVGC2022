"""Resumen de la precisión del modelo del rival de expectimax (ver ``opp_model_trace``).

    python -m showdown.inference_sim.analyze_opp_model logs/oppmodel.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def load(paths: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    for p in paths:
        with Path(p).open(encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(rows: Sequence[dict]) -> dict:
    ok = [r for r in rows if r.get('status') == 'ok']
    status: Dict[str, int] = defaultdict(int)
    for r in rows:
        status[r.get('status', '?')] += 1
    att = [r for r in ok if not r['control']]
    ctl = [r for r in ok if r['control']]
    out: dict = {'eventos': len(rows), 'estado': dict(status), 'con_prediccion': len(ok)}
    if not ok:
        return out
    out.update({
        'p_asignada_al_movimiento_real': _mean([r['p_move'] for r in ok]),
        'p_base_uniforme': _mean([r['uniform_p'] for r in ok]),
        'acierto_top1_movimiento': _mean([1.0 if r['top1_move_hit'] else 0.0 for r in ok]),
        'acierto_top1_movimiento_y_objetivo': _mean([1.0 if r['top1_exact_hit'] else 0.0 for r in ok]),
        'p_media_exacta_movimiento_y_objetivo': _mean([r['p_exact'] for r in ok]),
        'fraccion_real_de_movimientos_de_control': len(ctl) / len(ok),
        'masa_predicha_de_control': _mean([r['mass_control'] for r in ok]),
        'masa_predicha_de_cambio': _mean([r['mass_switch'] for r in ok]),
        'ataques': len(att),
        'acierto_top1_en_ataques': _mean([1.0 if r['top1_move_hit'] else 0.0 for r in att]),
        'p_asignada_en_ataques': _mean([r['p_move'] for r in att]),
        'control': len(ctl),
        'p_asignada_en_control': _mean([r['p_move'] for r in ctl]),
    })
    return out


def reliability(rows: Sequence[dict], bins: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.01)) -> List[str]:
    """Para cada franja de ``p_move`` predicha, qué fracción de veces fue el movimiento real.

    Cada fila es un movimiento realmente usado, así que la tabla correcta compara la
    masa media por franja con la frecuencia de acierto del top-1 en esa franja.
    """
    ok = [r for r in rows if r.get('status') == 'ok']
    lines = ['franja de p(movimiento real)   n   acierto_top1']
    for lo, hi in zip(bins, bins[1:]):
        grp = [r for r in ok if lo <= r['p_move'] < hi]
        if grp:
            lines.append(f'  [{lo:.2f}, {min(hi, 1.0):.2f})            {len(grp):6d}   '
                         f'{_mean([1.0 if r["top1_move_hit"] else 0.0 for r in grp]):.3f}')
    return lines


def by_species_top(rows: Sequence[dict], top: int = 6) -> List[str]:
    ok = [r for r in rows if r.get('status') == 'ok']
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in ok:
        groups[r['species']].append(r)
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:top]
    return [f'  {sp:18s} n={len(g):5d} p_real={_mean([x["p_move"] for x in g]):.3f} '
            f'top1={_mean([1.0 if x["top1_move_hit"] else 0.0 for x in g]):.3f}' for sp, g in ranked]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Precisión del modelo del rival')
    ap.add_argument('log', nargs='+')
    args = ap.parse_args(list(argv) if argv is not None else None)
    rows = load(args.log)
    s = summarize(rows)
    for k, v in s.items():
        print(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}')
    print(chr(10).join(reliability(rows)))
    print('especies más frecuentes:')
    print(chr(10).join(by_species_top(rows)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
