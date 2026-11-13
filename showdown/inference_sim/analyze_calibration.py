"""Analiza el registro de calibración (PS_CALIB_LOG) y propone tolerancias seguras.

Entrada: JSON lines de ``calibration.py``. Cada línea trae los residuos entre el
rango del calc (con las stats VERDADERAS del rival) y el daño observado:

    res_hi: el calc predice más daño del observado   (afecta a la cota superior)
    res_lo: el calc predice menos daño del observado (afecta a la cota inferior)

Una tolerancia nunca descarta el SP verdadero si es >= que el residuo máximo de
su categoría. La propuesta es ``ceil(max * margen) + 1`` con margen configurable
y exige un mínimo de observaciones; una categoría con pocas muestras conserva la
tolerancia actual (no hay evidencia para recortarla).

    python -m showdown.inference_sim.analyze_calibration logs/calib.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CURRENT = {'abs': 28, 'pct': 32, 'pct_bilateral_extra': 12}


def load(path: Path) -> List[dict]:
    rows = []
    with Path(path).open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def category(row: dict) -> str:
    unit = 'pct' if row.get('percent') else 'abs'
    who = 'opp_attacks' if row.get('opp_attacks') else 'opp_defends'
    kind = 'spread' if row.get('spread') else 'single'
    ko = 'ko' if row.get('ko') else 'nonko'
    return f'{unit}/{who}/{kind}/{ko}'


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summarize(rows: Iterable[dict]) -> Dict[str, dict]:
    by: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by[category(r)].append(r)
    out = {}
    for cat, rs in sorted(by.items()):
        ok = [r for r in rs if r.get('true_in_groups')]
        lo = [float(r['res_lo']) for r in ok]
        hi = [float(r['res_hi']) for r in ok]
        out[cat] = {
            'n': len(rs),
            'n_true': len(ok),
            'missing_true': len(rs) - len(ok),
            'res_lo_max': max(lo) if lo else 0.0,
            'res_lo_p99': percentile(lo, 0.99),
            'res_lo_nonzero': sum(1 for x in lo if x > 0),
            'res_hi_max': max(hi) if hi else 0.0,
            'res_hi_p99': percentile(hi, 0.99),
            'res_hi_nonzero': sum(1 for x in hi if x > 0),
        }
    return out


def propose(summary: Dict[str, dict], *, margin: float = 1.5, min_n: int = 300) -> Dict[str, dict]:
    """Tolerancia propuesta por categoría y lado (lo = cota inferior, hi = superior)."""
    out = {}
    for cat, s in summary.items():
        unit = cat.split('/')[0]
        cur = CURRENT['pct'] if unit == 'pct' else CURRENT['abs']
        enough = s['n_true'] >= min_n

        def tol(res_max: float) -> int:
            if not enough:
                return cur                      # sin evidencia: no se recorta
            return min(cur, int(math.ceil(res_max * margin)) + 1)

        out[cat] = {
            'enough_data': enough,
            'tol_lo': tol(s['res_lo_max']),
            'tol_hi': tol(s['res_hi_max']),
            'current': cur,
        }
    return out


def move_key(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def _unit(row: dict) -> str:
    return 'pct' if row.get('percent') else 'abs'


def _who(row: dict) -> str:
    return 'opp_attacks' if row.get('opp_attacks') else 'opp_defends'


def build_move_tolerances(
    rows: Iterable[dict],
    *,
    margin: float = 1.5,
    min_n: int = 60,
    cap: int = 14,
    lo_only: bool = False,
    floor_abs: int = 1,
    floor_pct: int = 1,
) -> Dict[str, dict]:
    """Tolerancia estrecha por (movimiento, unidad, quién ataca) medida con datos.

    Solo se incluyen combinaciones con ``min_n`` observaciones NO-KO cuya
    tolerancia resultante (``ceil(residuo_max * margen) + 1``) no supere ``cap``:
    las mecánicas mal modeladas (potencia según HP, golpes múltiples, objeto
    desconocido...) tienen residuos grandes y quedan con la tolerancia amplia.
    Los KO no se incluyen (el daño observado es solo cota inferior).
    """
    groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for r in rows:
        if r.get('ko') or not r.get('true_in_groups'):
            continue
        groups[(move_key(r['move']), _unit(r), _who(r))].append(r)
    out: Dict[str, dict] = {}
    for (mv, unit, who), rs in sorted(groups.items()):
        if len(rs) < min_n:
            continue
        floor = floor_pct if unit == 'pct' else floor_abs
        # Piso: un residuo máximo de 0 en pocas muestras no prueba que la cola sea 0.
        lo = max(floor, int(math.ceil(max(float(r['res_lo']) for r in rs) * margin)) + 1)
        hi = max(floor, int(math.ceil(max(float(r['res_hi']) for r in rs) * margin)) + 1)
        if lo > cap or (hi > cap and not lo_only):
            continue
        # lo_only: la cota superior no se estrecha (hi=None -> tolerancia amplia).
        out.setdefault(mv, {}).setdefault(unit, {})[who] = {
            'lo': lo, 'hi': None if lo_only else hi, 'n': len(rs),
        }
    return out


def check_violations(rows: Iterable[dict], tolerances: Dict[str, dict]) -> dict:
    """Cuenta observaciones (NO usadas para calibrar) donde el SP verdadero quedaría
    excluido por la tolerancia estrecha: debe ser 0 para considerarla segura."""
    seen = viol_lo = viol_hi = 0
    worst = []
    for r in rows:
        if r.get('ko') or not r.get('true_in_groups'):
            continue
        t = tolerances.get(move_key(r['move']), {}).get(_unit(r), {}).get(_who(r))
        if not t:
            continue
        seen += 1
        bad_lo = float(r['res_lo']) > t['lo']
        bad_hi = t.get('hi') is not None and float(r['res_hi']) > t['hi']
        viol_lo += bad_lo
        viol_hi += bad_hi
        if bad_lo or bad_hi:
            worst.append((r['move'], _unit(r), _who(r), r['res_lo'], r['res_hi'], t))
    return {'covered': seen, 'violations_lo': viol_lo, 'violations_hi': viol_hi,
            'examples': worst[:5]}


def coverage(rows: Iterable[dict], tolerances: Dict[str, dict]) -> dict:
    """Qué fracción de las observaciones no-KO usaría la tolerancia estrecha."""
    total = hit = 0
    for r in rows:
        if r.get('ko') or not r.get('true_in_groups'):
            continue
        total += 1
        if tolerances.get(move_key(r['move']), {}).get(_unit(r), {}).get(_who(r)):
            hit += 1
    return {'non_ko': total, 'narrow': hit, 'fraction': hit / total if total else 0.0}


def report(rows: List[dict], *, margin: float, min_n: int) -> str:
    summ = summarize(rows)
    prop = propose(summ, margin=margin, min_n=min_n)
    lines = [f'observaciones: {len(rows)}  (margen x{margin}, mínimo {min_n} por categoría)', '']
    lines.append(f'{"categoría":34s} {"n":>6s} {"lo_max":>7s} {"lo_p99":>7s} {"hi_max":>7s} '
                 f'{"hi_p99":>7s} | tol_lo tol_hi (actual)')
    for cat, s in summ.items():
        p = prop[cat]
        flag = '' if p['enough_data'] else '  (pocas muestras: se conserva la actual)'
        lines.append(
            f'{cat:34s} {s["n_true"]:6d} {s["res_lo_max"]:7.1f} {s["res_lo_p99"]:7.1f} '
            f'{s["res_hi_max"]:7.1f} {s["res_hi_p99"]:7.1f} | {p["tol_lo"]:6d} {p["tol_hi"]:6d} '
            f'({p["current"]}){flag}'
        )
        if s['missing_true']:
            lines.append(f'   {"":32s} sin grupo verdadero (ya podado antes): {s["missing_true"]}')
    return '\n'.join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('log', help='archivo JSONL de PS_CALIB_LOG')
    ap.add_argument('--margin', type=float, default=1.5)
    ap.add_argument('--min-n', type=int, default=300)
    ap.add_argument('--json', default=None, help='escribir la propuesta como JSON')
    ap.add_argument('--emit-moves', default=None,
                    help='escribir las tolerancias estrechas por movimiento (JSON)')
    ap.add_argument('--cap', type=int, default=14, help='tope de tolerancia estrecha')
    ap.add_argument('--floor-abs', type=int, default=1, help='piso de tolerancia (puntos de HP)')
    ap.add_argument('--floor-pct', type=int, default=1, help='piso de tolerancia (puntos %%)')
    ap.add_argument('--lo-only', action='store_true',
                    help='estrechar solo la cota inferior (la superior no pasó la validación)')
    ap.add_argument('--min-moves-n', type=int, default=60,
                    help='mínimo de observaciones por (movimiento, unidad, lado)')
    ap.add_argument('--check', default=None,
                    help='JSON de tolerancias por movimiento a comprobar contra este log')
    args = ap.parse_args(list(argv) if argv is not None else None)
    rows = load(Path(args.log))
    print(report(rows, margin=args.margin, min_n=args.min_n))
    if args.emit_moves:
        tol = build_move_tolerances(rows, margin=args.margin, min_n=args.min_moves_n, cap=args.cap,
                                    lo_only=args.lo_only, floor_abs=args.floor_abs,
                                    floor_pct=args.floor_pct)
        payload = {'version': 1, 'margin': args.margin, 'cap': args.cap, 'lo_only': bool(args.lo_only),
                   'min_n': args.min_moves_n, 'source_observations': len(rows), 'moves': tol}
        Path(args.emit_moves).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
        print(f'tolerancias estrechas: {len(tol)} movimientos -> {args.emit_moves}')
        print('cobertura sobre los mismos datos:', coverage(rows, tol))
    if args.check:
        tol = json.loads(Path(args.check).read_text(encoding='utf-8')).get('moves', {})
        print('comprobación (datos nuevos):', check_violations(rows, tol))
        print('cobertura:', coverage(rows, tol))
    if args.json:
        Path(args.json).write_text(
            json.dumps(propose(summarize(rows), margin=args.margin, min_n=args.min_n), indent=2),
            encoding='utf-8',
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
