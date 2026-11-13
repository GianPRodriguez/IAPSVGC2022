"""Análisis de reglas de acumulación de evidencia dentro de un BO3.

Hoy cada golpe decide solo: un valor de stat se excluye si UNA observación lo
contradice con una tolerancia muy amplia, y por eso casi nunca se poda. Una
alternativa es acumular: excluir un valor solo si ``k`` observaciones
independientes de la misma serie (BO3) lo contradicen con una tolerancia más
estrecha. Los equipos y sus sets no cambian entre los juegos de un BO3.

Entrada: JSONL de ``calibration.py`` generado con ``PS_CALIB_PROFILE=1``. Para
cada regla se mide, por (serie, especie, stat):

    riesgo : fracción de casos donde el valor VERDADERO queda excluido
    poder  : fracción media de valores falsos excluidos (cuánto se poda)

    python -m showdown.inference_sim.analyze_evidence train.jsonl --test test.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Rule = Tuple[str, int, int, float]  # (modo, tolerancia, k, fracción mínima)


def load(path: Path) -> List[dict]:
    rows = []
    with Path(path).open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('kind') != 'speed' and r.get('profile') and r.get('true_in_groups', True):
                rows.append(r)
    return rows


def group_by_case(rows: Iterable[dict], *, unit: str, who: Optional[str] = None
                  ) -> Dict[Tuple, List[dict]]:
    """Casos = (pid, serie, especie, stat); observaciones ordenadas por ``seq``."""
    cases: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in rows:
        if ('pct' if r.get('percent') else 'abs') != unit:
            continue
        if who and ('opp_attacks' if r.get('opp_attacks') else 'opp_defends') != who:
            continue
        cases[(r.get('pid'), r.get('series'), r['species'], r['stat'])].append(r)
    for obs in cases.values():
        obs.sort(key=lambda x: x.get('seq', 0))
    return cases


def wilson_upper(events: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 1.0
    p = events / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + margin) / den)


def evaluate_rule(cases: Dict[Tuple, List[dict]], rule: Rule) -> dict:
    """Aplica la regla a todos los casos con al menos ``k`` observaciones usables."""
    mode, tol, k, frac = rule
    key = f'{mode}{tol}'
    n_cases = n_true_out = 0
    power_sum = 0.0
    for obs in cases.values():
        usable = [o for o in obs if key in (o.get('profile') or {})]
        if len(usable) < k:
            continue
        finals = set(usable[0]['profile']['finals'])
        tf = usable[0].get('true_final')
        votes: Dict[int, int] = defaultdict(int)
        for o in usable:
            for f in o['profile'][key]:
                votes[f] += 1
        n = len(usable)
        excluded = {f for f, c in votes.items() if c >= k and c / n >= frac}
        n_cases += 1
        if tf in excluded:
            n_true_out += 1
        false_vals = finals - {tf}
        if false_vals:
            power_sum += len(excluded & false_vals) / len(false_vals)
    return {
        'cases': n_cases,
        'true_excluded': n_true_out,
        'risk': n_true_out / n_cases if n_cases else 0.0,
        'risk_ub95': wilson_upper(n_true_out, n_cases),
        'power': power_sum / n_cases if n_cases else 0.0,
    }


def default_rules(unit: str) -> List[Rule]:
    tols = (3, 6, 12, 20, 32) if unit == 'pct' else (2, 4, 8, 14, 28)
    rules: List[Rule] = []
    for mode in ('lo', 'bi'):
        for t in tols:
            for k in (1, 2, 3, 4):
                for frac in (0.6, 0.8, 1.0):
                    if k == 1 and frac != 1.0:
                        continue
                    rules.append((mode, t, k, frac))
    return rules


def format_frontier(rows: List[dict], *, max_risk: float = 0.01, min_power: float = 0.02,
                    top: int = 8) -> str:
    """Mejores reglas por poder con riesgo de entrenamiento <= ``max_risk`` (aunque no sea 0)."""
    cand = [r for r in rows
            if r['train']['cases'] >= 200 and r['train']['risk'] <= max_risk
            and r['train']['power'] >= min_power]
    cand.sort(key=lambda r: (-r['train']['power'], r['train']['risk']))
    lines = [f'  frontera: riesgo_train <= {max_risk:.3f} y poder >= {min_power:.2f}']
    for r in cand[:top]:
        tr, te = r['train'], r.get('test') or {}
        lines.append(f'    {str(r["rule"]):28s} riesgo tr {tr["risk"]:.4f} (UB95 {tr["risk_ub95"]:.4f}) '
                     f'poder {tr["power"]:.3f} | te: casos {te.get("cases", 0)} '
                     f'riesgo {te.get("risk", 0):.4f} poder {te.get("power", 0):.3f}')
    if not cand:
        lines.append('    (ninguna)')
    return chr(10).join(lines)


def correlation_report(cases: Dict[Tuple, List[dict]], rule: Rule) -> dict:
    """Si los fallos son independientes entre golpes, los casos con el valor verdadero
    excluido por k=1 se deben a UNA observación aislada; si son sistemáticos (objeto,
    habilidad, naturaleza desconocidos), el mismo Pokémon falla en varias."""
    mode, tol, _k, _frac = rule
    key = f'{mode}{tol}'
    flagged_cases = 0
    multi = 0
    shares: List[float] = []
    for obs in cases.values():
        usable = [o for o in obs if key in (o.get('profile') or {})]
        if not usable:
            continue
        tf = usable[0].get('true_final')
        hits = sum(1 for o in usable if tf in o['profile'][key])
        if hits:
            flagged_cases += 1
            if hits >= 2:
                multi += 1
            shares.append(hits / len(usable))
    return {
        'cases_with_any_false_alarm': flagged_cases,
        'of_which_repeated_(>=2_obs)': multi,
        'mean_share_of_obs_flagging_true': (sum(shares) / len(shares)) if shares else 0.0,
    }


def case_stats(cases: Dict[Tuple, List[dict]]) -> dict:
    sizes = sorted(len(v) for v in cases.values())
    if not sizes:
        return {'cases': 0}
    q = lambda p: sizes[min(len(sizes) - 1, int(p * (len(sizes) - 1)))]
    return {'cases': len(sizes), 'obs_per_case_p50': q(0.5), 'p90': q(0.9), 'max': sizes[-1],
            'share_ge2': sum(1 for x in sizes if x >= 2) / len(sizes),
            'share_ge3': sum(1 for x in sizes if x >= 3) / len(sizes)}


def sweep(train: Dict[Tuple, List[dict]], test: Optional[Dict[Tuple, List[dict]]],
          rules: Sequence[Rule]) -> List[dict]:
    out = []
    for rule in rules:
        row = {'rule': rule, 'train': evaluate_rule(train, rule)}
        if test is not None:
            row['test'] = evaluate_rule(test, rule)
        out.append(row)
    return out


def format_table(rows: List[dict], *, max_risk: float, top: int = 12) -> str:
    """Reglas con riesgo observado 0 en entrenamiento (y en prueba si la hay), por poder."""
    def ok(r: dict) -> bool:
        tr = r['train']
        if tr['cases'] < 200 or tr['true_excluded'] > 0:
            return False
        te = r.get('test')
        return te is None or (te['cases'] >= 100 and te['risk'] <= max_risk)

    good = sorted((r for r in rows if ok(r)), key=lambda r: -r['train']['power'])[:top]
    lines = [f'{"regla (modo,tol,k,frac)":28s} {"casos":>6s} {"riesgo tr":>10s} {"UB95 tr":>8s} '
             f'{"poder tr":>9s} | {"casos te":>8s} {"riesgo te":>10s} {"poder te":>9s}']
    for r in good:
        tr, te = r['train'], r.get('test') or {}
        lines.append(
            f'{str(r["rule"]):28s} {tr["cases"]:6d} {tr["risk"]:10.4f} {tr["risk_ub95"]:8.4f} '
            f'{tr["power"]:9.3f} | {te.get("cases", 0):8d} {te.get("risk", 0):10.4f} '
            f'{te.get("power", 0):9.3f}'
        )
    if not good:
        lines.append('(ninguna regla con riesgo 0 en entrenamiento y en prueba)')
    return '\n'.join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('train')
    ap.add_argument('--test', default=None)
    ap.add_argument('--max-risk', type=float, default=0.0,
                    help='riesgo máximo aceptado en la prueba independiente')
    ap.add_argument('--frontier-risk', type=float, default=0.01)
    ap.add_argument('--json', default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)
    train_rows = load(Path(args.train))
    test_rows = load(Path(args.test)) if args.test else None
    result = {}
    for unit in ('abs', 'pct'):
        for who in ('opp_attacks', 'opp_defends'):
            tr = group_by_case(train_rows, unit=unit, who=who)
            te = group_by_case(test_rows, unit=unit, who=who) if test_rows else None
            if not tr:
                continue
            title = f'{unit}/{who}'
            print(f'\n=== {title}  casos de entrenamiento: {case_stats(tr)}')
            rows = sweep(tr, te, default_rules(unit))
            base = next(r for r in rows if r['rule'] == ('lo', 28 if unit == 'abs' else 32, 1, 1.0))
            print(f'  regla actual (una observación, tolerancia amplia): riesgo '
                  f'{base["train"]["risk"]:.4f}, poder {base["train"]["power"]:.3f}')
            print(format_table(rows, max_risk=args.max_risk))
            print(format_frontier(rows, max_risk=args.frontier_risk))
            cr = correlation_report(tr, ('lo', 28 if unit == 'abs' else 32, 1, 1.0))
            print(f'  falsas alarmas de la regla actual por caso: {cr}')
            cr2 = correlation_report(tr, ('lo', 8 if unit == 'abs' else 12, 1, 1.0))
            print(f'  falsas alarmas con tolerancia estrecha:     {cr2}')
            result[title] = [{'rule': list(r['rule']), 'train': r['train'], 'test': r.get('test')}
                             for r in rows]
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
