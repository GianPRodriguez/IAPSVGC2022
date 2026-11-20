"""Resumen de la cola de medición de MCTS+PUCT v2 (agente G).

Lee los reportes copiados por ``scripts/server/mcts_g_v2.sh``
(``<out>/reports/<config>_s<seed>.txt``) y escribe una tabla por configuración:
series ganadas, WR con IC 95 % (Wilson) y, para cada configuración, la diferencia
pareada por semilla frente a la configuración base (``v2_n200``) en las semillas
comunes. Es exploración (protocolo_evaluacion.md): no respalda afirmaciones.

    python -m showdown.inference_sim.mcts_g_summary ~/iapsvgc_runs/mcts_g_v2 [base]

``base``: configuración de referencia para el Δ pareado (por defecto ``v2_n200``).
"""
from __future__ import annotations

import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from showdown.inference_sim.run_mcts_sims_sweep import parse_metrics_from_report

BASE = 'v2_n200'
_NAME = re.compile(r'^(?P<cfg>.+)_s(?P<seed>\d+)\.txt$')


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def load(out: Path) -> Dict[str, Dict[int, dict]]:
    res: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for p in sorted((out / 'reports').glob('*.txt')):
        m = _NAME.match(p.name)
        if not m:
            continue
        met = parse_metrics_from_report(p.read_text(encoding='utf-8', errors='replace'))
        errs = re.search(r'series_errors:\s*(\d+)', p.read_text(encoding='utf-8', errors='replace'))
        met['series_errors'] = int(errs.group(1)) if errs else 0
        res[m.group('cfg')][int(m.group('seed'))] = met
    return res


def paired_diff(a: Dict[int, dict], b: Dict[int, dict]) -> Optional[Tuple[float, float, int]]:
    """Media y semiancho IC 95 % (t aprox.) de WR_a − WR_b por semilla común."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    d = [a[s]['wr_raw'] - b[s]['wr_raw'] for s in seeds]
    mu = sum(d) / len(d)
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1))
    return mu, 2.0 * sd / math.sqrt(len(d)), len(seeds)


def render(res: Dict[str, Dict[int, dict]], base_cfg: str = BASE) -> str:
    lines: List[str] = [
        '# MCTS+PUCT v2 (G) — resumen de la cola', '',
        'Rival: greedy_joint validado. WR = series BO3 ganadas por G. Exploración.', '',
        f'| Config | Semillas | Series | Ganadas | WR | IC 95 % | Errores | Tiempo medio/semilla | Δ vs {base_cfg} (pareado) |',
        '|---|---|---:|---:|---:|---|---:|---:|---|',
    ]
    base = res.get(base_cfg, {})
    for cfg in sorted(res):
        rows = res[cfg]
        w = sum(r['wins'] for r in rows.values())
        n = sum(r['wins'] + r['losses'] for r in rows.values())
        lo, hi = wilson(w, n)
        err = sum(r['series_errors'] for r in rows.values())
        t = sum(r['elapsed_s'] for r in rows.values()) / max(1, len(rows))
        seeds = sorted(rows)
        span = f'{seeds[0]}–{seeds[-1]} ({len(seeds)})' if seeds else '-'
        pd = paired_diff(rows, base) if cfg != base_cfg else None
        pds = f'{pd[0]:+.1f} ± {pd[1]:.1f} pts ({pd[2]} sem.)' if pd else '-'
        wr = 100 * w / n if n else 0.0
        lines.append(f'| {cfg} | {span} | {n} | {w} | {wr:.1f} % | [{lo:.1f}, {hi:.1f}] | {err} | {t:.0f} s | {pds} |')
    lines.append('')
    return '\n'.join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    out = Path(argv[0]).expanduser()
    text = render(load(out), argv[1] if len(argv) > 1 else BASE)
    (out / 'summary.md').write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
