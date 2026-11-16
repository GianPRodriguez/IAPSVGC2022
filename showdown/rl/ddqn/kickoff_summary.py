"""Resumen de las ejecuciones previas al kick-off de DDQN (etapas 1–6 del checklist).

Lee lo que deja ``scripts/server/kickoff_queue.sh`` en un directorio y escribe
``kickoff_summary.md``: verificación de la reanudación, auditoría de la observación,
probe del preview, entrenamiento del preview PPO, tuning de DDQN y ablación
codificador × simetría.

    python -m showdown.rl.ddqn.kickoff_summary ~/iapsvgc_runs/kickoff
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _evals(ck: Path) -> List[dict]:
    return [r for r in _jsonl(ck / 'train_metrics.jsonl') if r.get('kind') == 'eval']


def _last_learner(ck: Path) -> dict:
    wins = [r for r in _jsonl(ck / 'train_metrics.jsonl')
            if r.get('kind') == 'window' and isinstance(r.get('learner'), dict) and 'mean_q' in r['learner']]
    return wins[-1]['learner'] if wins else {}


def _rate(ck: Path) -> str:
    rep = ck / 'reporte.txt'
    if not rep.is_file():
        return '?'
    m = re.search(r'steps_per_sec:\s*([\d.]+)', rep.read_text(encoding='utf-8', errors='replace'))
    return f'{float(m.group(1)):.0f}' if m else '?'


def _foe(row: dict, key: str) -> str:
    return f"{100 * float(row.get('win_rate', 0)):.0f} %"


def _eval_cells(ev: dict) -> List[str]:
    rows = {r['foe']: r for r in ev.get('rows', [])}
    return [(_foe(rows[f], 'win_rate') if f in rows else '—')
            for f in ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')]


def section_resume(d: Path) -> List[str]:
    out = ['## 1. Reanudación (verificación en vivo)', '']
    t2 = d / 'resume' / 'train2.txt'
    if not t2.is_file():
        return out + ['Sin datos.', '']
    txt = t2.read_text(encoding='utf-8', errors='replace')
    lines = [l for l in txt.splitlines() if 'reanudando' in l or 'ya completada' in l or l.startswith('[ddqn] DONE')]
    ok = bool(re.search(r'DONE 100\.0% \| steps 18k/18k', txt))
    out += ['```'] + [l[:200] for l in lines] + ['```', '',
            f"**Resultado:** {'OK: el contador global llega a 18k/18k tras reanudar' if ok else 'REVISAR: el contador global no llegó a 18k/18k'}", '']
    return out


def section_audit(d: Path) -> List[str]:
    out = ['## 2. Auditoría de la observación', '']
    p = d / 'obs_audit.json'
    if not p.is_file():
        return out + ['Sin datos.', '']
    a = json.loads(p.read_text(encoding='utf-8'))
    out += [f"Muestras: {a['muestras']} pasos reales; dimensiones: {a['dims']}.", '',
            '| Medida | Valor |', '|---|---|',
            f"| Valores que nunca cambian | {a['constantes_total']} ({100 * a['constantes_total'] / a['dims']:.1f} %) |",
            f"| Valores con \\|x\\| > 1,5 | {a['fuera_1_5_total']} |",
            f"| Valores con \\|x\\| > 5 | {a['fuera_5_total']} |",
            f"| Máximo \\|x\\| | {a['max_abs_global']:.2f} |", '',
            '| Sección | Dims | Constantes | > 1,5 | > 5 | Máx \\|x\\| | Densidad |', '|---|---|---|---|---|---|---|']
    for name, s in a['secciones'].items():
        out.append(f"| {name} | {s['dims']} | {s['constantes']} | {s['fuera_1_5']} | {s['fuera_5']} | "
                   f"{s['max_abs']:.2f} | {s['densidad']:.3f} |")
    return out + ['']


def section_probe(d: Path) -> List[str]:
    out = ['## 3. Probe del preview (¿cuánto importa el preview?)', '']
    p = d / 'preview_value_probe.json'
    if not p.is_file():
        return out + ['Sin datos.', '']
    pr = json.loads(p.read_text(encoding='utf-8'))
    out += ['```json', json.dumps(pr, indent=1, ensure_ascii=False)[:3000], '```', '']
    return out


def section_preview(d: Path) -> List[str]:
    out = ['## 4. Entrenamiento del preview PPO', '']
    rep = d / 'preview_ppo' / 'reporte.txt'
    if not rep.is_file():
        return out + ['Sin datos (¿falló el entrenamiento?). DDQN usó `/team 1234`.', '']
    out += ['```', rep.read_text(encoding='utf-8', errors='replace')[:3500], '```', '']
    used = d / 'preview_used.txt'
    if used.is_file():
        out += [f"Checkpoint usado por DDQN en tuning y ablación: `{used.read_text().strip()}`", '']
    return out


def section_tuning(d: Path) -> List[str]:
    out = ['## 5. Tuning de DDQN (400k pasos, semilla 0, validación sin exploración)', '',
           'Validación: 50 partidas BO1 por rival, rivales deterministas, mismas semillas en todas '
           'las configuraciones. "Final" = validación al paso 400k; "mejor" = la mejor de la corrida.', '',
           '| Config | Media final | random | solo | joint | expectimax | Mejor media (paso) | Q medio | Error TD | Pasos/s |',
           '|---|---|---|---|---|---|---|---|---|---|']
    best = None
    for ck in sorted((d / 'tune').glob('*')):
        ev = _evals(ck)
        if not ev:
            out.append(f'| {ck.name} | sin datos | | | | | | | | |')
            continue
        fin, top = ev[-1], max(ev, key=lambda e: e['mean_wr'])
        ln = _last_learner(ck)
        cells = _eval_cells(fin)
        out.append(f"| {ck.name} | **{100 * fin['mean_wr']:.1f} %** | " + ' | '.join(cells) +
                   f" | {100 * top['mean_wr']:.1f} % ({top['step'] // 1000}k) | {ln.get('mean_q', float('nan')):.2f} | "
                   f"{ln.get('td_error', float('nan')):.3f} | {_rate(ck)} |")
        if best is None or fin['mean_wr'] > best[1]:
            best = (ck.name, fin['mean_wr'])
    if best:
        out += ['', f'**Mejor configuración (media final):** `{best[0]}` con {100 * best[1]:.1f} %. '
                'Una sola semilla: diferencias de pocos puntos son ruido (≈ ±3,5 puntos en la media de 200 partidas).']
    return out + ['']


def section_ablation(d: Path) -> List[str]:
    out = ['## 6. Ablación codificador × simetría (400k pasos, 3 semillas)', '',
           '| Celda | Semillas | Media final (± desv. entre semillas) | Por semilla |', '|---|---|---|---|']
    cells: Dict[str, List[float]] = {}
    for ck in sorted((d / 'ablation').glob('*')):
        ev = _evals(ck)
        if not ev:
            continue
        cell = re.sub(r'_s\d+$', '', ck.name)
        cells.setdefault(cell, []).append(ev[-1]['mean_wr'])
    for cell, vals in sorted(cells.items()):
        sd = statistics.stdev(vals) if len(vals) > 1 else float('nan')
        out.append(f"| {cell} | {len(vals)} | {100 * statistics.mean(vals):.1f} % ± {100 * sd:.1f} | "
                   + ', '.join(f'{100 * v:.1f}' for v in vals) + ' |')
    if not cells:
        out.append('| sin datos | | | |')
    return out + ['']


def main(argv=None) -> int:
    d = Path((argv or sys.argv[1:])[0]).expanduser()
    lines = ['# Resumen previo al kick-off de DDQN', '',
             f"Directorio: `{d}`. Generado automáticamente al terminar la cola.", '']
    q = d / 'queue.log'
    if q.is_file():
        lines += ['```', q.read_text(encoding='utf-8', errors='replace')[-4000:], '```', '']
    for fn in (section_resume, section_audit, section_probe, section_preview, section_tuning,
               section_ablation):
        try:
            lines += fn(d)
        except Exception as exc:  # un fallo en una sección no debe impedir el resto
            lines += [f'Error al resumir ({fn.__name__}): {exc!r}', '']
    (d / 'kickoff_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
