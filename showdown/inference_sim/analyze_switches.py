"""Fase 2c de expectimax: frecuencia de cambios voluntarios y bucles de cambio.

Lee ``PS_SWITCH_LOG=<ruta>`` (``<ruta>.<pid>``, una línea por cambio voluntario) y los
logs de ``run_robust_e2e`` (líneas ``gN:tT``) para la duración de los juegos.

    python -m showdown.inference_sim.analyze_switches <prefijo_switch_log> [run1.log ...]
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict
from typing import Dict, List


def load_switches(prefix: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(glob.glob(prefix + '*')):
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def game_turns(log_paths: List[str]) -> List[int]:
    turns: List[int] = []
    for p in log_paths:
        with open(p, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                turns += [int(t) for t in re.findall(r'g\d+:t(\d+)[WLD]', line)]
    return turns


def summarize(rows: List[dict], turns: List[int]) -> List[str]:
    by_game: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_game[r['tag']].append(r)
    n_sw = len(rows)
    # Rebote: el que entra salió (por cambio voluntario) en los 2 turnos anteriores del mismo juego.
    bounce = 0
    for evs in by_game.values():
        evs.sort(key=lambda r: r['turn'])
        for i, r in enumerate(evs):
            if any(e['out'] == r['in'] and 0 <= r['turn'] - e['turn'] <= 2 for e in evs[:i]):
                bounce += 1
    out = [f'cambios voluntarios: {n_sw} en {len(by_game)} juegos con algún cambio']
    if turns:
        out.append(f'juegos: {len(turns)}  turnos medios por juego: {sum(turns) / len(turns):.1f}')
        out.append(f'cambios voluntarios por juego (todos los juegos): {n_sw / len(turns):.2f}')
    if n_sw:
        out.append(f'rebotes (vuelve a entrar quien salió hace <= 2 turnos): {bounce} '
                   f'({100.0 * bounce / n_sw:.1f} % de los cambios)')
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rows = load_switches(argv[0]) if argv else []
    for line in summarize(rows, game_turns(argv[1:])):
        print(line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
