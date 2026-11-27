"""Resumen de las series jugadas en la web de retos BO3 (``logs/online_web/bo3_*.json``).

Escribe ``logs/online_web/resumen.md``: totales por agente (series y partidas) y una fila por serie con sus
replays. Lo llama ``scripts/online/aws/collect_battles.sh`` después de traer los archivos de la EC2.

    python scripts/online/summarize_web.py [--dir logs/online_web]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]


def load_series(folder: Path) -> List[dict]:
    out = []
    for p in sorted(folder.glob('bo3_*.json')):
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        m = re.search(r'_(\d{8})_(\d{6})\.json$', p.name)
        d['_fecha'] = (f'{m[1][:4]}-{m[1][4:6]}-{m[1][6:]} {m[2][:2]}:{m[2][2:4]}' if m else '')
        d['_archivo'] = p.name
        out.append(d)
    out.sort(key=lambda d: d['_fecha'])
    return out


def replay_links(folder: Path, games: List[dict]) -> str:
    links = []
    for i, g in enumerate(games, 1):
        num = str(g.get('sala', '')).rsplit('-', 1)[-1]
        hits = sorted((folder / 'replays').glob(f'*{num}.html')) if num else []
        links.append(f'[P{i}](replays/{hits[0].name.replace(" ", "%20")})' if hits else f'P{i}')
    return ' '.join(links)


def summarize(series: List[dict], folder: Path) -> str:
    tot: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for d in series:
        t = tot[d.get('agente', '?')]
        t['series'] += 1
        if d.get('estado') == 'terminada':
            t['ganadas' if d.get('serie') == 'ganada' else 'perdidas' if d.get('serie') == 'perdida' else 'otras'] += 1
        else:
            t['incompletas'] += 1
        t['p_ganadas'] += int(d.get('victorias', 0))
        t['p_perdidas'] += int(d.get('derrotas', 0))
    rivals = {str(d.get('rival', '')).lower() for d in series}
    lines = [
        '# Web de retos BO3: series contra personas',
        '',
        f'Generado por `scripts/online/summarize_web.py` a partir de {len(series)} serie(s) contra '
        f'{len(rivals)} usuario(s) distinto(s). Las victorias y derrotas son del agente.',
        '',
        '## Totales por agente',
        '',
        '| Agente | Series | Ganadas | Perdidas | Incompletas | Partidas G–P | % partidas |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for agent, t in sorted(tot.items()):
        n = t['p_ganadas'] + t['p_perdidas']
        pct = f'{100 * t["p_ganadas"] / n:.0f} %' if n else '–'
        lines.append(f'| {agent} | {t["series"]} | {t["ganadas"]} | {t["perdidas"]} | {t["incompletas"]} | '
                     f'{t["p_ganadas"]}–{t["p_perdidas"]} | {pct} |')
    lines += ['', '## Series', '',
              '| Fecha (UTC) | Rival | Agente | Resultado | Estado | Turnos | Replays |',
              '|---|---|---|---|---|---|---|']
    for d in series:
        games = d.get('partidas') or []
        turns = ', '.join(str(g.get('turnos', '')) for g in games)
        lines.append(f'| {d["_fecha"]} | {d.get("rival", "")} | {d.get("agente", "")} | '
                     f'{d.get("victorias", 0)}–{d.get("derrotas", 0)} ({d.get("serie", "")}) | '
                     f'{d.get("estado", "")} | {turns} | {replay_links(folder, games)} |')
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=str(REPO / 'logs' / 'online_web'))
    args = ap.parse_args(argv)
    folder = Path(args.dir)
    series = load_series(folder)
    (folder / 'resumen.md').write_text(summarize(series, folder), encoding='utf-8')
    print(f'== resumen: {len(series)} serie(s) -> {folder / "resumen.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
