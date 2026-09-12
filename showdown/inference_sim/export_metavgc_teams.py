"""Exporta equipos MetaVGC Reg M-C a teams/ para uso offline."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from showdown.inference_sim.team_factory import TeamMeta, active_teams, to_showdown_export

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / 'teams' / 'teams' / 'gen9' / 'champions_regmc'


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', '', name)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return (cleaned or 'team')[:80]


def export_teams(out_dir: Path | None = None, teams: List[TeamMeta] | None = None) -> Path:
    dest = out_dir or DEFAULT_OUT
    dest.mkdir(parents=True, exist_ok=True)
    pool = teams if teams is not None else active_teams()
    for i, meta in enumerate(pool, start=1):
        fname = f'{i:02d}_{_safe_filename(meta.name)}.txt'
        path = dest / fname
        path.write_text(to_showdown_export(meta) + '\n', encoding='utf-8')
    # Índice legible
    index = dest / 'INDEX.txt'
    lines = [f'{i:02d}\t{t.name}\t{t.source_url}' for i, t in enumerate(pool, start=1)]
    index.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return dest


if __name__ == '__main__':
    out = export_teams()
    print(f'Exported {len(active_teams())} teams -> {out}')
