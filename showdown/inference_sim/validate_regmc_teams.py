"""Valida equipos MetaVGC contra reglas Reg M-C vía TeamValidator de Pokémon Showdown.

Equivalente offline del ValidarEquipo de start.py (popup |utm|), usando:
  node pokemon-showdown → TeamValidator.get(format).validateTeam(...)

Uso:
  python -m showdown.inference_sim.validate_regmc_teams
  python -m showdown.inference_sim.validate_regmc_teams --purge
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from showdown.inference_sim.paths import POKEMON_SHOWDOWN_ROOT, REPO_ROOT
from showdown.inference_sim.team_factory import (
    TeamMeta,
    _JSON_PATH,
    load_scraped_teams,
    to_showdown_export,
)

DEFAULT_FORMAT = os.environ.get(
    'ROBUST_E2E_FORMAT',
    os.environ.get('POKEMON_MODE', 'gen9championsvgc2026regmcbo3'),
)
REPORT_PATH = REPO_ROOT / 'logs' / 'regmc_team_validation.txt'
_BATCH_JS = POKEMON_SHOWDOWN_ROOT / 'tools' / 'validate_regmc_batch.js'


def _ensure_batch_script() -> Path:
    if _BATCH_JS.is_file():
        return _BATCH_JS
    raise FileNotFoundError(
        f'No está {_BATCH_JS}. Se necesita pokemon-showdown con tools/validate_regmc_batch.js'
    )


def validate_exports(
    rows: List[Dict[str, str]],
    *,
    format_id: str = DEFAULT_FORMAT,
) -> List[Dict]:
    """Valida exports Showdown. rows = [{name, text}, ...]."""
    script = _ensure_batch_script()
    if not (POKEMON_SHOWDOWN_ROOT / 'dist' / 'sim' / 'team-validator.js').is_file():
        raise FileNotFoundError(
            f'Falta build de PS en {POKEMON_SHOWDOWN_ROOT / "dist"}. '
            'Ejecuta: node build en pokemon-showdown'
        )
    payload = json.dumps(rows, ensure_ascii=False).encode('utf-8')
    proc = subprocess.run(
        ['node', str(script), format_id],
        input=payload,
        capture_output=True,
        cwd=str(POKEMON_SHOWDOWN_ROOT),
    )
    if not proc.stdout:
        err = (proc.stderr or b'').decode('utf-8', errors='replace')
        raise RuntimeError(
            f'validate_regmc_batch sin stdout (code={proc.returncode}): {err[:800]}'
        )
    out = json.loads(proc.stdout.decode('utf-8'))
    if not isinstance(out, list):
        raise RuntimeError('Respuesta de validate_regmc_batch no es una lista')
    return out


def validate_team_metas(
    teams: Optional[List[TeamMeta]] = None,
    *,
    format_id: str = DEFAULT_FORMAT,
) -> Tuple[List[TeamMeta], List[Tuple[TeamMeta, List[str]]]]:
    """Devuelve (ok_teams, [(bad_team, problems), ...])."""
    pool = list(teams) if teams is not None else load_scraped_teams()
    rows = [{'name': t.name, 'text': to_showdown_export(t)} for t in pool]
    results = validate_exports(rows, format_id=format_id)
    by_name = {r.get('name'): r for r in results}
    ok: List[TeamMeta] = []
    bad: List[Tuple[TeamMeta, List[str]]] = []
    for t in pool:
        hit = by_name.get(t.name)
        if hit is None:
            bad.append((t, ['missing validation result']))
            continue
        problems = list(hit.get('problems') or [])
        if hit.get('ok') and not problems:
            ok.append(t)
        else:
            bad.append((t, problems or ['invalid']))
    return ok, bad


def write_report(
    ok: List[TeamMeta],
    bad: List[Tuple[TeamMeta, List[str]]],
    *,
    format_id: str,
    path: Path = REPORT_PATH,
) -> Path:
    lines = [
        '=== Reg M-C team validation (Showdown TeamValidator) ===',
        f'format: {format_id}',
        f'total: {len(ok) + len(bad)}',
        f'ok: {len(ok)}',
        f'invalid: {len(bad)}',
        '',
    ]
    if bad:
        lines.append('--- invalid ---')
        for t, problems in bad:
            lines.append(f'* {t.name}')
            for p in problems:
                lines.append(f'  - {p}')
            lines.append('')
    else:
        lines.append('All teams valid for format.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def purge_invalid_from_json(
    bad_names: set,
    *,
    json_path: Path = _JSON_PATH,
) -> int:
    """Elimina del JSON los equipos inválidos. Devuelve cuántos quitó."""
    if not bad_names or not json_path.is_file():
        return 0
    raw = json.loads(json_path.read_text(encoding='utf-8'))
    if not isinstance(raw, list):
        return 0
    kept = []
    removed = 0
    for row in raw:
        name = row.get('name') or row.get('slug') or ''
        if name in bad_names:
            removed += 1
            continue
        kept.append(row)
    if removed:
        json_path.write_text(
            json.dumps(kept, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
    return removed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Valida metavgc_regmc_teams.json contra Reg M-C (Showdown TeamValidator)'
    )
    parser.add_argument('--format', default=DEFAULT_FORMAT)
    parser.add_argument(
        '--purge',
        action='store_true',
        help='Quita del JSON los equipos inválidos',
    )
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    ok, bad = validate_team_metas(format_id=args.format)
    report = write_report(ok, bad, format_id=args.format)
    if not args.quiet:
        print(
            f'REGMC_VALIDATE format={args.format} total={len(ok) + len(bad)} '
            f'ok={len(ok)} invalid={len(bad)} report={report}',
            flush=True,
        )
        for t, problems in bad[:30]:
            print(f'  FAIL {t.name}: {problems[0] if problems else "invalid"}', flush=True)
        if len(bad) > 30:
            print(f'  ... +{len(bad) - 30} more', flush=True)

    if args.purge and bad:
        n = purge_invalid_from_json({t.name for t, _ in bad})
        if not args.quiet:
            print(f'Purged {n} teams from {_JSON_PATH}', flush=True)

    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
