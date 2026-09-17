"""Historial de ExpectimaxConfig: changelog manual + snapshot en batches E2E.

Changelog (humano / evolución de knobs):
  showdown/battle_bots/expectimax_bayesian/CONFIG_HISTORY.md
  + CONFIG_HISTORY.jsonl (misma carpeta)

Runs E2E (automático si A o B es expectimax_bayesian):
  logs/expectimax_config_runs.jsonl

Registrar un cambio de parámetros tras editar ExpectimaxConfig::

  python -m showdown.battle_bots.expectimax_bayesian.config_history \\
      "anti-joint: menos threat, menos pivot panic" \\
      --changed switch_min_stay_p_ko=0.50,threat_weight=0.55,loss_weight=650,save_weight=140,syn_weight=0.55
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from showdown.battle_bots.expectimax_bayesian.expectimax import ExpectimaxConfig

_PKG = Path(__file__).resolve().parent
HISTORY_MD = _PKG / 'CONFIG_HISTORY.md'
HISTORY_JSONL = _PKG / 'CONFIG_HISTORY.jsonl'
# Repo root: expectimax_bayesian → battle_bots → showdown → <repo>
_REPO = _PKG.parents[2]
RUNS_JSONL = _REPO / 'logs' / 'expectimax_config_runs.jsonl'

# Snapshot conocido previo al primer tuning documentado (baseline en código).
BASELINE_PRE_TUNING: Dict[str, Any] = {
    'deadline_ms': 700.0,
    'phase1_budget_ms': 50.0,
    'phase2_end_ms': 680.0,
    'k_profiles': 5,
    'max_worlds': 16,
    'max_our_pairs': 36,
    'soft_ko_bonus': 1000.0,
    'enable_voluntary_switch': True,
    'max_switch_hybrids': 12,
    'loss_weight': 1000.0,
    'save_weight': 220.0,
    'chip_soft_cap': 30.0,
    'switch_min_stay_p_ko': 0.35,
    'overkill_waste_weight': 140.0,
    'ko_free_switch_bonus': 55.0,
    'entry_punish_weight': 1.15,
    'threat_weight': 0.75,
    'hp_rel_weight': 0.15,
    'syn_weight': 0.35,
    'condemned_chip_discount': 0.70,
    'condemned_lasting_boost': 1.35,
    'threat_relief_weight': 1.0,
    'entry_ability_weight': 1.0,
    'opp_policy_greedy_weight': 0.75,
    'opp_policy_control_weight': 0.15,
    'opp_policy_use_real_damage': False,
}


def config_to_dict(cfg: Optional[ExpectimaxConfig] = None) -> Dict[str, Any]:
    if cfg is None:
        from showdown.battle_bots.expectimax_bayesian.expectimax import load_expectimax_config
        cfg = load_expectimax_config()
    return asdict(cfg)


def config_diff(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    keys = sorted(set(before) | set(after))
    out: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        b, a = before.get(k), after.get(k)
        if b != a:
            out[k] = {'from': b, 'to': a}
    return out


def format_config_block(
    cfg: Optional[ExpectimaxConfig] = None,
    *,
    title: str = 'expectimax_config',
) -> str:
    d = config_to_dict(cfg)
    lines = [f'--- {title} ---']
    for f in fields(ExpectimaxConfig):
        lines.append(f'{f.name}: {d[f.name]!r}')
    return '\n'.join(lines)


def uses_expectimax(*players: str) -> bool:
    return any(str(p or '').strip() == 'expectimax_bayesian' for p in players)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + '\n')


def record_param_change(
    note: str,
    *,
    changed: Optional[Mapping[str, Any]] = None,
    result: str = '',
    cfg: Optional[ExpectimaxConfig] = None,
) -> Dict[str, Any]:
    """Añade entrada al changelog (JSONL + Markdown)."""
    now = datetime.now(timezone.utc).isoformat()
    current = config_to_dict(cfg)
    diff = dict(changed) if changed else config_diff(BASELINE_PRE_TUNING, current)
    # Si `changed` es mapa plano valor→nuevo, normalizar a from/to cuando se pueda.
    norm_diff: Dict[str, Any] = {}
    for k, v in diff.items():
        if isinstance(v, dict) and 'to' in v:
            norm_diff[k] = v
        else:
            norm_diff[k] = {
                'from': BASELINE_PRE_TUNING.get(k),
                'to': v,
            }

    row = {
        'ts_utc': now,
        'kind': 'param_change',
        'note': note.strip(),
        'result': (result or '').strip(),
        'changed': norm_diff,
        'config': current,
    }
    append_jsonl(HISTORY_JSONL, row)

    md_lines = [
        '',
        f'## {now[:10]} — {note.strip()}',
        '',
        f'- **UTC:** `{now}`',
    ]
    if result:
        md_lines.append(f'- **Resultado / contexto:** {result.strip()}')
    if norm_diff:
        md_lines.append('- **Cambios:**')
        for k, pair in sorted(norm_diff.items()):
            md_lines.append(f"  - `{k}`: {pair.get('from')!r} → {pair.get('to')!r}")
    else:
        md_lines.append('- **Cambios:** (snapshot completo; sin diff explícito)')
    md_lines.append('')
    md_lines.append('```')
    md_lines.append(format_config_block(cfg))
    md_lines.append('```')
    md_lines.append('')

    if not HISTORY_MD.exists():
        HISTORY_MD.write_text(
            '# ExpectimaxConfig — historial de parámetros\n\n'
            'Evolución de knobs hacia el comportamiento deseado '
            '(p.ej. superar `greedy_joint`).\n\n'
            'Registrar: '
            '`python -m showdown.battle_bots.expectimax_bayesian.config_history '
            '"nota" --changed k=v,...`\n',
            encoding='utf-8',
        )
    with HISTORY_MD.open('a', encoding='utf-8') as fh:
        fh.write('\n'.join(md_lines))
    return row


def record_e2e_run(
    *,
    player_a: str,
    player_b: str,
    teams: int,
    wins: int,
    losses: int,
    wr_raw: float,
    pair_a: int = 0,
    pair_b: int = 0,
    pair_draw: int = 0,
    need_fix: str = '',
    sync_ok: str = '',
    inf_ok: str = '',
    cfg: Optional[ExpectimaxConfig] = None,
) -> Optional[Dict[str, Any]]:
    """Snapshot automático tras robust E2E si hay expectimax en el matchup."""
    if not uses_expectimax(player_a, player_b):
        return None
    now = datetime.now(timezone.utc).isoformat()
    row = {
        'ts_utc': now,
        'kind': 'e2e_run',
        'player_a': player_a,
        'player_b': player_b,
        'teams': int(teams),
        'wins': int(wins),
        'losses': int(losses),
        'wr_raw': float(wr_raw),
        'pair_a': int(pair_a),
        'pair_b': int(pair_b),
        'pair_draw': int(pair_draw),
        'need_fix': need_fix,
        'sync_ok': sync_ok,
        'inf_ok': inf_ok,
        'config': config_to_dict(cfg),
    }
    append_jsonl(RUNS_JSONL, row)
    return row


def _parse_changed(spec: str) -> Dict[str, Any]:
    """Parse 'a=1,b=0.5,c=True' → dict de valores nuevos."""
    out: Dict[str, Any] = {}
    for part in (spec or '').split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise SystemExit(f'changed inválido (falta =): {part!r}')
        k, raw = part.split('=', 1)
        k, raw = k.strip(), raw.strip()
        if raw.lower() in ('true', 'false'):
            out[k] = raw.lower() == 'true'
        else:
            try:
                out[k] = int(raw) if raw.isdigit() or (
                    raw.startswith('-') and raw[1:].isdigit()
                ) else float(raw)
            except ValueError:
                out[k] = raw
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description='Registrar cambio de ExpectimaxConfig')
    p.add_argument(
        'note',
        nargs='?',
        default='',
        help='Por qué cambian los knobs',
    )
    p.add_argument(
        '--changed',
        default='',
        help='Lista k=v (valores nuevos). Diff from/to vs baseline si falta from.',
    )
    p.add_argument('--result', default='', help='Contexto WR / batch / hipótesis')
    p.add_argument(
        '--show',
        action='store_true',
        help='Solo imprimir config actual (no escribir)',
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.show:
        print(format_config_block())
        print(f'history_md: {HISTORY_MD}')
        print(f'history_jsonl: {HISTORY_JSONL}')
        print(f'runs_jsonl: {RUNS_JSONL}')
        return 0

    if not (args.note or '').strip():
        p.error('note requerida (salvo --show)')

    changed = _parse_changed(args.changed) if args.changed else None
    # Enriquecer from/to con baseline o config anterior no disponible → baseline.
    changed_ft = None
    if changed:
        cur = config_to_dict()
        changed_ft = {}
        for k, v in changed.items():
            changed_ft[k] = {
                'from': BASELINE_PRE_TUNING.get(k, cur.get(k)),
                'to': v,
            }

    row = record_param_change(
        args.note,
        changed=changed_ft or changed,
        result=args.result,
    )
    print(f'Registrado: {HISTORY_MD.name} + {HISTORY_JSONL.name}')
    print(json.dumps({'ts': row['ts_utc'], 'changed': row['changed']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
