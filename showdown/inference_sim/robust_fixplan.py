"""Fixplan del E2E robusto: solo hay trabajo si el reporte falló.

El batch escribe logs/robust_e2e_fixplan.txt:
  NEED_FIX=NO  → no gastes créditos Cursor; no abras el agente.
  NEED_FIX=YES → el usuario puede pedir manualmente "corrige el fixplan robusto".

No auto-corrige código. Solo resume fallos + archivos candidatos.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

from showdown.inference_sim.paths import REPO_ROOT

REPORT_PATH = REPO_ROOT / 'logs' / 'robust_e2e_report.txt'
FIXPLAN_PATH = REPO_ROOT / 'logs' / 'robust_e2e_fixplan.txt'

# Dónde tocar según tipo de fallo (orden de prioridad para el agente)
_SYNC_TARGETS = [
    'showdown/battle_modifier.py',
    'showdown/battle.py',
    'showdown/inference_sim/battle_state_check.py',
    'showdown/inference_sim/validated_greedy_player.py',
    'showdown/inference_sim/feeder.py',
]
_INFER_TARGETS = [
    'showdown/stat_inference.py',
    'showdown/battle_modifier.py',
    'showdown/inference_sim/soundness.py',
    'showdown/inference_sim/feeder.py',
]
_HEUR_TARGETS = [
    'showdown/battle_bots/greedy_solo/greedy.py',
    'showdown/battle_bots/greedy_solo/team_preview.py',
    'showdown/battle_bots/greedy_solo/pokeenv_player.py',
    'showdown/battle_bots/greedy_joint/joint.py',
    'showdown/battle_bots/greedy_joint/main.py',
    'showdown/battle_bots/expectimax_bayesian/expectimax.py',
    'showdown/battle_bots/expectimax_bayesian/main.py',
    'showdown/battle_bots/expectimax_bayesian/pokeenv_player.py',
    'showdown/inference_sim/run_robust_e2e.py',
    'showdown/inference_sim/bo3_runtime.py',
    'showdown/tools/calc_core.js',
]


def _flag(report: str, key: str) -> Optional[str]:
    m = re.search(rf'^{re.escape(key)}:\s*(.+)$', report, flags=re.M)
    return m.group(1).strip() if m else None


def parse_need_fix(report_text: str) -> bool:
    # Preferir sync_ok / inf_ok (polaridad = *_ok); fallback a nombres viejos.
    sync_ok = (_flag(report_text, 'sync_ok') or '').upper()
    inf_ok = (_flag(report_text, 'inf_ok') or '').upper()
    if not sync_ok:
        # state_mismatch=YES ⇒ sync malo
        mm = (_flag(report_text, 'state_mismatch') or '').upper()
        sync_ok = 'NO' if mm == 'YES' else ('YES' if mm == 'NO' else '')
    if not inf_ok:
        inf_ok = (_flag(report_text, 'inference_ok') or '').upper()
    errors = _flag(report_text, 'errors') or '0'
    try:
        err_n = int(errors.split()[0])
    except Exception:
        err_n = 0
    return sync_ok == 'NO' or inf_ok == 'NO' or err_n > 0


def _section_lines(report: str, header: str, limit: int = 25) -> List[str]:
    lines = report.splitlines()
    out: List[str] = []
    grab = False
    for ln in lines:
        if ln.startswith('--- ') and grab:
            break
        if ln.strip() == header:
            grab = True
            continue
        if grab and ln.strip():
            out.append(ln)
            if len(out) >= limit:
                break
    return out


def build_fixplan(report_text: str) -> str:
    need = parse_need_fix(report_text)
    sync_ok = (_flag(report_text, 'sync_ok') or '').upper()
    if not sync_ok:
        mm = (_flag(report_text, 'state_mismatch') or '').upper()
        sync_ok = 'NO' if mm == 'YES' else ('YES' if mm == 'NO' else '?')
    inf_ok = (
        _flag(report_text, 'inf_ok')
        or _flag(report_text, 'inference_ok')
        or '?'
    )
    flags_add = _flag(report_text, 'NEED_FLAGS_ADD') or '?'
    wins = (
        _flag(report_text, 'player_a_wins')
        or _flag(report_text, 'heuristic_wins')
        or '?'
    )
    losses = (
        _flag(report_text, 'player_b_wins')
        or _flag(report_text, 'random_wins')
        or '?'
    )
    errors = _flag(report_text, 'errors') or '?'
    mm = _flag(report_text, 'state_mismatches') or '?'
    fails = (
        _flag(report_text, 'fail_bo3_inf')
        or _flag(report_text, 'soundness_failures')
        or '?'
    )
    fail_matchup_inf = (
        _flag(report_text, 'fail_matchup_inf')
        or _flag(report_text, 'fail_inference')
        or '?'
    )
    pa = _flag(report_text, 'player_a') or 'greedy_solo'
    pb = _flag(report_text, 'player_b') or 'random'

    lines = [
        '=== Robust E2E Fixplan (manual Cursor fix) ===',
        f'NEED_FIX={"YES" if need else "NO"}',
        f'NEED_FLAGS_ADD: {flags_add}',
        f'sync_ok: {sync_ok}',
        f'inf_ok: {inf_ok}',
        f'fail_matchup_inf: {fail_matchup_inf}',
        f'fail_bo3_inf: {fails}',
        f'player_a: {pa}',
        f'player_b: {pb}',
        f'player_a_wins: {wins}',
        f'player_b_wins: {losses}',
        f'heuristic_wins: {wins}',
        f'random_wins: {losses}',
        f'errors: {errors}',
        f'state_mismatches: {mm}',
        f'soundness_failures: {fails}',
        '',
    ]

    if not need:
        lines += [
            'STATUS: clean (sync/inference/errors)',
            'ACTION: no tocar código por NEED_FIX; no abrir agente Cursor.',
        ]
        if str(flags_add).upper() == 'YES':
            lines += [
                'NOTE: NEED_FLAGS_ADD=YES es informativo (gaps calc/Battle).',
                'ACTION_FLAGS: solo si el usuario pide "agrega flags de inferencia".',
                'TARGETS_FLAGS:',
                '- showdown/inference_sim/inference_flag_audit.py',
                '- showdown/stat_inference.py',
                '- showdown/battle.py',
                '- showdown/battle_modifier.py',
            ]
            flag_samples = _section_lines(
                report_text, '--- inference flags (Battle storage / calc wiring) ---', 15
            )
            if flag_samples:
                lines.append('')
                lines.append('FLAGS_SAMPLES:')
                lines.extend(flag_samples[:12])
        lines.append(f'Source report: {REPORT_PATH}')
        return '\n'.join(lines) + '\n'

    lines.append('STATUS: failures present — wait for user to ask for a fix in chat.')
    lines.append('USER_PROMPT_HINT: "corrige el fixplan robusto" / "fix robust e2e"')
    lines.append('')
    lines.append('AGENT_RULES:')
    lines.append('- Leer SOLO este fixplan + samples abajo (no volcar reporte completo).')
    lines.append('- Corregir el mínimo de archivos listados en TARGETS.')
    lines.append('- Tras el fix: smoke `python -m showdown.inference_sim.run_robust_e2e --teams 4`')
    lines.append('- NO lanzar --teams 100 salvo que el usuario lo pida.')
    lines.append('')

    targets: List[str] = []
    if str(sync_ok).upper() == 'NO':
        lines.append('KIND: sync (Battle vs PS local)')
        targets.extend(_SYNC_TARGETS)
    if str(inf_ok).upper() == 'NO':
        lines.append('KIND: inference (OTS / Stat Points)')
        targets.extend(_INFER_TARGETS)
    try:
        if int(str(errors).split()[0]) > 0:
            lines.append('KIND: runtime/heuristic errors')
            targets.extend(_HEUR_TARGETS)
    except Exception:
        pass

    # unique preserve order
    seen = set()
    uniq = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    lines.append('')
    lines.append('TARGETS:')
    for t in uniq:
        lines.append(f'- {t}')

    tips = _section_lines(report_text, '--- improvements / notes for next runs ---', 10)
    if tips:
        lines.append('')
        lines.append('TIPS:')
        lines.extend(tips[:8])

    sync_samples = _section_lines(report_text, '--- sync mismatch samples (capped) ---', 15)
    if sync_samples:
        lines.append('')
        lines.append('SYNC_SAMPLES:')
        lines.extend(sync_samples[:12])

    inf_samples = _section_lines(report_text, '--- inference failure samples (capped) ---', 15)
    if inf_samples:
        lines.append('')
        lines.append('INFERENCE_SAMPLES:')
        lines.extend(inf_samples[:12])

    runtime_samples = _section_lines(report_text, '--- runtime error samples (capped) ---', 12)
    if runtime_samples:
        lines.append('')
        lines.append('RUNTIME_SAMPLES:')
        lines.extend(runtime_samples[:10])

    lines.append('')
    lines.append(f'Source report: {REPORT_PATH}')
    return '\n'.join(lines) + '\n'


def write_fixplan(report_text: str, path: Optional[Path] = None) -> Path:
    out = path or FIXPLAN_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    text = build_fixplan(report_text)
    out.write_text(text, encoding='utf-8')
    return out


def status_from_disk() -> str:
    if not REPORT_PATH.is_file():
        return 'NEED_FIX=UNKNOWN (no report yet)\n'
    text = REPORT_PATH.read_text(encoding='utf-8')
    plan = build_fixplan(text)
    write_fixplan(text)
    # Solo la línea clave para terminal / agente
    for ln in plan.splitlines():
        if ln.startswith('NEED_FIX='):
            return ln + f' fixplan={FIXPLAN_PATH}\n'
    return plan


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Estado / regeneración del fixplan robusto (sin batallas)'
    )
    parser.add_argument(
        '--refresh',
        action='store_true',
        help='Releer robust_e2e_report.txt y reescribir fixplan',
    )
    args = parser.parse_args(argv)
    if args.refresh or True:
        msg = status_from_disk()
        print(msg, end='')
        if FIXPLAN_PATH.is_file() and 'NEED_FIX=YES' in msg:
            return 1
        return 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
