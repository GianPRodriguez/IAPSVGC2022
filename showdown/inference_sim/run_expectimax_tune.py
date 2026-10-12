"""Tune autónomo expectimax (voluntary switch) vs greedy_joint — sin agente Cursor.

Flujo (créditos Cursor = 0 mientras corre):
  1) Baseline estable (overrides vacíos) teams=N seed fija.
  2) Fase CORE: solo 3 knobs (enable / max_switch_hybrids / switch_v_scale).
     Acepta si no empeora (pair WR, desempate WR_raw); si empeora → revert.
  3) Fase SECONDARY: solo si CORE quedó con enable=True y no peor que baseline;
     un knobs scoring a la vez.
  4) Al final: summary DONE/history + NEED_FIX acumulado (una sola señal).

Uso (dejar corriendo ~8–12 h)::

  python -m showdown.inference_sim.run_expectimax_tune --hours 10 --teams 100 --seed 10

Solo dry-run de candidatos::

  python -m showdown.inference_sim.run_expectimax_tune --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from showdown.battle_bots.expectimax_bayesian.expectimax import (
    ExpectimaxConfig,
    clear_active_overrides,
    load_expectimax_config,
    read_active_overrides,
    write_active_overrides,
)
from showdown.battle_bots.expectimax_bayesian.config_history import record_param_change
from showdown.inference_sim.paths import REPO_ROOT

REPORT_PATH = REPO_ROOT / 'logs' / 'robust_e2e_report.txt'
FIXPLAN_PATH = REPO_ROOT / 'logs' / 'robust_e2e_fixplan.txt'
TUNE_HISTORY = REPO_ROOT / 'logs' / 'expectimax_tune_history.jsonl'
TUNE_SUMMARY = REPO_ROOT / 'logs' / 'expectimax_tune_summary.txt'

# Tolerancia "no empeora" (pp). 0 = estricto.
_PAIR_EPS_PP = 0.0
_RAW_EPS_PP = 0.5


@dataclass
class RunMetrics:
    wr_raw: float
    pair_wr: float
    pair_a: int
    pair_b: int
    pair_draw: int
    pair_invalid: int
    need_fix: str
    sync_ok: str
    inf_ok: str
    wins: int
    losses: int
    elapsed_s: float
    c2_mean: Optional[float] = None

    def score_key(self) -> Tuple[float, float, float]:
        """Mayor es mejor: pair_wr, wr_raw, c2."""
        c2 = float(self.c2_mean) if self.c2_mean is not None else -999.0
        return (self.pair_wr, self.wr_raw, c2)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + '\n')


def parse_metrics_from_report(text: str) -> RunMetrics:
    def _f(pat: str, default: float = 0.0) -> float:
        m = re.search(pat, text, re.I | re.M)
        return float(m.group(1)) if m else default

    def _i(pat: str, default: int = 0) -> int:
        m = re.search(pat, text, re.I | re.M)
        return int(m.group(1)) if m else default

    def _s(pat: str, default: str = '') -> str:
        m = re.search(pat, text, re.I | re.M)
        return str(m.group(1)).strip() if m else default

    c2_m = re.search(r'reward_c2_mean:\s*([+\-]?[\d.]+)', text)
    return RunMetrics(
        wr_raw=_f(r'win_rate_player_a:\s*([\d.]+)%'),
        pair_wr=_f(r'win_rate_pairs:\s*([\d.]+)%'),
        pair_a=_i(r'pair_a_sweeps:\s*(\d+)'),
        pair_b=_i(r'pair_b_sweeps:\s*(\d+)'),
        pair_draw=_i(r'pair_draws:\s*(\d+)'),
        pair_invalid=_i(r'pair_invalid:\s*(\d+)'),
        need_fix=_s(r'NEED_FIX[=:]\s*(YES|NO)', 'NO'),
        sync_ok=_s(r'sync_ok[=:]\s*(YES|NO)', 'YES'),
        inf_ok=_s(r'inf_ok[=:]\s*(YES|NO)', 'YES'),
        wins=_i(r'player_a_wins:\s*(\d+)'),
        losses=_i(r'player_b_wins:\s*(\d+)'),
        elapsed_s=_f(r'elapsed_s:\s*([\d.]+)'),
        c2_mean=float(c2_m.group(1)) if c2_m else None,
    )


def parse_metrics_from_fixplan(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in ('NEED_FIX', 'sync_ok', 'inf_ok'):
        # fixplan uses NEED_FIX=NO or NEED_FIX: NO
        m = re.search(rf'{key}[=:]\s*(YES|NO)', text, re.I)
        if m:
            out[key] = m.group(1).upper()
    return out


def not_worse(cand: RunMetrics, base: RunMetrics) -> bool:
    """Aceptar si pair_wr no baja (eps) y wr_raw no baja más de RAW_EPS."""
    if cand.pair_wr + _PAIR_EPS_PP < base.pair_wr:
        return False
    if cand.wr_raw + _RAW_EPS_PP < base.wr_raw:
        return False
    return True


def core_candidates() -> List[Dict[str, Any]]:
    """Fase 1: 3 knobs núcleo. hybrids 1–2, scale pequeño."""
    out: List[Dict[str, Any]] = []
    for hybrids in (1, 2):
        for scale in (0.10, 0.15, 0.20, 0.25, 0.30):
            out.append({
                'enable_voluntary_switch': True,
                'max_switch_hybrids': hybrids,
                'switch_v_scale': scale,
            })
    return out


def secondary_candidates(best_core: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fase 2: un knobs scoring a la vez, encima del mejor CORE."""
    base = dict(best_core)
    trials: List[Dict[str, Any]] = []
    for stay in (0.45, 0.55):
        d = dict(base)
        d['switch_min_stay_p_ko'] = stay
        trials.append(d)
    for bonus in (40.0, 70.0):
        d = dict(base)
        d['ko_free_switch_bonus'] = bonus
        trials.append(d)
    for w in (0.9, 1.3):
        d = dict(base)
        d['entry_punish_weight'] = w
        trials.append(d)
    for w in (0.8, 1.2):
        d = dict(base)
        d['entry_ability_weight'] = w
        trials.append(d)
    for w in (0.8, 1.2):
        d = dict(base)
        d['threat_relief_weight'] = w
        trials.append(d)
    return trials


def run_robust_e2e(
    *,
    teams: int,
    seed: int,
    restart_ps: bool,
    player_b: str,
    workers: int,
) -> Tuple[int, str]:
    cmd = [
        sys.executable, '-m', 'showdown.inference_sim.run_robust_e2e',
        '--teams', str(teams),
        '--seed', str(seed),
        '--player-a', 'expectimax_bayesian',
        '--player-b', player_b,
        '--workers', str(workers),
    ]
    if restart_ps:
        cmd.append('--restart-ps')
    print(f'\n>>> {" ".join(cmd)}', flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=False,
        text=True,
    )
    report = REPORT_PATH.read_text(encoding='utf-8') if REPORT_PATH.is_file() else ''
    return int(proc.returncode), report


def apply_overrides(overrides: Dict[str, Any], *, note: str) -> None:
    write_active_overrides(overrides)
    try:
        before = asdict(ExpectimaxConfig())
        after = asdict(load_expectimax_config())
        changed = {
            k: {'from': before.get(k), 'to': after.get(k)}
            for k in overrides
            if before.get(k) != after.get(k)
        }
        record_param_change(note, changed=changed, result='tune_loop')
    except Exception as exc:
        print(f'[tune] config_history warn: {exc!r}', flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Expectimax voluntary-switch tune loop (no Cursor credits)')
    ap.add_argument('--teams', type=int, default=100)
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--hours', type=float, default=10.0, help='Techo de wall-clock (default 10h)')
    ap.add_argument('--player-b', default='greedy_joint')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--skip-baseline', action='store_true',
                    help='Usa métricas del último report como baseline (no re-corre OFF)')
    ap.add_argument('--no-restart-ps', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--max-iters', type=int, default=0, help='0 = sin límite aparte de --hours')
    args = ap.parse_args(list(argv) if argv is not None else None)

    deadline = time.time() + float(args.hours) * 3600.0
    restart = not bool(args.no_restart_ps)

    core = core_candidates()
    print(
        f'TUNE start utc={_utc()} teams={args.teams} seed={args.seed} '
        f'hours={args.hours} core_cands={len(core)} '
        f'secondary~={len(secondary_candidates({"enable_voluntary_switch": True, "max_switch_hybrids": 1, "switch_v_scale": 0.15}))}',
        flush=True,
    )
    if args.dry_run:
        for i, c in enumerate(core, 1):
            print(f'  core[{i}] {c}')
        print('dry-run OK (no e2e)')
        return 0

    need_fix_any = False
    history_rows: List[Dict[str, Any]] = []

    # --- Baseline: overrides OFF (dataclass estable) ---
    clear_active_overrides()
    if args.skip_baseline and REPORT_PATH.is_file():
        base_report = REPORT_PATH.read_text(encoding='utf-8')
        baseline = parse_metrics_from_report(base_report)
        print(
            f'[baseline] SKIP run → from last report '
            f'pair={baseline.pair_wr:.1f}% raw={baseline.wr_raw:.1f}%',
            flush=True,
        )
    else:
        rc, base_report = run_robust_e2e(
            teams=args.teams, seed=args.seed, restart_ps=restart,
            player_b=args.player_b, workers=args.workers,
        )
        baseline = parse_metrics_from_report(base_report)
        fp = FIXPLAN_PATH.read_text(encoding='utf-8') if FIXPLAN_PATH.is_file() else ''
        fp_m = parse_metrics_from_fixplan(fp)
        if fp_m.get('NEED_FIX') == 'YES' or baseline.need_fix == 'YES':
            need_fix_any = True
            baseline.need_fix = 'YES'
        row = {
            'ts_utc': _utc(),
            'kind': 'baseline',
            'accepted': True,
            'overrides': {},
            'metrics': asdict(baseline),
            'rc': rc,
        }
        _append_jsonl(TUNE_HISTORY, row)
        history_rows.append(row)
        print(
            f'[baseline] pair={baseline.pair_wr:.1f}% raw={baseline.wr_raw:.1f}% '
            f'NEED_FIX={baseline.need_fix}',
            flush=True,
        )

    best_overrides: Dict[str, Any] = {}
    best_metrics = baseline
    iters = 0

    def _time_left() -> bool:
        return time.time() < deadline

    def _iter_ok() -> bool:
        if args.max_iters and iters >= int(args.max_iters):
            return False
        return _time_left()

    def evaluate(overrides: Dict[str, Any], *, phase: str, label: str) -> Optional[RunMetrics]:
        nonlocal need_fix_any, iters, best_overrides, best_metrics
        if not _iter_ok():
            return None
        iters += 1
        apply_overrides(overrides, note=f'tune {phase}: {label}')
        rc, report = run_robust_e2e(
            teams=args.teams, seed=args.seed, restart_ps=restart,
            player_b=args.player_b, workers=args.workers,
        )
        metrics = parse_metrics_from_report(report)
        fp = FIXPLAN_PATH.read_text(encoding='utf-8') if FIXPLAN_PATH.is_file() else ''
        if parse_metrics_from_fixplan(fp).get('NEED_FIX') == 'YES' or metrics.need_fix == 'YES':
            need_fix_any = True
            metrics.need_fix = 'YES'
        accepted = not_worse(metrics, best_metrics)
        # También exigir no peor que baseline absoluta
        accepted = accepted and not_worse(metrics, baseline)
        row = {
            'ts_utc': _utc(),
            'kind': phase,
            'label': label,
            'accepted': accepted,
            'overrides': overrides,
            'metrics': asdict(metrics),
            'vs_best_pair_delta': metrics.pair_wr - best_metrics.pair_wr,
            'vs_best_raw_delta': metrics.wr_raw - best_metrics.wr_raw,
            'rc': rc,
        }
        _append_jsonl(TUNE_HISTORY, row)
        history_rows.append(row)
        print(
            f'[{phase}] {label} pair={metrics.pair_wr:.1f}% '
            f'(Δ{metrics.pair_wr - best_metrics.pair_wr:+.1f}) '
            f'raw={metrics.wr_raw:.1f}% NEED_FIX={metrics.need_fix} '
            f'→ {"ACCEPT" if accepted else "REVERT"}',
            flush=True,
        )
        if accepted:
            best_overrides = dict(overrides)
            best_metrics = metrics
            write_active_overrides(best_overrides)
        else:
            # Volver al mejor aceptado (o vacío = estable)
            if best_overrides:
                write_active_overrides(best_overrides)
            else:
                clear_active_overrides()
        return metrics

    # --- Fase CORE ---
    for i, cand in enumerate(core, 1):
        if not _iter_ok():
            print('[tune] time/iter budget → stop CORE', flush=True)
            break
        label = (
            f'hyb={cand["max_switch_hybrids"]} '
            f'scale={cand["switch_v_scale"]}'
        )
        evaluate(cand, phase='core', label=f'{i}/{len(core)} {label}')

    # --- Fase SECONDARY ---
    if best_overrides.get('enable_voluntary_switch'):
        secs = secondary_candidates(best_overrides)
        for i, cand in enumerate(secs, 1):
            if not _iter_ok():
                print('[tune] time/iter budget → stop SECONDARY', flush=True)
                break
            # etiqueta: knobs que difieren del best_core
            diff = {
                k: cand[k] for k in cand
                if cand.get(k) != best_overrides.get(k)
            }
            evaluate(cand, phase='secondary', label=f'{i}/{len(secs)} {diff}')
    else:
        print(
            '[tune] CORE no dejó voluntary ON sin empeorar → skip SECONDARY',
            flush=True,
        )

    # --- Persistencia final ---
    if best_overrides:
        write_active_overrides(best_overrides)
    else:
        clear_active_overrides()

    lines = [
        '=== Expectimax tune summary (agent: read once) ===',
        f'utc: {_utc()}',
        f'teams: {args.teams}',
        f'seed: {args.seed}',
        f'player_b: {args.player_b}',
        f'hours_budget: {args.hours}',
        f'iters: {iters}',
        f'NEED_FIX_ANY: {"YES" if need_fix_any else "NO"}',
        '',
        '--- baseline ---',
        f'pair_wr: {baseline.pair_wr:.1f}%',
        f'wr_raw: {baseline.wr_raw:.1f}%',
        f'c2_mean: {baseline.c2_mean}',
        '',
        '--- best ---',
        f'overrides: {json.dumps(best_overrides, sort_keys=True)}',
        f'pair_wr: {best_metrics.pair_wr:.1f}% (Δ{best_metrics.pair_wr - baseline.pair_wr:+.1f})',
        f'wr_raw: {best_metrics.wr_raw:.1f}% (Δ{best_metrics.wr_raw - baseline.wr_raw:+.1f})',
        f'c2_mean: {best_metrics.c2_mean}',
        '',
        '--- history (accepted only) ---',
    ]
    for r in history_rows:
        if not r.get('accepted'):
            continue
        m = r.get('metrics') or {}
        lines.append(
            f"  {r.get('kind')}: pair={m.get('pair_wr')}% raw={m.get('wr_raw')}% "
            f"ov={r.get('overrides')}"
        )
    lines += [
        '',
        f'history_jsonl: {TUNE_HISTORY}',
        f'active_overrides: {REPO_ROOT / "showdown" / "battle_bots" / "expectimax_bayesian" / "ACTIVE_OVERRIDES.json"}',
        '',
        'ACTION_FOR_AGENT:',
    ]
    if need_fix_any:
        lines.append(
            '  NEED_FIX=YES occurred in at least one DONE — open Cursor and '
            'run: python -m showdown.inference_sim.robust_fixplan then «corrige el fixplan robusto».'
        )
    else:
        lines.append('  NEED_FIX never YES — do not open agent for fixplan.')
    if best_overrides.get('enable_voluntary_switch'):
        lines.append('  Voluntary switch kept via ACTIVE_OVERRIDES.json (dataclass defaults still OFF).')
    else:
        lines.append('  No voluntary improvement found; overrides cleared (stable OFF).')

    TUNE_SUMMARY.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)

    try:
        from showdown.inference_sim.tune_plots import plot_expectimax_tune_run
        fig_info = plot_expectimax_tune_run(
            TUNE_HISTORY,
            run_id=f'expectimax_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}',
        )
        latest = (fig_info or {}).get('latest_figures') or {}
        if latest.get('svg'):
            print(f'figure={latest["svg"]}', flush=True)
    except Exception as exc:  # pragma: no cover
        print(f'[tune] figures warn: {exc!r}', flush=True)

    # Exit: 0 ok, 2 si hubo NEED_FIX (señal única para el humano/agente)
    return 2 if need_fix_any else 0


if __name__ == '__main__':
    raise SystemExit(main())
