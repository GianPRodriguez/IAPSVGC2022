"""Barrido automatizado de N_sim para MCTS+PUCT y MCTS–Nash.

Deja correr sin agente Cursor (patrón similar a ``run_expectimax_tune``).

Ejemplos::

  python -m showdown.inference_sim.run_mcts_sims_sweep --dry-run
  python -m showdown.inference_sim.run_mcts_sims_sweep --agent both ^
      --player-b greedy_joint --seed 10 --teams 40 --start-ps
  python -m showdown.inference_sim.run_mcts_sims_sweep --agent mcts_puct ^
      --sims 50,100,200,400 --delta-wr-pp 1.5 --time-ratio 1.8 --seed 10

Env por punto:
  mcts_puct      → MCTS_SIMS
  mcts_puct_nash → MCTS_NASH_SIMS

Salidas (por agente)::

  logs/mcts_sims_sweep_{agent}.json       — resumen final + puntos
  logs/mcts_sims_sweep_{agent}.txt        — tabla + recomendación
  logs/mcts_sims_sweep_{agent}_tracking.txt
  logs/mcts_sims_sweep_{agent}_history.jsonl  — un evento por cambio de N_sim
  logs/mcts_sims_sweep_{agent}_progress.json  — estado vivo del barrido
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from showdown.inference_sim.mcts_sims_sweep import (
    PlateauCriteria,
    SweepPoint,
    build_tracking_log,
    default_sims_grid,
    env_key_for_agent,
    format_sweep_table,
    format_tracking_text,
    frozen_hyperparams,
    step_transition,
    suggest_plateau,
)
from showdown.inference_sim.paths import REPO_ROOT

REPORT_PATH = REPO_ROOT / 'logs' / 'robust_e2e_report.txt'
OUT_DIR = REPO_ROOT / 'logs'

# Espejo de PLAYER_B_CHOICES sin importar poke-env (dry-run / docs).
PLAYER_B_CHOICES = (
    'random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian', 'ddqn', 'ppo',
    'mcts_puct', 'mcts_puct_nash', 'qd_ppo', 'qd_pga', 'qd_ppga',
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_sims_list(raw: str, agent: str) -> Tuple[int, ...]:
    s = (raw or '').strip()
    if not s:
        return default_sims_grid(agent)
    vals = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        return default_sims_grid(agent)
    return tuple(sorted(set(vals)))


def _paths(agent: str) -> Dict[str, Path]:
    stem = OUT_DIR / f'mcts_sims_sweep_{agent}'
    return {
        'json': stem.with_suffix('.json'),
        'txt': stem.with_suffix('.txt'),
        'tracking': OUT_DIR / f'mcts_sims_sweep_{agent}_tracking.txt',
        'history': OUT_DIR / f'mcts_sims_sweep_{agent}_history.jsonl',
        'progress': OUT_DIR / f'mcts_sims_sweep_{agent}_progress.json',
    }


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def parse_metrics_from_report(text: str) -> Dict[str, Any]:
    def _f(pat: str, default: float = 0.0) -> float:
        m = re.search(pat, text, re.I | re.M)
        return float(m.group(1)) if m else default

    def _i(pat: str, default: int = 0) -> int:
        m = re.search(pat, text, re.I | re.M)
        return int(m.group(1)) if m else default

    def _s(pat: str, default: str = '') -> str:
        m = re.search(pat, text, re.I | re.M)
        return str(m.group(1)).strip().upper() if m else default

    return {
        'wr_raw': _f(r'win_rate_player_a:\s*([\d.]+)'),
        'pair_wr': _f(r'win_rate_pairs:\s*([\d.]+)'),
        'elapsed_s': _f(r'elapsed_s:\s*([\d.]+)'),
        'wins': _i(r'player_a_wins:\s*(\d+)'),
        'losses': _i(r'player_b_wins:\s*(\d+)'),
        'pair_invalid': _i(r'pair_invalid:\s*(\d+)'),
        'need_fix': _s(r'NEED_FIX[=:]\s*(YES|NO)', 'NO'),
        'sync_ok': _s(r'sync_ok[=:]\s*(YES|NO)', 'YES'),
        'inf_ok': _s(r'inf_ok[=:]\s*(YES|NO)', 'YES'),
    }


def run_robust_point(
    *,
    agent: str,
    n_sims: int,
    player_b: str,
    teams: int,
    seed: int,
    workers: int,
    restart_ps: bool,
    start_ps: bool,
) -> Tuple[int, SweepPoint]:
    env = os.environ.copy()
    env_key = env_key_for_agent(agent)
    env[env_key] = str(int(n_sims))
    if agent == 'mcts_puct':
        env['MCTS_SEED'] = str(int(seed))
    else:
        env['MCTS_NASH_SEED'] = str(int(seed))
        env.setdefault('MCTS_SEED', str(int(seed)))

    cmd = [
        sys.executable, '-m', 'showdown.inference_sim.run_robust_e2e',
        '--teams', str(int(teams)),
        '--seed', str(int(seed)),
        '--player-a', agent,
        '--player-b', player_b,
        '--workers', str(int(workers)),
        '--procs', str(int(os.environ.get('ROBUST_E2E_PROCS', '1') or 1)),
    ]
    if restart_ps:
        cmd.append('--restart-ps')
    elif start_ps:
        cmd.append('--start-ps')

    print(f'\n>>> {env_key}={n_sims} {" ".join(cmd)}', flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True)
    wall = time.perf_counter() - t0
    report = REPORT_PATH.read_text(encoding='utf-8') if REPORT_PATH.is_file() else ''
    m = parse_metrics_from_report(report)
    elapsed = float(m['elapsed_s']) if m['elapsed_s'] > 0 else wall
    point = SweepPoint(
        n_sims=int(n_sims),
        wr_raw=float(m['wr_raw']),
        pair_wr=float(m['pair_wr']),
        elapsed_s=float(elapsed),
        wins=int(m['wins']),
        losses=int(m['losses']),
        pair_invalid=int(m['pair_invalid']),
        need_fix=str(m['need_fix']),
        sync_ok=str(m['sync_ok']),
        inf_ok=str(m['inf_ok']),
    )
    return int(proc.returncode), point


def _log_step_event(
    *,
    paths: Dict[str, Path],
    run_id: str,
    agent: str,
    step_i: int,
    total: int,
    prev: Optional[SweepPoint],
    cur: SweepPoint,
    criteria: PlateauCriteria,
    player_b: str,
    seed: int,
    teams: int,
    rc: Optional[int],
    interim_decision_n: Optional[int],
) -> Dict[str, Any]:
    """Append history + rewrite progress after each N_sim change."""
    tr = step_transition(prev, cur, prefer_metric=criteria.prefer_metric)
    event = {
        'ts_utc': _utc(),
        'event': 'hyperparam_step',
        'run_id': run_id,
        'agent': agent,
        'step_index': step_i,
        'steps_total': total,
        'env_key': env_key_for_agent(agent),
        'hyperparams_set': {
            env_key_for_agent(agent): int(cur.n_sims),
            'seed': int(seed),
        },
        'frozen_search_hyperparams': frozen_hyperparams(agent),
        'transition': tr,
        'point': cur.to_dict(),
        'return_code': rc,
        'interim_recommended_n_sims': interim_decision_n,
        'player_b': player_b,
        'teams': teams,
        'criteria': asdict(criteria),
    }
    _append_jsonl(paths['history'], event)

    progress = {
        'ts_utc': _utc(),
        'run_id': run_id,
        'agent': agent,
        'status': 'running' if step_i < total else 'done',
        'step_index': step_i,
        'steps_total': total,
        'current_n_sims': int(cur.n_sims),
        'n_sims_prev': tr.get('n_sims_prev'),
        'delta_metric_pp': tr.get('delta_metric_pp'),
        'time_ratio': tr.get('time_ratio'),
        'wr_raw': cur.wr_raw,
        'pair_wr': cur.pair_wr,
        'elapsed_s': cur.elapsed_s,
        'interim_recommended_n_sims': interim_decision_n,
        'history_path': str(paths['history']),
    }
    _write_json(paths['progress'], progress)

    d_m = tr.get('delta_metric_pp')
    tratio = tr.get('time_ratio')
    print(
        f'  TRACK step {step_i}/{total} '
        f'N_sim {tr.get("n_sims_prev")}->{cur.n_sims} '
        f'WR_raw={cur.wr_raw:.1f}% '
        f'delta_metric={d_m if d_m is None else f"{d_m:+.2f}pp"} '
        f'time_ratio={tratio if tratio is None else f"{tratio:.2f}x"} '
        f'interim_rec={interim_decision_n}',
        flush=True,
    )
    return event


def _finalize_artifacts(
    *,
    paths: Dict[str, Path],
    run_id: str,
    agent: str,
    player_b: str,
    teams: int,
    seed: int,
    workers: int,
    sims: Sequence[int],
    criteria: PlateauCriteria,
    points: Sequence[SweepPoint],
    decision,
    table: str,
    rcs: Sequence[Optional[int]],
    dry_run: bool,
) -> Dict[str, Any]:
    tracking = build_tracking_log(
        points,
        decision,
        agent=agent,
        criteria=criteria,
        player_b=player_b,
        seed=seed,
        teams=teams,
        run_id=run_id,
        dry_run=dry_run,
    )
    tracking_txt = format_tracking_text(tracking)
    paths['tracking'].parent.mkdir(parents=True, exist_ok=True)
    paths['tracking'].write_text(tracking_txt, encoding='utf-8')

    _append_jsonl(paths['history'], {
        'ts_utc': _utc(),
        'event': 'plateau_decision',
        'run_id': run_id,
        'agent': agent,
        'recommended_n_sims': decision.recommended_n_sims,
        'reason': decision.reason,
        'plateau_index': decision.plateau_index,
        'criteria': asdict(criteria),
        'frozen_search_hyperparams': frozen_hyperparams(agent),
    })

    payload = {
        'time_utc': _utc(),
        'run_id': run_id,
        'agent': agent,
        'dry_run': dry_run,
        'player_b': player_b,
        'teams': teams,
        'seed': seed,
        'workers': workers,
        'env_key': env_key_for_agent(agent),
        'varying_hyperparam': 'n_sims',
        'frozen_search_hyperparams': frozen_hyperparams(agent),
        'criteria': asdict(criteria),
        'sims_grid': list(sims),
        'points': [p.to_dict() for p in decision.points],
        'tracking': tracking,
        'recommended_n_sims': decision.recommended_n_sims,
        'reason': decision.reason,
        'return_codes': list(rcs),
        'table': table,
        'artifacts': {k: str(v) for k, v in paths.items()},
    }
    _write_json(paths['json'], payload)
    paths['txt'].write_text(
        f'=== MCTS N_sim sweep ({agent}) ===\n'
        f'time_utc: {_utc()}\n'
        f'run_id: {run_id}\n'
        f'rival: {player_b}\n'
        f'teams: {teams}\n'
        f'seed: {seed}\n'
        f'varying: n_sims via {env_key_for_agent(agent)}\n'
        f'frozen: {frozen_hyperparams(agent)}\n'
        f'criteria: delta_wr_pp={criteria.delta_wr_pp} '
        f'time_ratio={criteria.time_ratio} metric={criteria.prefer_metric}\n'
        f'tracking: {paths["tracking"]}\n'
        f'history: {paths["history"]}\n\n'
        + table
        + '\n'
        + tracking_txt,
        encoding='utf-8',
    )
    _write_json(paths['progress'], {
        'ts_utc': _utc(),
        'run_id': run_id,
        'agent': agent,
        'status': 'done',
        'recommended_n_sims': decision.recommended_n_sims,
        'reason': decision.reason,
        'artifacts': {k: str(v) for k, v in paths.items()},
    })
    try:
        from showdown.inference_sim.tune_plots import plot_mcts_sweep_run, read_jsonl
        rec_wr = None
        for row in read_jsonl(paths['history']):
            if row.get('event') != 'hyperparam_step':
                continue
            pt = row.get('point') or {}
            n_val = pt.get('n_sims', row.get('current_n_sims'))
            if n_val is not None and int(n_val) == int(decision.recommended_n_sims):
                wr = pt.get('wr_raw')
                if wr is not None:
                    rec_wr = float(wr) / 100.0 if float(wr) > 1.5 else float(wr)
                break
        fig_info = plot_mcts_sweep_run(
            agent=agent,
            history_path=paths['history'],
            run_id=run_id,
            recommended_score=rec_wr,
            recommended_n=int(decision.recommended_n_sims),
        )
        payload['figures'] = fig_info
    except Exception as exc:  # pragma: no cover
        payload['figures_error'] = repr(exc)

    print(tracking_txt, flush=True)
    fig_note = ''
    if payload.get('figures'):
        latest = payload['figures'].get('latest_figures') or {}
        fig_note = f' figure={latest.get("svg") or latest.get("png")}'
    print(
        f'DONE sweep agent={agent} recommend N_sim={decision.recommended_n_sims} '
        f'report={paths["txt"]} tracking={paths["tracking"]} '
        f'history={paths["history"]}{fig_note}',
        flush=True,
    )
    return payload


def sweep_agent(
    agent: str,
    *,
    sims: Sequence[int],
    player_b: str,
    teams: int,
    seed: int,
    workers: int,
    restart_ps: bool,
    start_ps: bool,
    criteria: PlateauCriteria,
    dry_run: bool,
) -> Dict[str, Any]:
    run_id = f'{agent}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}_{uuid.uuid4().hex[:8]}'
    paths = _paths(agent)
    # Nueva corrida: truncar history de este agente (un archivo por sweep activo).
    if paths['history'].is_file():
        paths['history'].unlink()

    print(
        f'SWEEP agent={agent} run_id={run_id} sims={list(sims)} rival={player_b} '
        f'teams={teams} seed={seed} delta_wr_pp={criteria.delta_wr_pp} '
        f'time_ratio={criteria.time_ratio} metric={criteria.prefer_metric}',
        flush=True,
    )
    _append_jsonl(paths['history'], {
        'ts_utc': _utc(),
        'event': 'sweep_start',
        'run_id': run_id,
        'agent': agent,
        'sims_grid': list(sims),
        'player_b': player_b,
        'seed': seed,
        'teams': teams,
        'criteria': asdict(criteria),
        'env_key': env_key_for_agent(agent),
        'varying_hyperparam': 'n_sims',
        'frozen_search_hyperparams': frozen_hyperparams(agent),
        'dry_run': dry_run,
    })

    if dry_run:
        fake = [
            SweepPoint(
                n_sims=n,
                wr_raw=40.0 + 0.02 * n,
                pair_wr=40.0 + 0.015 * n,
                elapsed_s=10.0 * (n / 50.0),
            )
            for n in sims
        ]
        if len(fake) >= 2:
            fake = list(fake)
            last = fake[-1]
            fake[-1] = SweepPoint(
                n_sims=last.n_sims,
                wr_raw=fake[-2].wr_raw + 0.2,
                pair_wr=fake[-2].pair_wr + 0.2,
                elapsed_s=fake[-2].elapsed_s * max(criteria.time_ratio, 2.0),
            )
        points: List[SweepPoint] = []
        prev: Optional[SweepPoint] = None
        for i, pt in enumerate(fake, start=1):
            points.append(pt)
            interim = suggest_plateau(points, criteria).recommended_n_sims
            _log_step_event(
                paths=paths,
                run_id=run_id,
                agent=agent,
                step_i=i,
                total=len(fake),
                prev=prev,
                cur=pt,
                criteria=criteria,
                player_b=player_b,
                seed=seed,
                teams=teams,
                rc=None,
                interim_decision_n=interim,
            )
            prev = pt
        decision = suggest_plateau(points, criteria)
        table = format_sweep_table(points, decision, prefer_metric=criteria.prefer_metric)
        print(table, flush=True)
        return _finalize_artifacts(
            paths=paths,
            run_id=run_id,
            agent=agent,
            player_b=player_b,
            teams=teams,
            seed=seed,
            workers=workers,
            sims=sims,
            criteria=criteria,
            points=points,
            decision=decision,
            table=table,
            rcs=[None] * len(points),
            dry_run=True,
        )

    points = []
    rcs: List[int] = []
    prev = None
    total = len(list(sims))
    for i, n in enumerate(sims, start=1):
        rc, pt = run_robust_point(
            agent=agent,
            n_sims=int(n),
            player_b=player_b,
            teams=teams,
            seed=seed,
            workers=workers,
            restart_ps=restart_ps,
            start_ps=start_ps,
        )
        rcs.append(rc)
        points.append(pt)
        interim = suggest_plateau(points, criteria).recommended_n_sims
        _log_step_event(
            paths=paths,
            run_id=run_id,
            agent=agent,
            step_i=i,
            total=total,
            prev=prev,
            cur=pt,
            criteria=criteria,
            player_b=player_b,
            seed=seed,
            teams=teams,
            rc=rc,
            interim_decision_n=interim,
        )
        prev = pt
        restart_ps = False
        start_ps = False

    decision = suggest_plateau(points, criteria)
    table = format_sweep_table(points, decision, prefer_metric=criteria.prefer_metric)
    print(table, flush=True)
    return _finalize_artifacts(
        paths=paths,
        run_id=run_id,
        agent=agent,
        player_b=player_b,
        teams=teams,
        seed=seed,
        workers=workers,
        sims=sims,
        criteria=criteria,
        points=points,
        decision=decision,
        table=table,
        rcs=rcs,
        dry_run=False,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description='Barrido N_sim MCTS open-loop / Nash (meseta WR vs tiempo)',
    )
    ap.add_argument(
        '--agent',
        default='both',
        choices=('mcts_puct', 'mcts_puct_nash', 'both'),
        help='Qué agente barrer (default: both)',
    )
    ap.add_argument(
        '--player-b',
        default='greedy_joint',
        help=f'Rival (default greedy_joint). Opciones: {", ".join(PLAYER_B_CHOICES)}',
    )
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--teams', type=int, default=40, help='Equipos robust e2e (par)')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--procs', type=int, default=1,
                    help='Procesos del robust E2E por punto (servidor: ~24; 0 = automático)')
    ap.add_argument(
        '--sims',
        default='',
        help='Lista CSV de N_sim (vacío = default por agente)',
    )
    ap.add_argument(
        '--delta-wr-pp',
        type=float,
        default=1.5,
        help='Mejora mínima de WR (pp) para justificar el siguiente N',
    )
    ap.add_argument(
        '--time-ratio',
        type=float,
        default=1.8,
        help='Factor de tiempo del siguiente punto para declarar meseta',
    )
    ap.add_argument(
        '--metric',
        default='wr_raw',
        choices=('wr_raw', 'pair_wr'),
        help='Métrica principal del criterio de meseta',
    )
    ap.add_argument('--start-ps', action='store_true')
    ap.add_argument('--restart-ps', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(list(argv) if argv is not None else None)
    os.environ['ROBUST_E2E_PROCS'] = str(int(args.procs))

    rival = str(args.player_b).strip().lower().replace('-', '_')
    if rival not in PLAYER_B_CHOICES:
        print(
            f'player-b inválido: {args.player_b!r}. '
            f'Opciones: {", ".join(PLAYER_B_CHOICES)}',
            file=sys.stderr,
        )
        return 2

    criteria = PlateauCriteria(
        delta_wr_pp=float(args.delta_wr_pp),
        time_ratio=float(args.time_ratio),
        prefer_metric=str(args.metric),
    )

    agents: List[str]
    if args.agent == 'both':
        agents = ['mcts_puct', 'mcts_puct_nash']
    else:
        agents = [str(args.agent)]

    restart = bool(args.restart_ps)
    start = bool(args.start_ps) and not restart

    for i, agent in enumerate(agents):
        sims = _parse_sims_list(args.sims, agent)
        sweep_agent(
            agent,
            sims=sims,
            player_b=rival,
            teams=int(args.teams),
            seed=int(args.seed),
            workers=int(args.workers),
            restart_ps=restart and i == 0,
            start_ps=start and i == 0,
            criteria=criteria,
            dry_run=bool(args.dry_run),
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
