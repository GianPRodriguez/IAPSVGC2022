"""Round-robin final de Conclusión (docs/conclusion_eval_protocol.md).

Uso:
  python -m showdown.inference_sim.run_conclusion_e2e --freeze-snapshot --seed 42
  python -m showdown.inference_sim.run_conclusion_e2e --write-pipelines
  python -m showdown.inference_sim.run_conclusion_e2e --seed 42 --start-ps
  python -m showdown.inference_sim.run_conclusion_e2e --smoke --seed 42 --start-ps
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from showdown.inference_sim.conclusion_eval.constants import (
    DEFAULT_N_PAIRS,
    DEFAULT_SNAPSHOT_N,
    OUT_DIR,
    PROGRESS_PATH,
    REPORT_PATH,
    RESULTS_PATH,
    SNAPSHOT_PATH,
)
from showdown.inference_sim.conclusion_eval.pipelines import (
    DEFAULT_PIPELINES,
    PipelineSpec,
    load_pipelines,
    save_pipelines,
)
from showdown.inference_sim.conclusion_eval.report import build_report_text, results_payload
from showdown.inference_sim.conclusion_eval.snapshot import (
    EvalSnapshot,
    build_snapshot,
    load_snapshot,
    materialize_pairs,
    save_snapshot,
)
from showdown.inference_sim.conclusion_eval.stats import (
    CrossResult,
    deterministic_cross_order,
    rank_pipelines,
)
from showdown.inference_sim.paths import BATTLE_FORMAT

for _name in ('poke-env', 'poke_env', 'asyncio'):
    logging.getLogger(_name).setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


async def _stop_players(*players) -> None:
    for pl in players:
        try:
            await pl.ps_client.stop_listening()
        except Exception:
            pass


async def _play_one(
    p1,
    p2,
    *,
    team_p1,
    team_p2,
    timeout_s: float,
) -> Tuple[Optional[bool], str]:
    from showdown.inference_sim.team_factory import to_showdown_export

    p1.update_team(to_showdown_export(team_p1))
    p2.update_team(to_showdown_export(team_p2))
    for pl, opp in ((p1, team_p2), (p2, team_p1)):
        try:
            from showdown.battle_bots.expectimax_bayesian.ots import (
                ots_sheet_from_team_meta,
            )
            if hasattr(pl, 'set_opp_ots_sheet'):
                pl.set_opp_ots_sheet(ots_sheet_from_team_meta(opp))
        except Exception:
            pass

    wins_before = int(getattr(p1, 'n_won_battles', 0) or 0)
    fin_before = int(getattr(p1, 'n_finished_battles', 0) or 0)
    try:
        await asyncio.wait_for(
            p1.battle_against(p2, n_battles=1),
            timeout=timeout_s,
        )
    except Exception as exc:
        return None, repr(exc)

    if int(getattr(p1, 'n_finished_battles', 0) or 0) <= fin_before:
        return None, 'no_finished_battle'
    return int(getattr(p1, 'n_won_battles', 0) or 0) > wins_before, ''


async def _play_with_retry(
    p1,
    p2,
    *,
    team_p1,
    team_p2,
    timeout_s: float,
) -> Tuple[Optional[bool], str, int]:
    won, err = await _play_one(
        p1, p2, team_p1=team_p1, team_p2=team_p2, timeout_s=timeout_s,
    )
    if won is not None:
        return won, '', 0
    won2, err2 = await _play_one(
        p1, p2, team_p1=team_p1, team_p2=team_p2, timeout_s=timeout_s,
    )
    if won2 is not None:
        return won2, '', 1
    return None, err2 or err or 'fail', 2


async def run_cross(
    pl_i: PipelineSpec,
    pl_j: PipelineSpec,
    *,
    i: int,
    j: int,
    team_pairs: Sequence[Tuple[Any, Any]],
    format_id: str,
    port: int,
    timeout_s: float,
    stamp: int,
    progress_cb=None,
) -> Tuple[CrossResult, List[Dict[str, Any]]]:
    from poke_env.ps_client import AccountConfiguration

    from showdown.inference_sim.paths import server_configuration_for_port
    from showdown.inference_sim.validated_greedy_player import make_opponent_player

    server = server_configuration_for_port(port)
    t0_a, t0_b = team_pairs[0]
    p1 = make_opponent_player(
        pl_i.combat,
        team=t0_a,
        account_configuration=AccountConfiguration(f'C{stamp}i{i}'[:18], None),
        battle_format=format_id,
        server_configuration=server,
    )
    p2 = make_opponent_player(
        pl_j.combat,
        team=t0_b,
        account_configuration=AccountConfiguration(f'C{stamp}j{j}'[:18], None),
        battle_format=format_id,
        server_configuration=server,
    )

    cr = CrossResult(i=i, j=j, id_i=pl_i.id, id_j=pl_j.id)
    failures: List[Dict[str, Any]] = []
    total = 2 * len(team_pairs)
    done = 0

    try:
        for k, (ta, tb) in enumerate(team_pairs):
            won, err, nfail = await _play_with_retry(
                p1, p2, team_p1=ta, team_p2=tb, timeout_s=timeout_s,
            )
            done += 1
            if won is None:
                cr.fails += 1
                failures.append({
                    'cross': f'{pl_i.id}_vs_{pl_j.id}',
                    'side': 'AB',
                    'pair_k': k,
                    'team_p1': ta.name,
                    'team_p2': tb.name,
                    'error': err,
                    'retries_failed': nfail,
                })
            else:
                cr.valid += 1
                cr.valid_ab += 1
                if won:
                    cr.wins_i += 1
                    cr.wins_i_ab += 1
                else:
                    cr.wins_j += 1
            if progress_cb:
                progress_cb(done, total, cr)

        for k, (ta, tb) in enumerate(team_pairs):
            won, err, nfail = await _play_with_retry(
                p1, p2, team_p1=tb, team_p2=ta, timeout_s=timeout_s,
            )
            done += 1
            if won is None:
                cr.fails += 1
                failures.append({
                    'cross': f'{pl_i.id}_vs_{pl_j.id}',
                    'side': 'BA',
                    'pair_k': k,
                    'team_p1': tb.name,
                    'team_p2': ta.name,
                    'error': err,
                    'retries_failed': nfail,
                })
            else:
                cr.valid += 1
                cr.valid_ba += 1
                if won:
                    cr.wins_i += 1
                    cr.wins_i_ba += 1
                else:
                    cr.wins_j += 1
            if progress_cb:
                progress_cb(done, total, cr)

            if (done % 25) == 0:
                try:
                    p1.reset_battles()
                    p2.reset_battles()
                except Exception:
                    pass
    finally:
        await _stop_players(p1, p2)

    return cr, failures


async def run_eval(
    *,
    seed: int,
    format_id: str,
    start_ps: bool,
    restart_ps: bool,
    timeout_s: float,
    port: int,
    snapshot: EvalSnapshot,
    pipelines: Sequence[PipelineSpec],
    smoke: bool,
    quiet: bool,
    progress_every: int,
) -> str:
    from showdown.inference_sim.ps_server import ensure_ps, ps_start_hint

    if not ensure_ps(start=start_ps or restart_ps, restart=restart_ps):
        raise RuntimeError(
            'Pokémon Showdown no responde. '
            f'Arranca: {ps_start_hint()}  o usa --start-ps / --restart-ps'
        )

    team_pairs = materialize_pairs(snapshot)
    order = deterministic_cross_order(len(pipelines), seed=seed)
    crosses: List[CrossResult] = []
    failures: List[Dict[str, Any]] = []
    stamp = int(time.time()) % 100000
    t0 = time.perf_counter()

    if not quiet:
        print(
            f'CONCLUSION start pipelines={len(pipelines)} crosses={len(order)} '
            f'pairs={len(team_pairs)} seed={seed} smoke={smoke}',
            flush=True,
        )

    for ci, (i, j) in enumerate(order):
        pl_i, pl_j = pipelines[i], pipelines[j]

        def _cb(done, total, cr, _ci=ci, _n=len(order), _i=pl_i.id, _j=pl_j.id):
            if quiet:
                return
            if done % max(1, progress_every) == 0 or done == total:
                print(
                    f'  cross[{_ci + 1}/{_n}] {_i} vs {_j} '
                    f'BO1 {done}/{total} W={cr.wins_i}-{cr.wins_j} E={cr.fails}',
                    flush=True,
                )

        cr, fails = await run_cross(
            pl_i,
            pl_j,
            i=i,
            j=j,
            team_pairs=team_pairs,
            format_id=format_id,
            port=port,
            timeout_s=timeout_s,
            stamp=stamp + ci,
            progress_cb=_cb,
        )
        crosses.append(cr)
        failures.extend(fails)

        elapsed = time.perf_counter() - t0
        _write_json(PROGRESS_PATH, {
            'done_crosses': ci + 1,
            'total_crosses': len(order),
            'last_cross': f'{pl_i.id}_vs_{pl_j.id}',
            'wins': f'{cr.wins_i}-{cr.wins_j}',
            'fails': cr.fails,
            'elapsed_s': round(elapsed, 2),
            'time_utc': datetime.now(timezone.utc).isoformat(),
        })
        if not quiet:
            print(
                f'  DONE cross {pl_i.id} vs {pl_j.id}: '
                f'{cr.wins_i}-{cr.wins_j} n={cr.valid} fail={cr.fails}',
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    ids = [p.id for p in pipelines]
    ranks = rank_pipelines(ids, crosses)
    report = build_report_text(
        seed=seed,
        format_id=format_id,
        snapshot_n=snapshot.snapshot_n,
        n_pairs=snapshot.n_pairs,
        pipelines=pipelines,
        crosses=crosses,
        ranks=ranks,
        elapsed_s=elapsed,
        smoke=smoke,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='utf-8')
    _write_json(
        RESULTS_PATH,
        results_payload(
            seed=seed,
            format_id=format_id,
            snapshot_n=snapshot.snapshot_n,
            n_pairs=snapshot.n_pairs,
            pipelines=pipelines,
            crosses=crosses,
            ranks=ranks,
            elapsed_s=elapsed,
            failures=failures,
            smoke=smoke,
        ),
    )

    top = ranks[0].pipeline_id if ranks else '-'
    print(
        f'DONE conclusion pipelines={len(pipelines)} crosses={len(crosses)} '
        f'top={top} fails={len(failures)} elapsed_s={elapsed:.1f} '
        f'report={REPORT_PATH}',
        flush=True,
    )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description='Round-robin final Conclusión (11 pipelines, snapshot A/B)',
    )
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--format', default=BATTLE_FORMAT)
    ap.add_argument('--snapshot-n', type=int, default=DEFAULT_SNAPSHOT_N)
    ap.add_argument('--pairs', type=int, default=DEFAULT_N_PAIRS)
    ap.add_argument(
        '--freeze-snapshot',
        action='store_true',
        help='Construir y guardar snapshot.json',
    )
    ap.add_argument(
        '--write-pipelines',
        action='store_true',
        help='Escribir pipelines.json default y salir (salvo que también corras eval)',
    )
    ap.add_argument('--snapshot', type=str, default='')
    ap.add_argument(
        '--smoke',
        action='store_true',
        help='Smoke: snapshot_n=4, pairs=2',
    )
    ap.add_argument(
        '--run',
        action='store_true',
        help='Ejecutar batallas (implícito con --smoke / --start-ps / --restart-ps)',
    )
    ap.add_argument('--start-ps', action='store_true')
    ap.add_argument('--restart-ps', action='store_true')
    ap.add_argument('--timeout', type=float, default=300.0)
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--progress-every', type=int, default=20)
    ap.add_argument('--export', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args(argv)

    if args.write_pipelines:
        path = save_pipelines(DEFAULT_PIPELINES)
        print(f'Wrote pipelines -> {path}', flush=True)

    want_run = bool(
        args.run or args.smoke or args.start_ps or args.restart_ps
    )
    want_freeze = bool(args.freeze_snapshot or (args.smoke and want_run))

    if args.write_pipelines and not want_freeze and not want_run:
        return 0

    snap_n = 4 if args.smoke else int(args.snapshot_n)
    n_pairs = 2 if args.smoke else int(args.pairs)
    snap_path = Path(args.snapshot) if args.snapshot else SNAPSHOT_PATH

    if args.export:
        from showdown.inference_sim.export_metavgc_teams import export_teams
        export_teams()

    if want_freeze or args.freeze_snapshot:
        snap = build_snapshot(
            seed=args.seed, snapshot_n=snap_n, n_pairs=n_pairs,
        )
        out = save_snapshot(snap, snap_path)
        print(
            f'Froze snapshot n={snap.snapshot_n} A={len(snap.pool_a)} '
            f'B={len(snap.pool_b)} pairs={len(snap.pairs)} -> {out}',
            flush=True,
        )
        if args.freeze_snapshot and not want_run:
            return 0

    if not want_run:
        print(
            'Nada que ejecutar. Usa --freeze-snapshot, --write-pipelines, '
            'o --run / --smoke / --start-ps para batallas.',
            flush=True,
        )
        return 0

    if not snap_path.is_file():
        print(
            f'No hay snapshot en {snap_path}. '
            'Corre: --freeze-snapshot --seed N',
            file=sys.stderr,
        )
        return 2

    snapshot = load_snapshot(snap_path)
    pipelines = load_pipelines()

    asyncio.run(
        run_eval(
            seed=int(args.seed),
            format_id=args.format,
            start_ps=args.start_ps,
            restart_ps=args.restart_ps,
            timeout_s=float(args.timeout),
            port=int(args.port),
            snapshot=snapshot,
            pipelines=pipelines,
            smoke=bool(args.smoke),
            quiet=not args.verbose,
            progress_every=int(args.progress_every),
        )
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
