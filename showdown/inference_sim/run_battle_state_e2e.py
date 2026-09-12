"""E2E: heuristic vs random + validación Battle IAP vs protocolo PS local.

Módulo aparte de run_heuristic_e2e / runner (soundness SP). Pensado para
detectar desyncs de Battle/Battler/Pokemon al cambiar reglas Champions.

Uso (agente Cursor / terminal — mínimo crédito):
  python -m showdown.inference_sim.run_battle_state_e2e --battles 20

Espera la línea DONE ... y lee logs/battle_state_e2e_report.txt.
No analices turno a turno.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from poke_env.player import RandomPlayer
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from showdown.inference_sim.battle_state_check import StateCheckReport
from showdown.inference_sim.export_metavgc_teams import export_teams
from showdown.inference_sim.paths import BATTLE_FORMAT, REPO_ROOT
from showdown.inference_sim.ps_server import ensure_ps, is_ps_up, ps_start_hint
from showdown.inference_sim.team_factory import (
    TeamMeta,
    active_teams,
    random_battle_pairings,
    to_showdown_export,
)
from showdown.inference_sim.validated_greedy_player import ValidatedGreedyPlayer

REPORT_PATH = REPO_ROOT / 'logs' / 'battle_state_e2e_report.txt'
PROGRESS_PATH = REPO_ROOT / 'logs' / 'battle_state_e2e_progress.json'

for _name in ('poke-env', 'poke_env', 'asyncio'):
    logging.getLogger(_name).setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)


def _ascii(s: str) -> str:
    return s.encode('ascii', 'replace').decode('ascii')


def _write_progress(payload: Dict[str, Any]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _build_report(
    *,
    n_battles: int,
    pool_n: int,
    wins: int,
    losses: int,
    errors: int,
    elapsed_s: float,
    seed: Optional[int],
    state: StateCheckReport,
    details: List[dict],
    include_details: bool,
) -> str:
    decided = wins + losses
    wr = (100.0 * wins / decided) if decided else 0.0
    bps = (wins + losses + errors) / elapsed_s if elapsed_s > 0 else 0.0
    if wins > losses:
        verdict = 'YES - heuristic beat random overall'
    elif wins == losses and decided:
        verdict = 'TIE'
    else:
        verdict = 'NO - random edged out / incomplete'

    lines = [
        '=== Battle State E2E Report (heuristic vs random + Battle sync) ===',
        f'time_utc: {datetime.now(timezone.utc).isoformat()}',
        f'format: {BATTLE_FORMAT}',
        f'ps_up: {is_ps_up()}',
        f'team_pool: {pool_n}',
        f'battles: {n_battles}',
        f'seed: {seed if seed is not None else "-"}',
        f'elapsed_s: {elapsed_s:.1f}',
        f'battles_per_s: {bps:.3f}',
        '',
        f'heuristic_wins: {wins}',
        f'heuristic_losses: {losses}',
        f'errors: {errors}',
        f'win_rate: {wr:.1f}%',
        f'verdict: {verdict}',
        '',
        f'state_mismatch: {state.flag_line()}',
        f'state_checks: {state.checks}',
        f'state_mismatches: {state._total_mismatches}',
        f'feed_errors: {state.feed_errors}',
        f'battle_errors: {state.battle_errors}',
        '',
        'Player A = ValidatedGreedyPlayer | Player B = RandomPlayer',
        'Contrast: IAP Battle/Battler/Pokemon vs poke-env (PS local protocol)',
    ]
    if state.mismatches:
        lines.append('')
        lines.append('--- state mismatch samples (capped) ---')
        for mm in state.mismatches:
            lines.append(
                f'- t{mm.turn} {mm.field}: pe={_ascii(mm.expected)} | '
                f'iap={_ascii(mm.got)}'
                + (f' | {_ascii(mm.detail)}' if mm.detail else '')
            )
    if include_details and details:
        lines.append('')
        lines.append('--- battles ---')
        for i, row in enumerate(details, start=1):
            status = 'ERROR' if row['error'] else ('WIN' if row['heuristic_won'] else 'LOSS')
            sm = 'MISMATCH' if row.get('state_mismatch') else 'ok'
            lines.append(
                f'{i}. [{status}/{sm}] {_ascii(row["pair"])}'
                + (f' | {row["error"]}' if row['error'] else '')
            )
    return '\n'.join(lines) + '\n'


async def _stop_players(*players) -> None:
    for pl in players:
        try:
            await pl.ps_client.stop_listening()
        except Exception:
            pass


async def run_batch(
    n_battles: int,
    *,
    format_id: str,
    start_ps: bool,
    restart_ps: bool,
    seed: Optional[int],
    delay_s: float,
    timeout_s: float,
    progress_every: int,
    include_details: bool,
    export: bool,
    quiet: bool,
) -> str:
    if not ensure_ps(start=start_ps or restart_ps, restart=restart_ps):
        raise RuntimeError(
            'Pokémon Showdown no responde en localhost:8000. '
            f'Arranca: {ps_start_hint()}  o usa --start-ps / --restart-ps'
        )

    if export:
        export_teams()

    pool = active_teams()
    pairs: List[Tuple[TeamMeta, TeamMeta]] = random_battle_pairings(n_battles, seed=seed)
    state = StateCheckReport()

    stamp = int(time.time()) % 100000
    heur_team0, rnd_team0 = pairs[0]
    heur = ValidatedGreedyPlayer(
        user_team=heur_team0,
        opp_team=rnd_team0,
        state_report=state,
        account_configuration=AccountConfiguration(f'HeurState{stamp}', None),
        battle_format=format_id,
    )
    rnd = RandomPlayer(
        account_configuration=AccountConfiguration(f'RndState{stamp}', None),
        battle_format=format_id,
        team=to_showdown_export(rnd_team0),
        server_configuration=LocalhostServerConfiguration,
        accept_open_team_sheet=True,
        max_concurrent_battles=1,
        log_level=30,
    )

    wins = 0
    losses = 0
    errors = 0
    details: List[dict] = []
    t0 = time.perf_counter()
    mismatches_before = 0

    if not quiet:
        print(
            f'BATCH state-e2e battles={n_battles} pool={len(pool)} '
            f'delay_s={delay_s} seed={seed}',
            flush=True,
        )

    try:
        for i, (team_a, team_b) in enumerate(pairs):
            if i % 2 == 0:
                heur_team, rnd_team = team_a, team_b
            else:
                heur_team, rnd_team = team_b, team_a

            pair_label = f'{heur_team.name} vs {rnd_team.name}'
            row = {
                'pair': pair_label,
                'heuristic_won': False,
                'error': '',
                'state_mismatch': False,
            }

            heur.reset_for_next_battle(heur_team, rnd_team)
            rnd.update_team(to_showdown_export(rnd_team))

            wins_before = heur.n_won_battles
            finished_before = heur.n_finished_battles
            mismatches_before = state._total_mismatches
            feed_before = state.feed_errors

            try:
                await asyncio.wait_for(
                    heur.battle_against(rnd, n_battles=1),
                    timeout=timeout_s,
                )
                if heur.n_finished_battles > finished_before:
                    row['heuristic_won'] = heur.n_won_battles > wins_before
                    if row['heuristic_won']:
                        wins += 1
                    else:
                        losses += 1
                else:
                    errors += 1
                    state.battle_errors += 1
                    row['error'] = 'no_finished_battle'
            except Exception as exc:
                errors += 1
                state.battle_errors += 1
                row['error'] = repr(exc)

            row['state_mismatch'] = (
                state._total_mismatches > mismatches_before
                or state.feed_errors > feed_before
            )

            if include_details or row['error'] or row['state_mismatch']:
                details.append(row)

            if (i + 1) % 25 == 0:
                try:
                    heur.reset_battles()
                    rnd.reset_battles()
                except Exception:
                    pass

            done = i + 1
            elapsed = time.perf_counter() - t0
            _write_progress({
                'done': done,
                'total': n_battles,
                'wins': wins,
                'losses': losses,
                'errors': errors,
                'state_mismatch': state.flag_line(),
                'state_mismatches': state._total_mismatches,
                'elapsed_s': round(elapsed, 2),
                'last_pair': _ascii(pair_label),
            })

            if not quiet and (done % max(1, progress_every) == 0 or done == n_battles):
                print(
                    f'  [{done}/{n_battles}] W={wins} L={losses} E={errors} '
                    f'state_mismatch={state.flag_line()} '
                    f'mm={state._total_mismatches} t={elapsed:.0f}s',
                    flush=True,
                )

            if delay_s > 0 and i + 1 < n_battles:
                await asyncio.sleep(delay_s)
    finally:
        await _stop_players(heur, rnd)

    elapsed = time.perf_counter() - t0
    report = _build_report(
        n_battles=n_battles,
        pool_n=len(pool),
        wins=wins,
        losses=losses,
        errors=errors,
        elapsed_s=elapsed,
        seed=seed,
        state=state,
        details=details,
        include_details=include_details,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='utf-8')

    decided = wins + losses
    wr = (100.0 * wins / decided) if decided else 0.0
    print(
        f'DONE battles={n_battles} W={wins} L={losses} E={errors} '
        f'WR={wr:.1f}% state_mismatch={state.flag_line()} '
        f'mm={state._total_mismatches} elapsed_s={elapsed:.1f} report={REPORT_PATH}',
        flush=True,
    )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Batch E2E: heuristic vs random + Battle state sync vs PS local'
    )
    parser.add_argument('--battles', type=int, default=20)
    parser.add_argument('--format', default=BATTLE_FORMAT)
    parser.add_argument('--start-ps', action='store_true')
    parser.add_argument(
        '--restart-ps',
        action='store_true',
        help='Reinicia PS local con --no-security',
    )
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--delay', type=float, default=0.0)
    parser.add_argument('--timeout', type=float, default=180.0)
    parser.add_argument('--progress-every', type=int, default=5)
    parser.add_argument('--details', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)

    if args.export_only:
        out = export_teams()
        print(f'Exported teams -> {out}')
        return 0

    quiet = not args.verbose
    if not quiet:
        print(f'PS up: {is_ps_up()} | format={args.format} | battles={args.battles}')

    asyncio.run(
        run_batch(
            args.battles,
            format_id=args.format,
            start_ps=args.start_ps,
            restart_ps=args.restart_ps,
            seed=args.seed,
            delay_s=args.delay,
            timeout_s=args.timeout,
            progress_every=args.progress_every,
            include_details=args.details,
            export=args.export,
            quiet=quiet,
        )
    )
    # Exit 1 solo si hubo mismatch de estado (útil para CI); batallas perdidas no fallan
    text = REPORT_PATH.read_text(encoding='utf-8') if REPORT_PATH.exists() else ''
    if 'state_mismatch: YES' in text:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
