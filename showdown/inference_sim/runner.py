"""CLI: batallas Champions locales con poke-env + validación soundness."""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import List

from poke_env.player import RandomPlayer
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from showdown.inference_sim.mirror_player import InferenceMirrorPlayer
from showdown.inference_sim.paths import BATTLE_FORMAT, REPORT_PATH
from showdown.inference_sim.ps_server import ensure_ps, is_ps_up, ps_start_hint
from showdown.inference_sim.soundness import SoundnessFailure, SoundnessReport, format_report
from showdown.inference_sim.team_factory import (
    METAVGC_TEAMS,
    active_teams,
    battle_pairings,
    to_showdown_export,
)


async def run_battles(
    n_battles: int,
    *,
    format_id: str = BATTLE_FORMAT,
    start_ps: bool = False,
) -> SoundnessReport:
    if not ensure_ps(start=start_ps):
        raise RuntimeError(
            'Pokémon Showdown no responde en localhost:8000. '
            f'Arranca: {ps_start_hint()}  (repo Champions) o --start-ps'
        )

    report = SoundnessReport()
    teams_used: List[str] = []
    pairs = battle_pairings(n_battles)
    pool = active_teams()
    print(f'Team pool: {len(pool)} MetaVGC sets')

    for i, (user_meta, opp_meta) in enumerate(pairs):
        teams_used.append(f'{user_meta.name} vs {opp_meta.name}')
        print(f'\n=== Battle {i + 1}/{n_battles}: {teams_used[-1]} ===')

        mirror = InferenceMirrorPlayer(
            user_team=user_meta,
            opp_team_ref=opp_meta,
            report=report,
            account_configuration=AccountConfiguration(f'InfBot{i}', None),
            battle_format=format_id,
        )
        foe = RandomPlayer(
            account_configuration=AccountConfiguration(f'RndBot{i}', None),
            battle_format=format_id,
            team=to_showdown_export(opp_meta),
            server_configuration=LocalhostServerConfiguration,
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
        )

        try:
            await asyncio.wait_for(mirror.battle_against(foe, n_battles=1), timeout=300)
            print(
                f'  finished wins={mirror.n_won_battles} '
                f'checks_so_far={report.checks} fails={len(report.failures)} '
                f'protocol_lines={len(mirror.protocol_log)}'
            )
        except Exception as exc:
            print(f'  ERROR: {exc!r}')
            # Timeouts / team reject no cuentan como fallo de soundness SP
            if 'Timeout' in type(exc).__name__ or 'timeout' in str(exc).lower():
                report.battle_errors += 1
            else:
                report.add_failure(SoundnessFailure(
                    battle_tag=f'battle-{i}',
                    species='?',
                    stat='?',
                    true_sp=-1,
                    true_final=-1,
                    candidates=[],
                    stat_range=None,
                    detail=f'battle exception: {exc!r}',
                    recent_lines=list(mirror.protocol_log[-40:]),
                ))
                report.battle_errors += 1
        finally:
            from showdown.inference_sim.ps_client_shutdown import stop_players

            await stop_players(mirror, foe)

    report._teams_used = teams_used  # type: ignore[attr-defined]
    return report


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='E2E inference soundness via poke-env')
    parser.add_argument('--battles', type=int, default=3)
    parser.add_argument('--format', default=BATTLE_FORMAT)
    parser.add_argument('--start-ps', action='store_true')
    parser.add_argument('--list-teams', action='store_true')
    args = parser.parse_args(argv)

    if args.list_teams:
        for t in active_teams():
            print(f'- {t.name}\n  {t.source_url}')
        return 0

    print(f'PS up: {is_ps_up()} | format={args.format} | battles={args.battles}')
    report = asyncio.run(
        run_battles(args.battles, format_id=args.format, start_ps=args.start_ps)
    )
    teams = getattr(report, '_teams_used', [t.name for t in active_teams()])
    text = format_report(report, battles=args.battles, teams=teams)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding='utf-8')
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))
    print(f'\nWrote {REPORT_PATH}')
    return 0 if report.ok else 1


if __name__ == '__main__':
    sys.exit(main())
