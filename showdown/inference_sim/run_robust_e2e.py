"""E2E robusto: pools A/B disjuntos + BO3 + sync Battle + soundness OTS.

Entrada: N equipos (par). Ej. 100 → pool A (50) heurístico, pool B (50) random,
sin equipos en común → 50 series BO3.

Por turno: contraste IAP Battle vs poke-env (PS local) + validación de inferencia
SP contra OTS ground-truth.

Uso (mínimo crédito Cursor):
  python -m showdown.inference_sim.run_robust_e2e --teams 100

Espera DONE ... y lee logs/robust_e2e_report.txt. No analices turnos.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from poke_env.player import RandomPlayer
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from showdown.inference_sim.battle_state_check import StateCheckReport
from showdown.inference_sim.export_metavgc_teams import export_teams
from showdown.inference_sim.inference_flag_audit import (
    FlagAuditResult,
    audit_inference_flags,
    collect_pool_abilities,
    format_audit_section,
)
from showdown.inference_sim.paths import REPO_ROOT
from showdown.inference_sim.ps_server import ensure_ps, is_ps_up, ps_start_hint
from showdown.inference_sim.soundness import SoundnessReport, format_timeline_example
from showdown.inference_sim.team_factory import (
    TeamMeta,
    active_teams,
    disjoint_ab_pairings,
    to_showdown_export,
)
from showdown.inference_sim.validated_greedy_player import ValidatedGreedyPlayer
from showdown.inference_sim.robust_fixplan import FIXPLAN_PATH, write_fixplan

REPORT_PATH = REPO_ROOT / 'logs' / 'robust_e2e_report.txt'
PROGRESS_PATH = REPO_ROOT / 'logs' / 'robust_e2e_progress.json'

DEFAULT_FORMAT = os.environ.get(
    'ROBUST_E2E_FORMAT',
    os.environ.get('POKEMON_MODE', 'gen9championsvgc2026regmcbo3'),
)

for _name in ('poke-env', 'poke_env', 'asyncio'):
    logging.getLogger(_name).setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)


def _ascii(s: str) -> str:
    return s.encode('ascii', 'replace').decode('ascii')


def _write_progress(payload: Dict[str, Any]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _improvements(
    state: StateCheckReport,
    sound: SoundnessReport,
    flags: Optional[FlagAuditResult] = None,
) -> List[str]:
    tips: List[str] = []
    if flags is not None and flags.need_flags_add:
        highs = [g.flag for g in flags.gaps if g.priority == 'high']
        tip = 'Flags: NEED_FLAGS_ADD=YES (no abre agente solo; pedir "agrega flags de inferencia").'
        if highs:
            tip += f' Prioridad: {", ".join(highs[:4])}.'
        tips.append(tip)
    if state.state_mismatch:
        fields = Counter(m.field.split('.')[0] for m in state.mismatches)
        top = ', '.join(f'{k}×{v}' for k, v in fields.most_common(5)) or 'varios'
        tips.append(
            f'Sync: revisar battle_modifier para campos frecuentes ({top}); '
            'priorizar switch/weather/fields/side conditions.'
        )
    if state.feed_errors:
        tips.append(
            f'Sync feed: {state.feed_errors} excepciones al aplicar protocolo → '
            'blindar handlers en BATTLE_MODIFIERS_LOOKUP.'
        )
    if not sound.ok:
        kinds = Counter()
        for f in sound.failures:
            if 'excluido' in (f.detail or ''):
                kinds['sp_excluido'] += 1
            elif 'fuera de rango' in (f.detail or ''):
                kinds['final_fuera'] += 1
            else:
                kinds['otro'] += 1
        tips.append(
            f'Inferencia FAIL ({sound.failure_count}): {dict(kinds)}. '
            'Revisar mega/forme lookup OTS, sash/crit skip y bases post-mega.'
        )
    if sound.skipped and sound.observations and sound.skipped > sound.observations:
        tips.append(
            'Inferencia: muchos skips vs observations → ampliar señales '
            '(-damage / speed ties) o relajar filtros sash/crit.'
        )
    if sound.ok and not state.state_mismatch:
        tips.append(
            'OK: sin desync ni exclusión de SP verdadero. '
            'Próxima corrida: subir --teams o cambiar seed tras reglas nuevas.'
        )
    elif sound.ok and sound.narrowing_events == 0:
        tips.append(
            'Inferencia PASS pero sin estrechamiento observado → '
            'verificar que el feed llega a -damage/turn en BO3 game 2+.'
        )
    if not tips:
        tips.append('Sin observaciones accionables en esta corrida.')
    return tips


def _build_report(
    *,
    n_teams: int,
    n_bo3: int,
    format_id: str,
    pool_n: int,
    wins: int,
    losses: int,
    errors: int,
    elapsed_s: float,
    seed: Optional[int],
    state: StateCheckReport,
    sound: SoundnessReport,
    details: List[dict],
    include_details: bool,
    flags: Optional[FlagAuditResult] = None,
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
        '=== Robust E2E Report (A/B pools + BO3 + sync + OTS inference) ===',
        f'time_utc: {datetime.now(timezone.utc).isoformat()}',
        f'format: {format_id}',
        f'ps_up: {is_ps_up()}',
        f'team_pool_available: {pool_n}',
        f'teams_used: {n_teams} (A={n_bo3} heuristic, B={n_bo3} random, disjoint)',
        f'bo3_series: {n_bo3}',
        f'seed: {seed if seed is not None else "-"}',
        f'elapsed_s: {elapsed_s:.1f}',
        f'bo3_per_s: {bps:.3f}',
        '',
        '--- results ---',
        f'heuristic_wins: {wins}',
        f'random_wins: {losses}',
        f'errors: {errors}',
        f'win_rate_heuristic: {wr:.1f}%',
        f'verdict: {verdict}',
        '',
        '--- sync (Battle vs PS local) ---',
        f'state_mismatch: {state.flag_line()}',
        f'state_checks: {state.checks}',
        f'state_mismatches: {state._total_mismatches}',
        f'feed_errors: {state.feed_errors}',
        '',
        '--- inference (OTS ground truth) ---',
        f'inference_ok: {"YES" if sound.ok else "NO"}',
        f'soundness_checks: {sound.checks}',
        f'soundness_failures: {sound.failure_count}',
        f'belief_observations: {sound.observations}',
        f'belief_skips: {sound.skipped}',
        f'narrowing_events: {sound.narrowing_events}',
        f'battle_errors: {sound.battle_errors}',
        '',
    ]
    if flags is not None:
        lines.extend(format_audit_section(flags))
        lines.append('')
    lines.append('--- improvements / notes for next runs ---')
    for tip in _improvements(state, sound, flags):
        lines.append(f'- {tip}')

    if state.mismatches:
        lines.append('')
        lines.append('--- sync mismatch samples (capped) ---')
        for mm in state.mismatches[:20]:
            lines.append(
                f'- t{mm.turn} {mm.field}: pe={_ascii(mm.expected)} | '
                f'iap={_ascii(mm.got)}'
            )

    if sound.failures:
        lines.append('')
        lines.append('--- inference failure samples (capped) ---')
        for fail in sound.failures[:15]:
            lines.append(
                f'- {fail.species}.{fail.stat}: {_ascii(fail.detail)} '
                f'(true_sp={fail.true_sp}, range={fail.stat_range}, '
                f'n_cand={len(fail.candidates)})'
            )

    # Timeline breve (1 ejemplo) — útil sin volcar todo
    try:
        tl = format_timeline_example(sound)
        if tl and not (len(tl) == 1 and 'Sin timeline' in tl[0]):
            lines.append('')
            lines.extend(_ascii(x) for x in tl[:40])
    except Exception:
        pass

    if include_details and details:
        lines.append('')
        lines.append('--- bo3 series ---')
        for i, row in enumerate(details, start=1):
            status = 'ERROR' if row['error'] else ('WIN' if row['heuristic_won'] else 'LOSS')
            tags = []
            if row.get('state_mismatch'):
                tags.append('SYNC')
            if row.get('inference_fail'):
                tags.append('INF')
            tag_s = '/'.join(tags) if tags else 'ok'
            lines.append(
                f'{i}. [{status}|{tag_s}] {_ascii(row["pair"])}'
                + (f' | {row["error"]}' if row['error'] else '')
            )

    lines.append('')
    lines.append('Player A = ValidatedGreedyPlayer | Player B = RandomPlayer')
    return '\n'.join(lines) + '\n'


async def _stop_players(*players) -> None:
    for pl in players:
        try:
            await pl.ps_client.stop_listening()
        except Exception:
            pass


async def run_batch(
    n_teams: int,
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
    if n_teams % 2 != 0 or n_teams < 2:
        raise ValueError('--teams debe ser un entero par >= 2')

    if not ensure_ps(start=start_ps or restart_ps, restart=restart_ps):
        raise RuntimeError(
            'Pokémon Showdown no responde en localhost:8000. '
            f'Arranca: {ps_start_hint()}  o usa --start-ps / --restart-ps'
        )

    if export:
        export_teams()

    pool = active_teams()
    pairs: List[Tuple[TeamMeta, TeamMeta]] = disjoint_ab_pairings(n_teams, seed=seed)
    n_bo3 = len(pairs)
    state = StateCheckReport()
    sound = SoundnessReport()
    # Auditoría estática una vez (pool abilities = señal barata de gaps relevantes)
    used_metas: List[TeamMeta] = []
    for a, b in pairs:
        used_metas.extend([a, b])
    flag_audit = audit_inference_flags(
        pool_abilities=collect_pool_abilities(used_metas),
    )

    stamp = int(time.time()) % 100000
    heur_team0, rnd_team0 = pairs[0]
    heur = ValidatedGreedyPlayer(
        user_team=heur_team0,
        opp_team=rnd_team0,
        state_report=state,
        soundness_report=sound,
        account_configuration=AccountConfiguration(f'HeurRobust{stamp}', None),
        battle_format=format_id,
    )
    rnd = RandomPlayer(
        account_configuration=AccountConfiguration(f'RndRobust{stamp}', None),
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

    if not quiet:
        print(
            f'BATCH robust teams={n_teams} bo3={n_bo3} format={format_id} seed={seed}',
            flush=True,
        )

    try:
        for i, (heur_team, rnd_team) in enumerate(pairs):
            pair_label = f'{heur_team.name} vs {rnd_team.name}'
            row = {
                'pair': pair_label,
                'heuristic_won': False,
                'error': '',
                'state_mismatch': False,
                'inference_fail': False,
            }

            heur.reset_for_next_battle(heur_team, rnd_team)
            rnd.update_team(to_showdown_export(rnd_team))

            wins_before = heur.n_won_battles
            finished_before = heur.n_finished_battles
            mm_before = state._total_mismatches
            feed_before = state.feed_errors
            fail_before = sound.failure_count

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
                    sound.battle_errors += 1
                    row['error'] = 'no_finished_bo3'
            except Exception as exc:
                errors += 1
                sound.battle_errors += 1
                row['error'] = repr(exc)

            row['state_mismatch'] = (
                state._total_mismatches > mm_before or state.feed_errors > feed_before
            )
            row['inference_fail'] = sound.failure_count > fail_before

            if include_details or row['error'] or row['state_mismatch'] or row['inference_fail']:
                details.append(row)

            if (i + 1) % 10 == 0:
                try:
                    heur.reset_battles()
                    rnd.reset_battles()
                except Exception:
                    pass

            done = i + 1
            elapsed = time.perf_counter() - t0
            _write_progress({
                'done': done,
                'total': n_bo3,
                'wins': wins,
                'losses': losses,
                'errors': errors,
                'state_mismatch': state.flag_line(),
                'inference_ok': 'YES' if sound.ok else 'NO',
                'soundness_failures': sound.failure_count,
                'elapsed_s': round(elapsed, 2),
                'last_pair': _ascii(pair_label),
            })

            if not quiet and (done % max(1, progress_every) == 0 or done == n_bo3):
                print(
                    f'  [{done}/{n_bo3}] W={wins} L={losses} E={errors} '
                    f'sync={state.flag_line()} inf={"OK" if sound.ok else "FAIL"} '
                    f't={elapsed:.0f}s',
                    flush=True,
                )

            if delay_s > 0 and i + 1 < n_bo3:
                await asyncio.sleep(delay_s)
    finally:
        await _stop_players(heur, rnd)

    elapsed = time.perf_counter() - t0
    report = _build_report(
        n_teams=n_teams,
        n_bo3=n_bo3,
        format_id=format_id,
        pool_n=len(pool),
        wins=wins,
        losses=losses,
        errors=errors,
        elapsed_s=elapsed,
        seed=seed,
        state=state,
        sound=sound,
        details=details,
        include_details=include_details,
        flags=flag_audit,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='utf-8')
    fixplan = write_fixplan(report)

    decided = wins + losses
    wr = (100.0 * wins / decided) if decided else 0.0
    need = 'YES' if 'NEED_FIX=YES' in fixplan.read_text(encoding='utf-8') else 'NO'
    print(
        f'DONE teams={n_teams} bo3={n_bo3} W={wins} L={losses} E={errors} '
        f'WR={wr:.1f}% state_mismatch={state.flag_line()} '
        f'inference_ok={"YES" if sound.ok else "NO"} '
        f'inf_fail={sound.failure_count} NEED_FIX={need} '
        f'NEED_FLAGS_ADD={flag_audit.flag_line()} '
        f'elapsed_s={elapsed:.1f} report={REPORT_PATH} fixplan={FIXPLAN_PATH}',
        flush=True,
    )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Robust E2E: disjoint A/B pools, BO3, Battle sync + OTS inference'
    )
    parser.add_argument(
        '--teams',
        type=int,
        default=100,
        help='N equipos (par). Genera N/2 series BO3 (A=heuristic, B=random)',
    )
    parser.add_argument('--format', default=DEFAULT_FORMAT)
    parser.add_argument('--start-ps', action='store_true')
    parser.add_argument('--restart-ps', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--delay', type=float, default=0.0)
    parser.add_argument(
        '--timeout',
        type=float,
        default=600.0,
        help='Timeout por serie BO3 (s)',
    )
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
        print(
            f'PS up: {is_ps_up()} | format={args.format} | '
            f'teams={args.teams} bo3={args.teams // 2}'
        )

    asyncio.run(
        run_batch(
            args.teams,
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

    text = REPORT_PATH.read_text(encoding='utf-8') if REPORT_PATH.exists() else ''
    if 'state_mismatch: YES' in text or 'inference_ok: NO' in text:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
