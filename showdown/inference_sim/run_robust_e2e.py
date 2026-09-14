"""E2E robusto: pools A/B disjuntos + BO3 cruzado + sync Battle + soundness OTS.

Entrada: N equipos (par). Ej. 100 → pool A (50) / pool B (50) disjuntos → 50
matchups. Cada matchup juega dos series BO3 (equipos cruzados A↔B):
  - A gana ambas → +1 pair_a
  - B gana ambas → +1 pair_b
  - 1-1 → +1 pair_draw
  - error/sync/inf en cualquiera → matchup invalidado (fallos separados)

Reporta WR batalla crudo (info), WR de pares (el que importa) e IC bootstrap 95%.

Uso:
  python -m showdown.inference_sim.run_robust_e2e --teams 100
  python -m showdown.inference_sim.run_robust_e2e --teams 100 \\
      --player-a expectimax_bayesian --player-b greedy_joint
  python -m showdown.inference_sim.run_robust_e2e --teams 100 --workers 2 --start-ps

Workers (default 2): un proceso PS por puerto (8000,8001 o PS_PORTS=...).
El servidor NO se reinicia por cada BO3; solo con --start-ps / --restart-ps.

Formato DONE (multilínea)::

  DONE teams=N A=... B=...
  bo3=N: W=... L=... E=... WR_raw=...%
  matchups=N: pair_A=... pair_B=... draw=... invalid=...
  sync_ok=YES|NO fail_sync=...
  inf_ok=YES|NO fail_matchup_inf=... fail_bo3_inf=...
  NEED_FIX=YES|NO NEED_FLAGS_ADD=YES|NO elapsed_s=...
"""
from __future__ import annotations

# Red de seguridad OTS/BO3 (IndexError en |showteam| → no congelar el batch).
import showdown.inference_sim.pokeenv_showteam_patch  # noqa: F401

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from poke_env.ps_client import AccountConfiguration

from showdown.inference_sim.battle_state_check import StateCheckReport
from showdown.inference_sim.export_metavgc_teams import export_teams
from showdown.inference_sim.inference_flag_audit import (
    FlagAuditResult,
    audit_inference_flags,
    collect_pool_abilities,
    format_audit_section,
)
from showdown.inference_sim.chat_quiet import set_chat_quiet
from showdown.inference_sim.paths import REPO_ROOT, server_configuration_for_port
from showdown.inference_sim.ps_server import (
    ensure_ps_ports,
    is_ps_up,
    parse_ps_ports,
    ps_start_hint,
)
from showdown.inference_sim.soundness import SoundnessReport, format_timeline_example
from showdown.inference_sim.team_factory import (
    TeamMeta,
    active_teams,
    disjoint_ab_pairings,
    to_showdown_export,
)
from showdown.inference_sim.validated_greedy_player import (
    PLAYER_A_CHOICES,
    PLAYER_B_CHOICES,
    make_opponent_player,
    validated_player_class,
)
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


def _series_won(*, game_wins: int, games_done: int) -> bool:
    """Victoria de la serie BO3 = mayoría de partidas (no bastar ganar una)."""
    if games_done <= 0:
        return False
    return game_wins > (games_done - game_wins)


def _bootstrap_ci(
    outcomes: Sequence[int],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """outcomes +1 (A barre), -1 (B barre), 0 (split) → mean y IC 95%."""
    vals = [int(x) for x in outcomes]
    if not vals:
        return 0.0, 0.0, 0.0
    mean = sum(vals) / float(len(vals))
    rng = random.Random(seed)
    n = len(vals)
    means: List[float] = []
    for _ in range(max(100, int(n_boot))):
        s = 0.0
        for _i in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / float(n))
    means.sort()
    lo_i = int(len(means) * (alpha / 2.0))
    hi_i = int(len(means) * (1.0 - alpha / 2.0))
    hi_i = min(len(means) - 1, max(lo_i, hi_i))
    return mean, means[lo_i], means[hi_i]


def _pair_wr_ci(
    pair_a: int,
    pair_b: int,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 1,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """WR de A en pares decisivos + IC bootstrap 95%."""
    decisive = int(pair_a) + int(pair_b)
    if decisive <= 0:
        return None, None, None
    trials = [1] * int(pair_a) + [0] * int(pair_b)
    rng = random.Random(seed)
    n = len(trials)
    wr = sum(trials) / float(n)
    boots: List[float] = []
    for _ in range(max(100, int(n_boot))):
        s = 0
        for _i in range(n):
            s += trials[rng.randrange(n)]
        boots.append(s / float(n))
    boots.sort()
    lo_i = int(len(boots) * (alpha / 2.0))
    hi_i = int(len(boots) * (1.0 - alpha / 2.0))
    hi_i = min(len(boots) - 1, max(lo_i, hi_i))
    return wr, boots[lo_i], boots[hi_i]


def _classify_series_fail(
    *,
    outcome: str,
    err_detail: str,
    state_mismatch: bool,
    inference_fail: bool,
) -> List[str]:
    """Tipos de fallo que invalidan un matchup cruzado (pueden ser varios)."""
    kinds: List[str] = []
    if outcome == 'timeout' or 'TimeoutError' in (err_detail or ''):
        kinds.append('timeout')
    elif outcome == 'reject':
        kinds.append('reject')
    elif outcome != 'ok':
        kinds.append('runtime')
    elif err_detail:
        kinds.append('runtime')
    if state_mismatch:
        kinds.append('sync')
    if inference_fail:
        kinds.append('inference')
    return kinds


def _compact_series_line(
    *,
    done: int,
    total: int,
    status: str,
    games: str,
    wins: int,
    losses: int,
    errors: int,
    sync: str,
    inf: str,
) -> str:
    return (
        f'[{done}/{total}] {status} {games} | '
        f'W={wins} L={losses} E={errors} sync={sync} inf={inf}'
    )


class _TeamRejectWatch(logging.Handler):
    """Detecta |popup| de team reject / move inválido para fallar rápido."""

    def __init__(self, event: asyncio.Event, bucket: Dict[str, str]):
        super().__init__(level=logging.WARNING)
        self._event = event
        self._bucket = bucket

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        low = msg.lower()
        if (
            'team was rejected' in low
            or 'is an invalid move' in low
            or 'is an illegal move' in low
            or "can't learn" in low
        ):
            self._bucket['msg'] = msg.replace('\n', ' ')[:240]
            self._event.set()


async def _cancel_challenges(*players) -> None:
    from showdown.inference_sim.bo3_runtime import reset_bo3_runtime_state_many

    for pl in players:
        try:
            # PS moderno: cancelar challenge saliendo / sin /cancel legacy.
            await pl.ps_client.send_message('/reject all')
        except Exception:
            pass
        try:
            await pl.ps_client.send_message('/leave')
        except Exception:
            pass
    # Siempre drenar cola BO3 tras cancel/timeout (evita cascada 600s).
    reset_bo3_runtime_state_many(list(players))


async def _run_one_series(
    player_a_pl,
    player_b_pl,
    *,
    timeout_s: float,
) -> Tuple[str, str]:
    """Ejecuta 1 challenge BO3. Devuelve (outcome, error_detail).

    outcome: 'ok' | 'reject' | 'timeout' | 'error'

    Si el BO3 ya está decidido en series_games pero battle_against no retorna
    (cola bestof contaminada / |win| padre perdido), fuerza 'ok' tras un grace
    corto en lugar de esperar todo --timeout.
    """
    from showdown.inference_sim.bo3_runtime import bo3_series_decided

    reject_event = asyncio.Event()
    reject_bucket: Dict[str, str] = {}
    watch = _TeamRejectWatch(reject_event, reject_bucket)
    loggers = [
        logging.getLogger(getattr(player_a_pl, 'username', '') or ''),
        logging.getLogger(getattr(player_b_pl, 'username', '') or ''),
        logging.getLogger('poke-env'),
        logging.getLogger('poke_env'),
    ]
    for lg in loggers:
        if lg.name:
            lg.addHandler(watch)
    battle_task = asyncio.create_task(
        player_a_pl.battle_against(player_b_pl, n_battles=1)
    )
    reject_task = asyncio.create_task(reject_event.wait())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + float(timeout_s)
    last_games = 0
    last_progress = loop.time()
    decided_since: Optional[float] = None
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            slice_s = min(15.0, remaining)
            done_set, _pending = await asyncio.wait(
                {battle_task, reject_task},
                timeout=slice_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if reject_task in done_set and reject_event.is_set():
                battle_task.cancel()
                try:
                    await battle_task
                except (asyncio.CancelledError, Exception):
                    pass
                await _cancel_challenges(player_a_pl, player_b_pl)
                return 'reject', reject_bucket.get('msg') or 'team_rejected'
            if battle_task in done_set:
                exc = battle_task.exception()
                if exc is not None:
                    await _cancel_challenges(player_a_pl, player_b_pl)
                    return 'error', repr(exc)
                return 'ok', ''

            games = list(getattr(player_a_pl, 'series_games', None) or [])
            n_games = len(games)
            if n_games > last_games:
                last_games = n_games
                last_progress = loop.time()
            if bo3_series_decided(games):
                if decided_since is None:
                    decided_since = loop.time()
                # BO3 cerrado en hijas pero join() del padre colgado.
                if (loop.time() - decided_since) >= 25.0:
                    battle_task.cancel()
                    try:
                        await battle_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    await _cancel_challenges(player_a_pl, player_b_pl)
                    return 'ok', 'forced_ok_after_bo3_hang'
            else:
                decided_since = None
                # Mid-series stall (p.ej. confirmready / handler muerto).
                if n_games >= 1 and (loop.time() - last_progress) >= 180.0:
                    break

        # timeout
        battle_task.cancel()
        reject_task.cancel()
        try:
            await battle_task
        except (asyncio.CancelledError, Exception):
            pass
        await _cancel_challenges(player_a_pl, player_b_pl)
        return 'timeout', f'TimeoutError({timeout_s}s)'
    finally:
        if not reject_task.done():
            reject_task.cancel()
            try:
                await reject_task
            except (asyncio.CancelledError, Exception):
                pass
        for lg in loggers:
            if lg.name:
                try:
                    lg.removeHandler(watch)
                except Exception:
                    pass


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
    n_matchups: int,
    format_id: str,
    pool_n: int,
    wins: int,
    losses: int,
    errors: int,
    pair_a: int,
    pair_b: int,
    pair_draw: int,
    pair_invalid: int,
    fail_kinds: Counter,
    pair_outcomes: List[int],
    elapsed_s: float,
    seed: Optional[int],
    state: StateCheckReport,
    sound: SoundnessReport,
    details: List[dict],
    include_details: bool,
    flags: Optional[FlagAuditResult] = None,
    player_a: str = 'greedy_solo',
    player_b: str = 'random',
) -> str:
    decided = wins + losses
    wr_raw = (100.0 * wins / decided) if decided else 0.0
    pair_dec = pair_a + pair_b
    wr_pair = (100.0 * pair_a / pair_dec) if pair_dec else 0.0
    wr_ci, wr_lo, wr_hi = _pair_wr_ci(pair_a, pair_b)
    score_mean, score_lo, score_hi = _bootstrap_ci(pair_outcomes)
    series_total = wins + losses + errors
    bps = series_total / elapsed_s if elapsed_s > 0 else 0.0
    if pair_dec and wr_ci is not None and wr_lo is not None and wr_hi is not None:
        if wr_lo > 0.5:
            verdict = f'YES - {player_a} beats {player_b} on paired WR (IC>50%)'
        elif wr_hi < 0.5:
            verdict = f'NO - {player_b} beats {player_a} on paired WR (IC<50%)'
        else:
            verdict = 'UNCLEAR - paired WR IC includes 50%'
    elif pair_a > pair_b:
        verdict = f'LEAN {player_a} (few decisive pairs)'
    elif pair_b > pair_a:
        verdict = f'LEAN {player_b} (few decisive pairs)'
    else:
        verdict = 'TIE / incomplete'

    def _pct(x: Optional[float]) -> str:
        return f'{100.0 * x:.1f}%' if x is not None else '-'

    lines = [
        '=== Robust E2E Report (A/B pools + crossed BO3 + sync + OTS) ===',
        f'time_utc: {datetime.now(timezone.utc).isoformat()}',
        f'format: {format_id}',
        f'ps_up: {is_ps_up()}',
        f'team_pool_available: {pool_n}',
        f'teams_used: {n_teams} (A={n_matchups} {player_a}, B={n_matchups} {player_b}, disjoint)',
        f'player_a: {player_a}',
        f'player_b: {player_b}',
        f'matchups: {n_matchups} (each = 2 crossed BO3)',
        f'bo3_series: {n_bo3}',
        f'seed: {seed if seed is not None else "-"}',
        f'elapsed_s: {elapsed_s:.1f}',
        f'bo3_per_s: {bps:.3f}',
        '',
    ]
    try:
        from showdown.battle_bots.expectimax_bayesian.config_history import (
            format_config_block,
            uses_expectimax,
        )
        if uses_expectimax(player_a, player_b):
            lines.append(format_config_block())
            lines.append('')
    except Exception:
        pass
    lines += [
        '--- results (battle/series raw, informative) ---',
        f'player_a_series_wins: {wins}',
        f'player_b_series_wins: {losses}',
        f'player_a_wins: {wins}',
        f'player_b_wins: {losses}',
        f'heuristic_wins: {wins}',
        f'random_wins: {losses}',
        f'series_errors: {errors}',
        f'errors: {errors}',
        f'win_rate_battle_raw: {wr_raw:.1f}%',
        f'win_rate_player_a: {wr_raw:.1f}%',
        f'win_rate_heuristic: {wr_raw:.1f}%',
        '',
        '--- results (paired crossed matchups — primary) ---',
        f'pair_a_sweeps: {pair_a}',
        f'pair_b_sweeps: {pair_b}',
        f'pair_draws: {pair_draw}',
        f'pair_invalid: {pair_invalid}',
        f'win_rate_pairs: {wr_pair:.1f}%',
        f'pair_wr_ci95: [{_pct(wr_lo)}, {_pct(wr_hi)}]',
        f'pair_score_mean: {score_mean:+.3f}',
        f'pair_score_ci95: [{score_lo:+.3f}, {score_hi:+.3f}]',
        f'verdict: {verdict}',
        '',
        '--- failure kinds (invalidating matchups; series may count multiple) ---',
        f'fail_timeout: {int(fail_kinds.get("timeout", 0))}',
        f'fail_runtime: {int(fail_kinds.get("runtime", 0))}',
        f'fail_reject: {int(fail_kinds.get("reject", 0))}',
        f'fail_sync: {int(fail_kinds.get("sync", 0))}',
        f'fail_matchup_inf: {int(fail_kinds.get("inference", 0))}',
        f'fail_bo3_inf: {int(sound.failure_count)}',
        # aliases (compat parsers / fixplan antiguo)
        f'fail_inference: {int(fail_kinds.get("inference", 0))}',
        '',
        '--- sync (Battle vs PS local) ---',
        f'sync_ok: {"YES" if state.ok else "NO"}',
        f'state_mismatch: {state.flag_line()}',
        f'state_checks: {state.checks}',
        f'state_mismatches: {state._total_mismatches}',
        f'feed_errors: {state.feed_errors}',
        '',
        '--- inference (OTS ground truth) ---',
        f'inf_ok: {"YES" if sound.ok else "NO"}',
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
            status = 'ERROR' if row['error'] else ('WIN' if row['player_a_won'] else 'LOSS')
            tags = []
            if row.get('state_mismatch'):
                tags.append('SYNC')
            if row.get('inference_fail'):
                tags.append('INF')
            tag_s = '/'.join(tags) if tags else 'ok'
            games = row.get('games') or ''
            lines.append(
                f'{i}. [{status}|{tag_s}] {games} {_ascii(row["pair"])}'
                + (f' | {row["error"]}' if row['error'] else '')
            )

    lines.append('')
    lines.append(f'Player A = validated({player_a}) | Player B = {player_b}')
    return '\n'.join(lines) + '\n'


async def _stop_players(*players) -> None:
    for pl in players:
        try:
            await pl.ps_client.stop_listening()
        except Exception:
            pass


def _merge_state_reports(dst: StateCheckReport, src: StateCheckReport) -> None:
    dst.checks += src.checks
    dst.feed_errors += src.feed_errors
    dst.battle_errors += src.battle_errors
    dst._total_mismatches += src._total_mismatches
    room = dst._cap - len(dst.mismatches)
    if room > 0:
        dst.mismatches.extend(src.mismatches[:room])


def _merge_sound_reports(dst: SoundnessReport, src: SoundnessReport) -> None:
    dst.checks += src.checks
    dst.observations += src.observations
    dst.skipped += src.skipped
    dst.narrowing_events += src.narrowing_events
    dst.battle_errors += src.battle_errors
    dst._total_failures += src._total_failures
    room = dst._failure_cap - len(dst.failures)
    if room > 0:
        dst.failures.extend(src.failures[:room])
    for sid, hist in (src.timelines or {}).items():
        dst.timelines.setdefault(sid, []).extend(hist)


def _split_round_robin(items: List[Any], n: int) -> List[List[Any]]:
    """Reparte ítems en n buckets intercalados (carga similar)."""
    n = max(1, int(n))
    buckets: List[List[Any]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return buckets


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
    player_a: str = 'greedy_solo',
    player_b: str = 'random',
    workers: int = 2,
) -> str:
    if n_teams % 2 != 0 or n_teams < 2:
        raise ValueError('--teams debe ser un entero par >= 2')

    set_chat_quiet(quiet)
    workers = max(1, int(workers))
    ports = parse_ps_ports(workers=workers)
    if len(ports) < workers:
        workers = len(ports)
    ports = ports[:workers]

    if not ensure_ps_ports(ports, start=start_ps or restart_ps, restart=restart_ps):
        hints = ', '.join(ps_start_hint(p) for p in ports)
        raise RuntimeError(
            f'Pokémon Showdown no responde en puertos {ports}. '
            f'Arranca: {hints}  o usa --start-ps / --restart-ps'
        )

    if export:
        export_teams()

    pool = active_teams()
    pairs: List[Tuple[TeamMeta, TeamMeta]] = disjoint_ab_pairings(n_teams, seed=seed)
    n_matchups = len(pairs)
    n_bo3 = n_matchups * 2
    state = StateCheckReport()
    sound = SoundnessReport()
    used_metas: List[TeamMeta] = []
    for a, b in pairs:
        used_metas.extend([a, b])
    flag_audit = audit_inference_flags(
        pool_abilities=collect_pool_abilities(used_metas),
    )

    stamp = int(time.time()) % 100000
    a_cls = validated_player_class(player_a)
    a_nick = {
        'greedy_solo': 'SoloR',
        'greedy_joint': 'JointR',
        'expectimax_bayesian': 'ExpBay',
    }.get(player_a, 'AR')
    b_nick = {
        'random': 'RndR',
        'greedy_solo': 'SoloO',
        'greedy_joint': 'JointO',
        'expectimax_bayesian': 'ExpBayO',
    }.get(player_b, 'BR')

    wins = 0
    losses = 0
    errors = 0
    pair_a_sweeps = 0
    pair_b_sweeps = 0
    pair_draws = 0
    pair_invalid = 0
    fail_kinds: Counter = Counter()
    pair_outcomes: List[int] = []
    details: List[dict] = []
    t0 = time.perf_counter()
    series_i = 0
    matchups_done = 0
    print_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()

    print(
        f'BATCH robust teams={n_teams} matchups={n_matchups} bo3={n_bo3} '
        f'(crossed) A={player_a} B={player_b} format={format_id} seed={seed} '
        f'workers={workers} ports={ports} quiet={int(quiet)}',
        flush=True,
    )

    async def _one_crossed_series(
        player_a_pl,
        player_b_pl,
        local_state: StateCheckReport,
        local_sound: SoundnessReport,
        ta: TeamMeta,
        tb: TeamMeta,
        *,
        label: str,
    ) -> dict:
        nonlocal wins, losses, errors, series_i
        row = {
            'pair': label,
            'player_a_won': False,
            'error': '',
            'state_mismatch': False,
            'inference_fail': False,
            'games': '',
            'outcome': 'ok',
        }
        player_a_pl.reset_for_next_battle(ta, tb)
        player_b_pl.update_team(to_showdown_export(tb))
        # Ambos lados pueden tener cola BO3 sucia tras un timeout previo.
        try:
            from showdown.inference_sim.bo3_runtime import reset_bo3_runtime_state
            reset_bo3_runtime_state(player_b_pl)
            if hasattr(player_b_pl, 'series_games'):
                player_b_pl.series_games = []
            if hasattr(player_b_pl, '_logged_game_tags'):
                player_b_pl._logged_game_tags = set()
        except Exception:
            pass

        wins_before = player_a_pl.n_won_battles
        finished_before = player_a_pl.n_finished_battles
        mm_before = local_state._total_mismatches
        feed_before = local_state.feed_errors
        fail_before = local_sound.failure_count

        outcome, err_detail = await _run_one_series(
            player_a_pl, player_b_pl, timeout_s=timeout_s,
        )
        row['outcome'] = outcome

        games_done = player_a_pl.n_finished_battles - finished_before
        game_wins = player_a_pl.n_won_battles - wins_before
        games_compact = (
            player_a_pl.series_games_compact()
            if hasattr(player_a_pl, 'series_games_compact')
            else '-'
        )
        row['games'] = games_compact

        async with print_lock:
            if outcome == 'ok' and games_done > 0:
                row['player_a_won'] = _series_won(
                    game_wins=game_wins, games_done=games_done,
                )
                if row['player_a_won']:
                    wins += 1
                    status = 'WIN'
                else:
                    losses += 1
                    status = 'LOSS'
            else:
                errors += 1
                local_sound.battle_errors += 1
                if outcome == 'ok':
                    row['error'] = 'no_finished_bo3'
                else:
                    row['error'] = err_detail or outcome
                status = 'ERR'
                try:
                    player_a_pl.reset_battles()
                    player_b_pl.reset_battles()
                except Exception:
                    pass

            row['state_mismatch'] = (
                local_state._total_mismatches > mm_before
                or local_state.feed_errors > feed_before
            )
            row['inference_fail'] = local_sound.failure_count > fail_before
            series_i += 1
            done_now = series_i
            w, l, e = wins, losses, errors
            if include_details or row['error'] or row['state_mismatch'] or row['inference_fail']:
                details.append(row)

        print(
            _compact_series_line(
                done=done_now,
                total=n_bo3,
                status=status,
                games=games_compact,
                wins=w,
                losses=l,
                errors=e,
                sync='OK' if not row['state_mismatch'] else 'FAIL',
                inf='OK' if not row['inference_fail'] else 'FAIL',
            )
            + (f' | {row["error"][:80]}' if row['error'] else '')
            + f' | {_ascii(label)[:70]}',
            flush=True,
        )
        return row

    async def _worker(
        worker_id: int,
        port: int,
        matchup_pairs: List[Tuple[TeamMeta, TeamMeta]],
    ):
        nonlocal matchups_done
        local_state = StateCheckReport()
        local_sound = SoundnessReport()
        local_fail_kinds: Counter = Counter()
        local_pair_outcomes: List[int] = []
        tallies = {
            'pair_a': 0,
            'pair_b': 0,
            'pair_draw': 0,
            'pair_invalid': 0,
        }
        if not matchup_pairs:
            return local_state, local_sound, local_fail_kinds, local_pair_outcomes, tallies

        cfg = server_configuration_for_port(port)
        team_a0, team_b0 = matchup_pairs[0]
        player_a_pl = a_cls(
            user_team=team_a0,
            opp_team=team_b0,
            state_report=local_state,
            soundness_report=local_sound,
            account_configuration=AccountConfiguration(
                f'{a_nick}{stamp % 10000}w{worker_id}'[:18], None
            ),
            battle_format=format_id,
            server_configuration=cfg,
        )
        player_b_pl = make_opponent_player(
            player_b,
            team=team_b0,
            account_configuration=AccountConfiguration(
                f'{b_nick}{stamp % 10000}w{worker_id}'[:18], None
            ),
            battle_format=format_id,
            server_configuration=cfg,
        )

        try:
            for i, (team_a, team_b) in enumerate(matchup_pairs):
                base_label = f'{team_a.name} vs {team_b.name}'
                fwd = await _one_crossed_series(
                    player_a_pl,
                    player_b_pl,
                    local_state,
                    local_sound,
                    team_a,
                    team_b,
                    label=f'{base_label} [A→B]',
                )
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
                rev = await _one_crossed_series(
                    player_a_pl,
                    player_b_pl,
                    local_state,
                    local_sound,
                    team_b,
                    team_a,
                    label=f'{base_label} [A←B]',
                )

                fail_fwd = _classify_series_fail(
                    outcome=fwd.get('outcome') or 'ok',
                    err_detail=fwd.get('error') or '',
                    state_mismatch=bool(fwd.get('state_mismatch')),
                    inference_fail=bool(fwd.get('inference_fail')),
                )
                fail_rev = _classify_series_fail(
                    outcome=rev.get('outcome') or 'ok',
                    err_detail=rev.get('error') or '',
                    state_mismatch=bool(rev.get('state_mismatch')),
                    inference_fail=bool(rev.get('inference_fail')),
                )
                fails = fail_fwd + fail_rev
                if fails:
                    tallies['pair_invalid'] += 1
                    for k in fails:
                        local_fail_kinds[k] += 1
                    pair_status = 'INVALID'
                else:
                    a_win_fwd = bool(fwd.get('player_a_won'))
                    a_win_rev = bool(rev.get('player_a_won'))
                    if a_win_fwd and a_win_rev:
                        tallies['pair_a'] += 1
                        local_pair_outcomes.append(1)
                        pair_status = 'PAIR_A'
                    elif (not a_win_fwd) and (not a_win_rev):
                        tallies['pair_b'] += 1
                        local_pair_outcomes.append(-1)
                        pair_status = 'PAIR_B'
                    else:
                        tallies['pair_draw'] += 1
                        local_pair_outcomes.append(0)
                        pair_status = 'PAIR_DRAW'

                if (i + 1) % max(1, progress_every) == 0:
                    try:
                        player_a_pl.reset_battles()
                        player_b_pl.reset_battles()
                    except Exception:
                        pass

                async with progress_lock:
                    matchups_done += 1
                    done_m = matchups_done
                    elapsed = time.perf_counter() - t0
                    _write_progress({
                        'done': done_m,
                        'total': n_matchups,
                        'series_done': series_i,
                        'series_total': n_bo3,
                        'player_a': player_a,
                        'player_b': player_b,
                        'workers': workers,
                        'ports': ports,
                        'wins': wins,
                        'losses': losses,
                        'errors': errors,
                        'elapsed_s': round(elapsed, 2),
                        'last_pair': _ascii(base_label),
                        'last_status': pair_status,
                        'last_games': f'{fwd.get("games")}|{rev.get("games")}',
                        'last_error': fwd.get('error') or rev.get('error') or '',
                        'worker': worker_id,
                        'port': port,
                    })

                if delay_s > 0 and i + 1 < len(matchup_pairs):
                    await asyncio.sleep(delay_s)
        finally:
            await _stop_players(player_a_pl, player_b_pl)

        return local_state, local_sound, local_fail_kinds, local_pair_outcomes, tallies

    chunks = _split_round_robin(pairs, workers)
    worker_results = await asyncio.gather(*[
        _worker(i, ports[i], chunks[i])
        for i in range(workers)
    ])

    for local_state, local_sound, local_fk, local_po, tallies in worker_results:
        _merge_state_reports(state, local_state)
        _merge_sound_reports(sound, local_sound)
        fail_kinds.update(local_fk)
        pair_outcomes.extend(local_po)
        pair_a_sweeps += tallies['pair_a']
        pair_b_sweeps += tallies['pair_b']
        pair_draws += tallies['pair_draw']
        pair_invalid += tallies['pair_invalid']

    elapsed = time.perf_counter() - t0
    report = _build_report(
        n_teams=n_teams,
        n_bo3=n_bo3,
        n_matchups=n_matchups,
        format_id=format_id,
        pool_n=len(pool),
        wins=wins,
        losses=losses,
        errors=errors,
        pair_a=pair_a_sweeps,
        pair_b=pair_b_sweeps,
        pair_draw=pair_draws,
        pair_invalid=pair_invalid,
        fail_kinds=fail_kinds,
        pair_outcomes=pair_outcomes,
        elapsed_s=elapsed,
        seed=seed,
        state=state,
        sound=sound,
        details=details,
        include_details=include_details,
        flags=flag_audit,
        player_a=player_a,
        player_b=player_b,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='utf-8')
    fixplan = write_fixplan(report)

    decided = wins + losses
    wr_raw = (100.0 * wins / decided) if decided else 0.0
    need = 'YES' if 'NEED_FIX=YES' in fixplan.read_text(encoding='utf-8') else 'NO'
    sync_ok = 'YES' if state.ok else 'NO'
    inf_ok = 'YES' if sound.ok else 'NO'
    try:
        from showdown.battle_bots.expectimax_bayesian.config_history import (
            record_e2e_run,
        )
        record_e2e_run(
            player_a=player_a,
            player_b=player_b,
            teams=n_teams,
            wins=wins,
            losses=losses,
            wr_raw=wr_raw,
            pair_a=pair_a_sweeps,
            pair_b=pair_b_sweeps,
            pair_draw=pair_draws,
            need_fix=need,
            sync_ok=sync_ok,
            inf_ok=inf_ok,
        )
    except Exception:
        pass
    print(
        f'DONE teams={n_teams} A={player_a} B={player_b}\n'
        f'bo3={n_bo3}: W={wins} L={losses} E={errors} WR_raw={wr_raw:.1f}%\n'
        f'matchups={n_matchups}: pair_A={pair_a_sweeps} pair_B={pair_b_sweeps} '
        f'draw={pair_draws} invalid={pair_invalid}\n'
        f'sync_ok={sync_ok} fail_sync={fail_kinds.get("sync", 0)}\n'
        f'inf_ok={inf_ok} fail_matchup_inf={fail_kinds.get("inference", 0)} '
        f'fail_bo3_inf={sound.failure_count}\n'
        f'NEED_FIX={need} NEED_FLAGS_ADD={flag_audit.flag_line()} '
        f'elapsed_s={elapsed:.1f} workers={workers} ports={ports}',
        flush=True,
    )
    return report

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Robust E2E: disjoint A/B pools, crossed BO3 pairs, '
            'Battle sync + OTS inference'
        )
    )
    parser.add_argument(
        '--teams',
        type=int,
        default=100,
        help='N equipos (par). Genera N/2 matchups × 2 BO3 cruzados',
    )
    parser.add_argument(
        '--player-a',
        choices=list(PLAYER_A_CHOICES),
        default='greedy_solo',
        help='Agente validated (sync+OTS). Default: greedy_solo',
    )
    parser.add_argument(
        '--player-b',
        choices=list(PLAYER_B_CHOICES),
        default='random',
        help='Oponente. Default: random',
    )
    parser.add_argument('--format', default=DEFAULT_FORMAT)
    parser.add_argument('--start-ps', action='store_true')
    parser.add_argument('--restart-ps', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--delay', type=float, default=0.0)
    parser.add_argument(
        '--timeout',
        type=float,
        default=180.0,
        help='Timeout por serie BO3 (s). Default 180; sube si expectimax es lento.',
    )
    parser.add_argument('--progress-every', type=int, default=5)
    parser.add_argument('--details', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Workers paralelos (default 2). Puertos: PS_PORTS o PS_PORT..+N-1',
    )
    args = parser.parse_args(argv)

    if args.export_only:
        out = export_teams()
        print(f'Exported teams -> {out}')
        return 0

    quiet = not args.verbose
    ports = parse_ps_ports(workers=max(1, args.workers))
    if not quiet:
        print(
            f'PS ports={ports} up={[is_ps_up(port=p) for p in ports]} | '
            f'format={args.format} | '
            f'teams={args.teams} matchups={args.teams // 2} '
            f'bo3={args.teams} (crossed) '
            f'A={args.player_a} B={args.player_b} workers={args.workers}'
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
            player_a=args.player_a,
            player_b=args.player_b,
            workers=args.workers,
        )
    )

    text = REPORT_PATH.read_text(encoding='utf-8') if REPORT_PATH.exists() else ''
    sync_bad = (
        'sync_ok: NO' in text
        or 'state_mismatch: YES' in text
    )
    inf_bad = (
        'inf_ok: NO' in text
        or 'inference_ok: NO' in text
    )
    if sync_bad or inf_bad:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
