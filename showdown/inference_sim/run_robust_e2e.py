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
  bo3=N: W=... L=... E=... S=... WR_raw=...%
  matchups=N: pair_A=... pair_B=... draw=... invalid=...
  sync_ok=YES|NO fail_sync=...
  inf_ok=YES|NO fail_matchup_inf=... fail_bo3_inf=...
  NEED_FIX=YES|NO NEED_FLAGS_ADD=YES|NO elapsed_s=...

  S = series saltadas (short_bo3 / partial_timeout / empty); no cuentan W/L.
  Cortas (<2 games): 1 reintento. BO3 ya decidido: forfeit de games huérfanos.
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
from pathlib import Path
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
    skips: int,
    sync: str,
    inf: str,
) -> str:
    return (
        f'[{done}/{total}] {status} {games} | '
        f'W={wins} L={losses} E={errors} S={skips} sync={sync} inf={inf}'
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


async def _cancel_challenges(*players, drain_queue: bool = True) -> None:
    """Limpia salas entre series (no spamear cancelchallenge/reject).

    Tras un BO3 normal no hay challenge pendiente: enviar
    '/cancelchallenge' + '/reject' produce popups
    «You are not challenging…» / «…is not challenging you».
    Basta con /leave de rooms conocidas + drenar cola BO3 al abandonar.
    """
    from showdown.inference_sim.bo3_runtime import reset_bo3_runtime_state_many

    pls = [pl for pl in players if pl is not None]
    for pl in pls:
        try:
            battles = getattr(pl, 'battles', None) or {}
            for tag in list(battles.keys()):
                if not tag:
                    continue
                # Marcar stale ANTES del leave: un choose en vuelo no debe
                # reenviar /choose a la sala que acabamos de abandonar.
                mark = getattr(pl, '_mark_room_left', None)
                if callable(mark):
                    try:
                        mark(tag)
                    except Exception:
                        pass
                try:
                    await pl.ps_client.send_message(f'/leave {tag}')
                except Exception:
                    pass
            # También salir de salas padre game-bestof* huérfanas.
            for tag in list(getattr(pl, '_bestof_games', None) or []):
                if not tag:
                    continue
                try:
                    await pl.ps_client.send_message(f'/leave {tag}')
                except Exception:
                    pass
        except Exception:
            pass
    # Drenar cola solo al abandonar (timeout/retry). No drenar si la serie
    # sigue viva: join() devolvería 'ok' prematuro → short_bo3.
    reset_bo3_runtime_state_many(pls, drain_queue=drain_queue)


# Timeout serie BO3: baseline histórico con deadline_ms=550 → 420s.
# Escala con ExpectimaxConfig.deadline_ms (cobertura 75% ≈ 1600ms → ~1220s).
_EXPECTIMAX_SERIES_TIMEOUT_BASELINE_S = 420.0
_EXPECTIMAX_SERIES_TIMEOUT_BASELINE_DEADLINE_MS = 550.0
# Validated greedy (sync+OTS+calc) en BO3 1-1→g3: 180s provocaba short_bo3.
_GREEDY_SERIES_TIMEOUT_S = 480.0
_MAX_SERIES_ATTEMPTS = 3


def _expectimax_deadline_ms() -> float:
    try:
        from showdown.battle_bots.expectimax_bayesian.expectimax import load_expectimax_config
        return float(load_expectimax_config().deadline_ms)
    except Exception:
        return _EXPECTIMAX_SERIES_TIMEOUT_BASELINE_DEADLINE_MS


def _expectimax_turn_wall_s(deadline_ms: Optional[float] = None) -> float:
    """Mismo techo que pokeenv `_choose_move_guarded` (deadline + margen)."""
    dl = float(deadline_ms) if deadline_ms is not None else _expectimax_deadline_ms()
    return max(2.5, dl / 1000.0 + 2.0)


def default_series_timeout_s(*, uses_expectimax: bool) -> float:
    """Timeout por serie BO3 según si hay expectimax (sigue a deadline_ms)."""
    if not uses_expectimax:
        return float(_GREEDY_SERIES_TIMEOUT_S)
    dl = _expectimax_deadline_ms()
    scaled = (
        _EXPECTIMAX_SERIES_TIMEOUT_BASELINE_S
        * (dl / _EXPECTIMAX_SERIES_TIMEOUT_BASELINE_DEADLINE_MS)
    )
    # Mínimo el baseline; tope 40min/serie para batch local.
    return float(max(_EXPECTIMAX_SERIES_TIMEOUT_BASELINE_S, min(scaled, 2400.0)))


def _series_activity_fingerprint(player) -> Tuple[int, Tuple[Tuple[str, int], ...], int]:
    """Señal de progreso mid-BO3: games cerrados + turnos de hijas activas."""
    games = list(getattr(player, 'series_games', None) or [])
    battles = getattr(player, 'battles', None) or {}
    active: List[Tuple[str, int]] = []
    for tag, bat in battles.items():
        if getattr(bat, 'finished', False):
            continue
        key = str(tag or '').lstrip('>')
        if not key.startswith('battle-'):
            continue
        active.append((key, int(getattr(bat, 'turn', 0) or 0)))
    active.sort()
    bestof_n = len(getattr(player, '_bestof_games', None) or ())
    return (len(games), tuple(active), bestof_n)


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
    last_fp = _series_activity_fingerprint(player_a_pl)
    last_progress = loop.time()
    decided_since: Optional[float] = None
    # Stall mid-game: un solo game puede durar muchos turnos × wall expectimax.
    turn_wall = _expectimax_turn_wall_s()
    stall_floor = max(120.0, turn_wall * 90.0 + 60.0)
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
                await _cancel_challenges(player_a_pl, player_b_pl, drain_queue=True)
                return 'reject', reject_bucket.get('msg') or 'team_rejected'
            if battle_task in done_set:
                exc = battle_task.exception()
                if exc is not None:
                    await _cancel_challenges(player_a_pl, player_b_pl, drain_queue=True)
                    games = list(getattr(player_a_pl, 'series_games', None) or [])
                    # Serie ya decidida: excepción al cerrar/cola no es ERR.
                    if bo3_series_decided(games):
                        return 'ok', f'forced_ok_after_exc:{type(exc).__name__}'
                    return 'error', repr(exc)
                games = list(getattr(player_a_pl, 'series_games', None) or [])
                # join() prematuro (cola drenada) → no tratar como ok limpio.
                if not bo3_series_decided(games):
                    await _cancel_challenges(player_a_pl, player_b_pl, drain_queue=True)
                    return 'error', f'short_bo3_early_join:games={len(games)}'
                return 'ok', ''

            fp = _series_activity_fingerprint(player_a_pl)
            if fp != last_fp:
                last_fp = fp
                last_progress = loop.time()
            games = list(getattr(player_a_pl, 'series_games', None) or [])
            n_games = len(games)
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
                    await _cancel_challenges(player_a_pl, player_b_pl, drain_queue=True)
                    return 'ok', 'forced_ok_after_bo3_hang'
            else:
                decided_since = None
                # Mid-series stall: floor por wall×turnos; techo con --timeout.
                stall_s = max(
                    stall_floor,
                    min(float(timeout_s) * 0.6, float(timeout_s) - 30.0),
                )
                if n_games >= 1 and (loop.time() - last_progress) >= stall_s:
                    break
                # Sin ningún game tras gran parte del budget → reto/preview colgado.
                if (
                    n_games == 0
                    and (loop.time() - last_progress)
                    >= max(stall_floor * 0.5, float(timeout_s) * 0.5)
                ):
                    break

        # timeout
        battle_task.cancel()
        reject_task.cancel()
        try:
            await battle_task
        except (asyncio.CancelledError, Exception):
            pass
        await _cancel_challenges(player_a_pl, player_b_pl, drain_queue=True)
        # Si ya hubo progreso decisivo en series_games, no marcar timeout duro.
        games = list(getattr(player_a_pl, 'series_games', None) or [])
        if bo3_series_decided(games):
            return 'ok', 'forced_ok_on_timeout_after_bo3_decided'
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
    reward_rows: Optional[List] = None,
    reward_analysis_lines: Optional[List[str]] = None,
    skip_sync: bool = False,
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
        f'skip_sync: {"YES" if skip_sync else "NO"}',
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
    ]
    try:
        from showdown.inference_sim.reward_telemetry import format_reward_report_section

        lines.extend(
            format_reward_report_section(
                reward_rows or [],
                player_a=player_a,
            )
        )
        lines.append('')
    except Exception:
        lines.extend([
            '--- reward telemetry (player_a perspective, Spec v1.1) ---',
            'reward_games: 0',
            'reward_error: unavailable',
            '',
        ])
    lines += [
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
        f'sync_ok: {"SKIP" if skip_sync else ("YES" if state.ok else "NO")}',
        f'state_mismatch: {"NO" if skip_sync else state.flag_line()}',
        f'state_checks: {state.checks}',
        f'state_mismatches: {state._total_mismatches}',
        f'feed_errors: {state.feed_errors}',
        '',
        '--- inference (OTS ground truth) ---',
        f'inf_ok: {"SKIP" if skip_sync else ("YES" if sound.ok else "NO")}',
        f'inference_ok: {"SKIP" if skip_sync else ("YES" if sound.ok else "NO")}',
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
    else:
        # Siempre dejar muestras de ERR (fixplan runtime) aunque no --details.
        err_rows = [r for r in details if r.get('error')]
        if err_rows:
            lines.append('')
            lines.append('--- runtime error samples (capped) ---')
            for i, row in enumerate(err_rows[:10], start=1):
                lines.append(
                    f'{i}. {row.get("games") or "-"} | {_ascii(row.get("error") or "")} | '
                    f'{_ascii(row.get("pair") or "")}'
                )

    if reward_analysis_lines:
        lines.append('')
        lines.extend(reward_analysis_lines)

    lines.append('')
    lines.append(
        f'Player A = validated({player_a}'
        f'{" skip_sync" if skip_sync else ""}) | Player B = {player_b}'
    )
    return '\n'.join(lines) + '\n'


async def _stop_players(*players) -> None:
    from showdown.inference_sim.ps_client_shutdown import stop_players

    await stop_players(*players)


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
    skip_sync: bool = False,
    shard: Optional[Tuple[int, int]] = None,
    shard_out: Optional[str] = None,
) -> str:
    if n_teams % 2 != 0 or n_teams < 2:
        raise ValueError('--teams debe ser un entero par >= 2')
    worker_offset = 0

    set_chat_quiet(quiet)
    workers = max(1, int(workers))
    skip_sync = bool(skip_sync)
    try:
        from showdown.inference_sim import calibration
        calibration.install()  # solo si PS_CALIB_LOG está definido
    except Exception:
        pass
    if seed is not None:
        from showdown.inference_sim.ps_server import seed_python_rngs
        seed_python_rngs(int(seed) + 7919 * (int(shard[0]) if shard else 0))
    # Silenciar popups / CancelledError cosméticos en loggers poke-env y root.
    try:
        from showdown.inference_sim.stale_room_guard import _StaleRoomPopupFilter
        filt = _StaleRoomPopupFilter()
        filt.name = '_stale_room_popup_filter'  # type: ignore[attr-defined]
        for name in ('poke-env', 'poke_env', ''):
            lg = logging.getLogger(name)
            if not any(getattr(f, 'name', '') == filt.name for f in lg.filters):
                lg.addFilter(filt)
    except Exception:
        pass
    ports = parse_ps_ports(workers=workers)
    if len(ports) < workers:
        workers = len(ports)
    ports = ports[:workers]

    if 'greedy_joint' in (player_a, player_b):
        try:
            from showdown.battle_bots.greedy_joint.diag import reset as reset_joint_diag
            reset_joint_diag()
        except Exception:
            pass

    if 'expectimax' in str(player_a).lower() or 'expectimax' in str(player_b).lower():
        try:
            from showdown.battle_bots.expectimax_bayesian.divergence import (
                reset_divergence_stats,
            )
            reset_divergence_stats()
        except Exception:
            pass

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
    if shard is not None:
        k, n_shards = int(shard[0]), int(shard[1])
        pairs = pairs[k::n_shards]
        worker_offset = k * max(1, int(workers))
    n_matchups = len(pairs)
    n_bo3 = n_matchups * 2
    state = StateCheckReport()
    sound = SoundnessReport()
    # --skip-sync: no contrast ni OTS (soundness_report=None en players).
    sound_for_players: Optional[SoundnessReport] = None if skip_sync else sound
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
    skips = 0
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
        f'workers={workers} ports={ports} timeout_s={timeout_s} quiet={int(quiet)} '
        f'skip_sync={int(skip_sync)}',
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
        nonlocal wins, losses, errors, skips, series_i
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

        from showdown.inference_sim.bo3_runtime import bo3_series_decided as _decided
        outcome, err_detail = 'error', 'no_attempt'
        attempt_notes: List[str] = []
        for attempt in range(1, _MAX_SERIES_ATTEMPTS + 1):
            if attempt > 1:
                try:
                    await _cancel_challenges(
                        player_a_pl, player_b_pl, drain_queue=True,
                    )
                    player_a_pl.reset_battles()
                    player_b_pl.reset_battles()
                except Exception:
                    pass
                player_a_pl.reset_for_next_battle(ta, tb)
                player_b_pl.update_team(to_showdown_export(tb))
                try:
                    from showdown.inference_sim.bo3_runtime import (
                        reset_bo3_runtime_state,
                    )
                    reset_bo3_runtime_state(player_b_pl)
                    if hasattr(player_b_pl, 'series_games'):
                        player_b_pl.series_games = []
                    if hasattr(player_b_pl, '_logged_game_tags'):
                        player_b_pl._logged_game_tags = set()
                except Exception:
                    pass
                wins_before = player_a_pl.n_won_battles
                finished_before = player_a_pl.n_finished_battles

            outcome, err_detail = await _run_one_series(
                player_a_pl, player_b_pl, timeout_s=timeout_s,
            )
            series_games_try = list(
                getattr(player_a_pl, 'series_games', None) or []
            )
            if _decided(series_games_try) and outcome in (
                'ok', 'timeout', 'error',
            ):
                if attempt > 1:
                    attempt_notes.append(f'ok_after_attempt_{attempt}')
                break
            # Serie incompleta: reintentar (short_bo3 / timeout / early join).
            note = err_detail or outcome or 'undecided'
            attempt_notes.append(f'a{attempt}:{note}')
            if attempt >= _MAX_SERIES_ATTEMPTS:
                break

        if attempt_notes and not (
            _decided(list(getattr(player_a_pl, 'series_games', None) or []))
            and outcome == 'ok'
        ):
            err_detail = (
                f'retry_x{len(attempt_notes)}:'
                + ','.join(attempt_notes[-3:])
            )
        elif attempt_notes and outcome == 'ok':
            err_detail = ','.join(attempt_notes[-2:]) if len(attempt_notes) > 1 else ''
        row['outcome'] = outcome

        games_done = player_a_pl.n_finished_battles - finished_before
        game_wins = player_a_pl.n_won_battles - wins_before
        series_games = list(getattr(player_a_pl, 'series_games', None) or [])
        # BO3 ya decidido en series_games pero poke-env no incrementó n_finished
        # (hang post-|win| / forced_ok): no marcar no_finished_bo3.
        from showdown.inference_sim.bo3_runtime import bo3_series_decided
        # BO3 ya decidido en series_games: timeout/exc post-|win| no es ERR.
        if bo3_series_decided(series_games) and (
            outcome in ('ok', 'timeout', 'error')
            or str(err_detail).startswith('forced_ok')
        ):
            games_done = max(int(games_done), len(series_games))
            game_wins = sum(1 for _t, won in series_games if won)
            if outcome != 'ok':
                outcome = 'ok'
                row['outcome'] = 'ok'
                err_detail = err_detail or 'counted_from_series_games'
        # Timeout / early-join a mitad de serie: NO contar como W/L.
        elif (
            outcome in ('timeout', 'error')
            and series_games
            and not bo3_series_decided(series_games)
        ):
            games_done = len(series_games)
            game_wins = sum(1 for _t, won in series_games if won)
            outcome = 'partial_timeout'
            row['outcome'] = outcome
            err_detail = err_detail or 'partial_bo3_timeout'
            row['error'] = err_detail

        games_compact = (
            player_a_pl.series_games_compact()
            if hasattr(player_a_pl, 'series_games_compact')
            else '-'
        )
        row['games'] = games_compact
        row['rewards'] = list(getattr(player_a_pl, 'series_rewards', None) or [])

        def _counts_as_decided_bo3(n_done: int, n_wins: int) -> bool:
            n_loss = int(n_done) - int(n_wins)
            return int(n_done) >= 3 or int(n_wins) >= 2 or n_loss >= 2

        async with print_lock:
            decided = _counts_as_decided_bo3(games_done, game_wins) or bo3_series_decided(
                series_games
            )
            if outcome == 'ok' and games_done > 0 and decided:
                row['player_a_won'] = _series_won(
                    game_wins=game_wins, games_done=games_done,
                )
                if row['player_a_won']:
                    wins += 1
                    status = 'WIN'
                else:
                    losses += 1
                    status = 'LOSS'
            elif outcome == 'ok' and games_done > 0 and not decided:
                row['error'] = row.get('error') or err_detail or 'short_bo3'
                status = 'SKIP'
                skips += 1
                try:
                    player_a_pl.reset_battles()
                    player_b_pl.reset_battles()
                except Exception:
                    pass
            elif (
                outcome == 'partial_timeout'
                or (
                    outcome in ('ok', 'timeout')
                    and games_done <= 0
                    and not series_games
                )
            ):
                if not row.get('error'):
                    row['error'] = (
                        'empty_bo3'
                        if outcome == 'ok'
                        else (err_detail or 'timeout_empty')
                    )
                status = 'SKIP'
                skips += 1
                try:
                    player_a_pl.reset_battles()
                    player_b_pl.reset_battles()
                except Exception:
                    pass
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
                False if skip_sync else (
                    local_state._total_mismatches > mm_before
                    or local_state.feed_errors > feed_before
                )
            )
            row['inference_fail'] = (
                False if skip_sync else (local_sound.failure_count > fail_before)
            )
            series_i += 1
            done_now = series_i
            w, l, e, s = wins, losses, errors, skips
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
                skips=s,
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
            return (
                local_state,
                local_sound,
                local_fail_kinds,
                local_pair_outcomes,
                tallies,
                [],
            )

        cfg = server_configuration_for_port(port)
        team_a0, team_b0 = matchup_pairs[0]
        player_a_pl = a_cls(
            user_team=team_a0,
            opp_team=team_b0,
            state_report=local_state,
            soundness_report=sound_for_players,
            skip_sync=skip_sync,
            account_configuration=AccountConfiguration(
                f'{a_nick}{stamp % 10000}w{worker_id + worker_offset}'[:18], None
            ),
            battle_format=format_id,
            server_configuration=cfg,
        )
        player_b_pl = make_opponent_player(
            player_b,
            team=team_b0,
            account_configuration=AccountConfiguration(
                f'{b_nick}{stamp % 10000}w{worker_id + worker_offset}'[:18], None
            ),
            battle_format=format_id,
            server_configuration=cfg,
        )

        try:
            from showdown.inference_sim import decision_trace
            decision_trace.install(player_a_pl, 'A')
            decision_trace.install(player_b_pl, 'B')
            from showdown.inference_sim import opp_model_trace
            if hasattr(player_a_pl, 'set_opp_ots_sheet'):  # solo jugadores expectimax
                opp_model_trace.install(player_a_pl)
            if hasattr(player_b_pl, 'set_opp_ots_sheet'):
                opp_model_trace.install(player_b_pl)
        except Exception:
            pass
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
                        'skips': skips,
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

        reward_rows = list(getattr(player_a_pl, 'all_rewards', None) or [])
        return (
            local_state,
            local_sound,
            local_fail_kinds,
            local_pair_outcomes,
            tallies,
            reward_rows,
        )

    chunks = _split_round_robin(pairs, workers)
    worker_results = await asyncio.gather(*[
        _worker(i, ports[i], chunks[i])
        for i in range(workers)
    ])

    all_reward_rows: List = []
    for local_state, local_sound, local_fk, local_po, tallies, reward_rows in worker_results:
        _merge_state_reports(state, local_state)
        _merge_sound_reports(sound, local_sound)
        fail_kinds.update(local_fk)
        pair_outcomes.extend(local_po)
        pair_a_sweeps += tallies['pair_a']
        pair_b_sweeps += tallies['pair_b']
        pair_draws += tallies['pair_draw']
        pair_invalid += tallies['pair_invalid']
        all_reward_rows.extend(reward_rows or [])

    elapsed = time.perf_counter() - t0
    if shard_out:
        _dump_shard_payload(
            shard_out,
            {
                'wins': wins, 'losses': losses, 'errors': errors, 'skips': skips,
                'pair_a': pair_a_sweeps, 'pair_b': pair_b_sweeps,
                'pair_draw': pair_draws, 'pair_invalid': pair_invalid,
                'fail_kinds': dict(fail_kinds), 'pair_outcomes': list(pair_outcomes),
                'details': list(details), 'state': state, 'sound': sound,
                'reward_rows': list(all_reward_rows), 'elapsed_s': elapsed,
                'n_matchups': n_matchups,
                'joint_diag': _joint_diag_snapshot(player_a, player_b),
                'divergence': _divergence_snapshot(player_a, player_b),
            },
        )
        return ''
    return _finalize_batch(
        n_teams=n_teams, n_bo3=n_bo3, n_matchups=n_matchups, format_id=format_id,
        pool_n=len(pool), wins=wins, losses=losses, errors=errors, skips=skips,
        pair_a_sweeps=pair_a_sweeps, pair_b_sweeps=pair_b_sweeps,
        pair_draws=pair_draws, pair_invalid=pair_invalid, fail_kinds=fail_kinds,
        pair_outcomes=pair_outcomes, elapsed=elapsed, seed=seed, state=state,
        sound=sound, details=details, include_details=include_details,
        flag_audit=flag_audit, player_a=player_a, player_b=player_b,
        all_reward_rows=all_reward_rows, skip_sync=skip_sync,
        workers=workers, ports=ports,
    )


def _finalize_batch(
    *,
    n_teams: int,
    n_bo3: int,
    n_matchups: int,
    format_id: str,
    pool_n: int,
    wins: int,
    losses: int,
    errors: int,
    skips: int,
    pair_a_sweeps: int,
    pair_b_sweeps: int,
    pair_draws: int,
    pair_invalid: int,
    fail_kinds: Counter,
    pair_outcomes: List[int],
    elapsed: float,
    seed: Optional[int],
    state: StateCheckReport,
    sound: SoundnessReport,
    details: List[dict],
    include_details: bool,
    flag_audit,
    player_a: str,
    player_b: str,
    all_reward_rows: List,
    skip_sync: bool,
    workers: int,
    ports: List[int],
) -> str:
    """Reporte, fixplan, diagnósticos y resumen final (común a 1 o K procesos)."""
    analysis_lines: List[str] = []
    divergence_lines: List[str] = []
    try:
        from showdown.inference_sim.reward_telemetry import (
            build_reward_run_record,
            format_reward_analysis,
            load_reward_history,
            write_reward_analysis,
        )

        rec = build_reward_run_record(
            player_a=player_a,
            player_b=player_b,
            teams=n_teams,
            seed=seed,
            wins=wins,
            losses=losses,
            errors=errors,
            reward_rows=all_reward_rows,
            elapsed_s=elapsed,
        )
        hist_before = load_reward_history(matchup=str(rec.get('matchup') or ''), limit=40)
        write_reward_analysis(rec, history_before=hist_before)
        analysis_lines = format_reward_analysis(rec, history=hist_before + [rec])
    except Exception as exc:
        analysis_lines = [
            '--- reward analysis (auto) ---',
            f'reward_analysis_error: {exc!r}',
        ]
    if 'expectimax' in str(player_a).lower() or 'expectimax' in str(player_b).lower():
        try:
            from showdown.battle_bots.expectimax_bayesian.divergence import (
                format_divergence_summary,
                write_divergence_summary,
            )

            write_divergence_summary(extra={
                'player_a': player_a,
                'player_b': player_b,
                'teams': n_teams,
                'wr_raw': (100.0 * wins / (wins + losses)) if (wins + losses) else 0.0,
            })
            divergence_lines = format_divergence_summary()
            if analysis_lines:
                analysis_lines = list(analysis_lines) + [''] + list(divergence_lines)
            else:
                analysis_lines = list(divergence_lines)
        except Exception as exc:
            divergence_lines = [
                '--- expectimax divergence ---',
                f'divergence_error: {exc!r}',
            ]
            analysis_lines = list(analysis_lines or []) + [''] + divergence_lines

    report = _build_report(
        n_teams=n_teams,
        n_bo3=n_bo3,
        n_matchups=n_matchups,
        format_id=format_id,
        pool_n=pool_n,
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
        reward_rows=all_reward_rows,
        reward_analysis_lines=analysis_lines,
        skip_sync=skip_sync,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='utf-8')
    fixplan = write_fixplan(report)

    joint_diag_path = ''
    if 'greedy_joint' in (player_a, player_b):
        try:
            from showdown.battle_bots.greedy_joint.diag import dump as dump_joint_diag
            jp = dump_joint_diag(REPORT_PATH.parent / 'greedy_joint_pair_diag.txt')
            joint_diag_path = str(jp)
        except Exception as exc:
            joint_diag_path = f'error:{exc!r}'

    decided = wins + losses
    wr_raw = (100.0 * wins / decided) if decided else 0.0
    need = 'YES' if 'NEED_FIX=YES' in fixplan.read_text(encoding='utf-8') else 'NO'
    sync_ok = 'SKIP' if skip_sync else ('YES' if state.ok else 'NO')
    inf_ok = 'SKIP' if skip_sync else ('YES' if sound.ok else 'NO')
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
        f'bo3={n_bo3}: W={wins} L={losses} E={errors} S={skips} WR_raw={wr_raw:.1f}%\n'
        f'matchups={n_matchups}: pair_A={pair_a_sweeps} pair_B={pair_b_sweeps} '
        f'draw={pair_draws} invalid={pair_invalid}\n'
        f'sync_ok={sync_ok} fail_sync={fail_kinds.get("sync", 0)}'
        f'{" (skipped)" if skip_sync else ""}\n'
        f'inf_ok={inf_ok} fail_matchup_inf={fail_kinds.get("inference", 0)} '
        f'fail_bo3_inf={sound.failure_count}\n'
        f'NEED_FIX={need} NEED_FLAGS_ADD={flag_audit.flag_line()} '
        f'elapsed_s={elapsed:.1f} workers={workers} ports={ports}\n'
        f'reward_analysis={REPORT_PATH.parent / "robust_e2e_reward_analysis.txt"}'
        + (
            f'\njoint_pair_diag={joint_diag_path}'
            if joint_diag_path else ''
        )
        + (
            f'\ndivergence={REPORT_PATH.parent / "expectimax_divergence.txt"}'
            if divergence_lines else ''
        ),
        flush=True,
    )
    if divergence_lines:
        # Compact one-liner for terminal (full block in txt/report).
        compact = [
            ln for ln in divergence_lines
            if ln.startswith('div_') or ln.startswith('solo_')
        ]
        if compact:
            print(' | '.join(compact), flush=True)
    return report

def _joint_diag_snapshot(player_a: str, player_b: str):
    if 'greedy_joint' not in (player_a, player_b):
        return None
    try:
        from showdown.battle_bots.greedy_joint.diag import snapshot
        return snapshot()
    except Exception:
        return None


def _divergence_snapshot(player_a: str, player_b: str):
    if 'expectimax' not in f'{player_a} {player_b}'.lower():
        return None
    try:
        from showdown.battle_bots.expectimax_bayesian.divergence import (
            snapshot_divergence_stats,
        )
        return snapshot_divergence_stats()
    except Exception:
        return None


def _dump_shard_payload(path: str, payload: Dict[str, Any]) -> None:
    import pickle

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('wb') as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)


def run_batch_multiprocess(
    n_teams: int,
    *,
    procs: int,
    workers_per_proc: int,
    format_id: str,
    start_ps: bool,
    restart_ps: bool,
    seed: Optional[int],
    timeout_s: float,
    include_details: bool,
    export: bool,
    quiet: bool,
    player_a: str,
    player_b: str,
    skip_sync: bool,
    delay_s: float = 0.0,
    progress_every: int = 5,
) -> str:
    """Reparte los matchups en ``procs`` procesos (un núcleo de Python cada uno).

    Los workers de :func:`run_batch` son corrutinas de un solo proceso: solapan
    la espera de red, pero las decisiones de greedy_joint / expectimax compiten
    por un único núcleo. Aquí cada proceso recibe ``matchups[k::procs]`` y su
    propio bloque de puertos Showdown; el padre combina y escribe un único
    reporte, idéntico al de una corrida de un proceso.
    """
    import pickle
    import random as _random
    import subprocess as _sp
    import tempfile

    procs = max(1, int(procs))
    wpp = max(1, int(workers_per_proc))
    if seed is None:
        # Todos los procesos deben generar exactamente los mismos pares.
        seed = _random.randrange(1, 2**31 - 1)
        if 'PS_BATTLE_SEED' not in os.environ and not os.environ.get('PS_NO_BATTLE_SEED'):
            from showdown.inference_sim.ps_server import set_battle_seed
            set_battle_seed(seed)
    from showdown.inference_sim.paths import PS_PORT

    base = int(PS_PORT)
    blocks = [[base + k * wpp + j for j in range(wpp)] for k in range(procs)]
    all_ports = [p for b in blocks for p in b]
    if not ensure_ps_ports(all_ports, start=start_ps or restart_ps, restart=restart_ps):
        hints = ', '.join(ps_start_hint(p) for p in all_ports)
        raise RuntimeError(f'Pokémon Showdown no responde en {all_ports}. Arranca: {hints}')
    if export:
        export_teams()

    if 'greedy_joint' in (player_a, player_b):
        try:
            from showdown.battle_bots.greedy_joint.diag import reset as reset_joint_diag
            reset_joint_diag()
        except Exception:
            pass
    if 'expectimax' in f'{player_a} {player_b}'.lower():
        try:
            from showdown.battle_bots.expectimax_bayesian.divergence import (
                reset_divergence_stats,
            )
            reset_divergence_stats()
        except Exception:
            pass

    n_matchups_total = len(disjoint_ab_pairings(n_teams, seed=seed))
    procs = min(procs, max(1, n_matchups_total))
    print(
        f'BATCH robust (multiproceso) teams={n_teams} matchups={n_matchups_total} '
        f'bo3={n_matchups_total * 2} procs={procs} workers/proc={wpp} '
        f'A={player_a} B={player_b} seed={seed} ports={all_ports[:procs * wpp]} '
        f'timeout_s={timeout_s} skip_sync={int(skip_sync)}',
        flush=True,
    )
    t0 = time.perf_counter()
    tmpdir = Path(tempfile.mkdtemp(prefix='robust_e2e_shards_'))
    children = []
    for k in range(procs):
        out = tmpdir / f'shard_{k}.pkl'
        argv = [
            sys.executable, '-m', 'showdown.inference_sim.run_robust_e2e',
            '--teams', str(n_teams),
            '--player-a', player_a, '--player-b', player_b,
            '--format', format_id, '--seed', str(seed),
            '--timeout', str(timeout_s), '--workers', str(wpp),
            '--delay', str(delay_s), '--progress-every', str(progress_every),
            '--shard', f'{k}/{procs}', '--shard-out', str(out),
        ]
        if skip_sync:
            argv.append('--skip-sync')
        if include_details:
            argv.append('--details')
        if not quiet:
            argv.append('--verbose')
        env = dict(os.environ)
        env['PS_PORTS'] = ','.join(str(p) for p in blocks[k])
        env.setdefault('PYTHONIOENCODING', 'utf-8')
        children.append((k, out, _sp.Popen(argv, env=env)))

    failures = []
    for k, out, proc in children:
        rc = proc.wait()
        # rc=1 en un hijo solo significa sync/inferencia NO en su porción: el
        # reporte combinado lo reflejará. Solo falta el payload si el hijo cayó.
        if not out.exists():
            failures.append(f'shard {k} sin resultados (exit={rc})')
    if failures:
        raise RuntimeError('; '.join(failures))

    wins = losses = errors = skips = 0
    pair_a = pair_b = pair_draw = pair_invalid = 0
    fail_kinds: Counter = Counter()
    pair_outcomes: List[int] = []
    details: List[dict] = []
    all_reward_rows: List = []
    state = StateCheckReport()
    sound = SoundnessReport()
    for k, out, _proc in children:
        with out.open('rb') as fh:
            p = pickle.load(fh)
        wins += p['wins']
        losses += p['losses']
        errors += p['errors']
        skips += p['skips']
        pair_a += p['pair_a']
        pair_b += p['pair_b']
        pair_draw += p['pair_draw']
        pair_invalid += p['pair_invalid']
        fail_kinds.update(p['fail_kinds'])
        pair_outcomes.extend(p['pair_outcomes'])
        details.extend(p['details'])
        all_reward_rows.extend(p['reward_rows'])
        _merge_state_reports(state, p['state'])
        _merge_sound_reports(sound, p['sound'])
        if p.get('joint_diag'):
            from showdown.battle_bots.greedy_joint.diag import merge as merge_joint_diag
            merge_joint_diag(p['joint_diag'])
        if p.get('divergence') is not None:
            from showdown.battle_bots.expectimax_bayesian.divergence import (
                merge_divergence_stats,
            )
            merge_divergence_stats(p['divergence'])
        try:
            out.unlink()
        except OSError:
            pass
    try:
        tmpdir.rmdir()
    except OSError:
        pass

    pool = active_teams()
    used_metas: List[TeamMeta] = []
    for a, b in disjoint_ab_pairings(n_teams, seed=seed):
        used_metas.extend([a, b])
    flag_audit = audit_inference_flags(pool_abilities=collect_pool_abilities(used_metas))
    return _finalize_batch(
        n_teams=n_teams, n_bo3=n_matchups_total * 2, n_matchups=n_matchups_total,
        format_id=format_id, pool_n=len(pool), wins=wins, losses=losses,
        errors=errors, skips=skips, pair_a_sweeps=pair_a, pair_b_sweeps=pair_b,
        pair_draws=pair_draw, pair_invalid=pair_invalid, fail_kinds=fail_kinds,
        pair_outcomes=pair_outcomes, elapsed=time.perf_counter() - t0, seed=seed,
        state=state, sound=sound, details=details, include_details=include_details,
        flag_audit=flag_audit, player_a=player_a, player_b=player_b,
        all_reward_rows=all_reward_rows, skip_sync=skip_sync,
        workers=procs * wpp, ports=all_ports[:procs * wpp],
    )


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
        default=None,
        help=(
            'Timeout por serie BO3 (s). Default: 480s greedy validated; con '
            'expectimax escala con ExpectimaxConfig.deadline_ms (baseline '
            '550ms→420s; cobertura 75%% ~1600ms→~1220s).'
        ),
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
    parser.add_argument(
        '--skip-sync',
        action='store_true',
        help=(
            'Tune rápido: desactiva sync Battle + OTS (sin espejo IAP). '
            'Solo para comparar agentes; no usar para NEED_FIX de sync/inf.'
        ),
    )
    parser.add_argument(
        '--procs',
        type=int,
        default=1,
        help=(
            'Procesos en paralelo (cada uno con --workers corrutinas y su bloque '
            'de puertos). Servidor: ~24; 0 = automático según núcleos.'
        ),
    )
    parser.add_argument(
        '--battle-seed', type=int, default=None,
        help=(
            'Semilla base de las batallas de Showdown (default: --seed). Hace que dos '
            'corridas con la misma semilla jueguen las mismas batallas (azar de críticos, '
            'rolls y empates). Solo aplica a servidores que arranca esta corrida.'
        ),
    )
    parser.add_argument(
        '--no-battle-seed', action='store_true',
        help='No fijar la semilla de batalla (azar de Showdown distinto en cada corrida)',
    )
    parser.add_argument('--shard', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--shard-out', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.export_only:
        out = export_teams()
        print(f'Exported teams -> {out}')
        return 0

    quiet = not args.verbose
    ports = parse_ps_ports(workers=max(1, args.workers))
    uses_expectimax = (
        'expectimax' in str(args.player_a).lower()
        or 'expectimax' in str(args.player_b).lower()
    )
    timeout_s = float(args.timeout) if args.timeout is not None else (
        default_series_timeout_s(uses_expectimax=uses_expectimax)
    )
    if not quiet:
        print(
            f'PS ports={ports} up={[is_ps_up(port=p) for p in ports]} | '
            f'format={args.format} | '
            f'teams={args.teams} matchups={args.teams // 2} '
            f'bo3={args.teams} (crossed) '
            f'A={args.player_a} B={args.player_b} workers={args.workers} '
            f'timeout_s={timeout_s}'
        )

    # Windows: Proactor cierra el pipe mientras aún escribe el websocket →
    # AssertionError cosmético en _loop_writing. Selector evita esa race.
    if sys.platform == 'win32':
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    from showdown.inference_sim.ps_server import set_battle_seed
    from showdown.rl.parallel import default_num_envs, raise_nofile_limit

    raise_nofile_limit()
    if args.no_battle_seed:
        os.environ.pop('PS_BATTLE_SEED', None)
    elif args.battle_seed is not None:
        set_battle_seed(args.battle_seed)
    elif args.seed is not None and not os.environ.get('PS_BATTLE_SEED'):
        set_battle_seed(args.seed)
    procs = int(args.procs)
    if procs == 0:
        procs = default_num_envs()
    if os.environ.get('PS_BATTLE_SEED') and (procs <= 1 or args.shard):
        from showdown.inference_sim.ps_server import ensure_fixed_hash_seed
        ensure_fixed_hash_seed()
    if procs > 1 and not args.shard:
        run_batch_multiprocess(
            args.teams,
            procs=procs,
            workers_per_proc=max(1, args.workers),
            format_id=args.format,
            start_ps=args.start_ps,
            restart_ps=args.restart_ps,
            seed=args.seed,
            timeout_s=timeout_s,
            include_details=args.details,
            export=args.export,
            quiet=quiet,
            player_a=args.player_a,
            player_b=args.player_b,
            skip_sync=bool(args.skip_sync),
            delay_s=args.delay,
            progress_every=args.progress_every,
        )
        return _exit_code_from_report()

    shard = None
    if args.shard:
        k, _, n_sh = str(args.shard).partition('/')
        shard = (int(k), int(n_sh))

    asyncio.run(
        run_batch(
            args.teams,
            format_id=args.format,
            start_ps=args.start_ps,
            restart_ps=args.restart_ps,
            seed=args.seed,
            delay_s=args.delay,
            timeout_s=timeout_s,
            progress_every=args.progress_every,
            include_details=args.details,
            export=args.export,
            quiet=quiet,
            player_a=args.player_a,
            player_b=args.player_b,
            workers=args.workers,
            skip_sync=bool(args.skip_sync),
            shard=shard,
            shard_out=args.shard_out,
        )
    )
    if args.shard_out:
        return 0
    return _exit_code_from_report()


def _exit_code_from_report() -> int:
    text = REPORT_PATH.read_text(encoding='utf-8') if REPORT_PATH.exists() else ''
    skip_sync_run = 'skip_sync: YES' in text
    sync_bad = (
        not skip_sync_run
        and (
            'sync_ok: NO' in text
            or 'state_mismatch: YES' in text
        )
    )
    inf_bad = (
        not skip_sync_run
        and (
            'inf_ok: NO' in text
            or 'inference_ok: NO' in text
            or 'soundness_ok: NO' in text
        )
    )
    if sync_bad or inf_bad:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
