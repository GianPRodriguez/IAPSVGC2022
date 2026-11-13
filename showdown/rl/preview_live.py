"""BO1 live para Team Preview: N batallas reales en paralelo con π_c congelado.

Las specs de preview (PPO, Nash, QD-PGA) exigen en producción que cada etiqueta
salga de un BO1 real jugado por ``greedy_joint`` congelado (máximo 40 turnos),
no del proxy de cobertura del backend ``mock``. Este módulo lo implementa:

- ``LivePreviewPool``: N procesos ``spawn``; cada uno tiene dos pilotos
  (nosotros / rival) en su propio servidor Showdown y juega un BO1 por trabajo.
- Formación forzada: el piloto envía ``/team <orden>`` elegido por el agente
  de preview; si un lado no la trae, usa el preview propio de π_c.
- Cand. 1 desde nuestro lado: +1 victoria, −1 derrota, 0 empate. Superar el
  tope de turnos cuenta como empate (la spec limita el episodio a 40 turnos).
- Robustez: un worker caído o colgado se relanza y el BO1 se reintenta.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PILOTS = ('greedy_joint', 'expectimax_bayesian')

# Tiempo máximo por BO1 antes de darlo por colgado (se reintenta en otro intento
# con el worker reiniciado). Un BO1 con greedy_joint dura ~5 s (máx. 40 turnos
# < 1 min); un cuelgue ocasional bloqueaba el lote entero hasta los 600 s.
DEFAULT_BATTLE_TIMEOUT_S = {'greedy_joint': 180.0, 'expectimax_bayesian': 600.0}


def default_battle_timeout_s(pilot: str) -> float:
    return DEFAULT_BATTLE_TIMEOUT_S.get(str(pilot), 600.0)


@dataclass(frozen=True)
class PreviewMatch:
    """Un BO1: exports Showdown de ambos equipos y órdenes de preview (``'1234'``)."""

    our_export: str
    foe_export: str
    our_order: Optional[str] = None
    foe_order: Optional[str] = None


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

def _pilot_class(pilot: str):
    key = str(pilot or 'greedy_joint').strip().lower()
    if key == 'greedy_joint':
        from showdown.battle_bots.greedy_joint.pokeenv_player import GreedyJointPlayer as base
    elif key == 'expectimax_bayesian':
        from showdown.battle_bots.expectimax_bayesian.pokeenv_player import (
            ExpectimaxBayesianPlayer as base,
        )
    else:
        raise ValueError(f'piloto inválido {pilot!r}; opciones: {", ".join(PILOTS)}')

    class _PreviewPilot(base):  # type: ignore[misc, valid-type]
        """π_c congelado con formación de preview forzada y tope de turnos."""

        forced_order: Optional[str] = None
        max_turns: int = 40

        def teampreview(self, battle):
            order = self.forced_order
            if not order:
                return super().teampreview(battle)
            chosen = {int(ch) for ch in order}
            for i, mon in enumerate(battle.teampreview_team or [], start=1):
                try:
                    mon._selected_in_teampreview = i in chosen  # requisito poke-env
                except Exception:
                    pass
            self.last_preview_order = order
            return f'/team {order}'

        def choose_move(self, battle):
            if int(getattr(battle, 'turn', 0) or 0) > int(self.max_turns):
                from poke_env.player.battle_order import ForfeitBattleOrder

                capped = getattr(self, '_turn_capped', None)
                if capped is None:
                    capped = self._turn_capped = set()
                capped.add(battle.battle_tag)
                return ForfeitBattleOrder()
            return super().choose_move(battle)

    return _PreviewPilot


class _MatchRunner:
    """Dos pilotos en un servidor Showdown; juega BO1 bajo demanda."""

    def __init__(self, *, port: Optional[int], worker_id: int, pilot: str, max_turns: int,
                 battle_format: Optional[str], start_ps: bool):
        from poke_env.ps_client import AccountConfiguration

        from showdown.inference_sim.paths import BATTLE_FORMAT, PS_PORT, server_configuration_for_port
        from showdown.inference_sim.ps_server import ensure_ps

        self.port = int(port if port is not None else PS_PORT)
        if not ensure_ps(start=start_ps, port=self.port):
            raise RuntimeError(f'Showdown no responde en {self.port}')
        cfg = server_configuration_for_port(self.port)
        fmt = battle_format or BATTLE_FORMAT
        cls = _pilot_class(pilot)
        stamp = int(time.time()) % 100000
        common = dict(
            battle_format=fmt,
            server_configuration=cfg,
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
            log_level=40,
        )
        # Equipo inicial cualquiera: se reemplaza en cada trabajo.
        from showdown.inference_sim.team_factory import active_teams, to_showdown_export
        t0 = to_showdown_export(active_teams()[0])
        self.us = cls(account_configuration=AccountConfiguration(f'PvA{worker_id}x{stamp}'[:18], None),
                      team=t0, **common)
        self.foe = cls(account_configuration=AccountConfiguration(f'PvB{worker_id}x{stamp}'[:18], None),
                       team=t0, **common)
        for p in (self.us, self.foe):
            p.max_turns = int(max_turns)

    def play(self, match: PreviewMatch, timeout_s: float) -> Dict[str, Any]:
        import asyncio

        self.us.update_team(match.our_export)
        self.foe.update_team(match.foe_export)
        self.us.forced_order = match.our_order or None
        self.foe.forced_order = match.foe_order or None
        before = set(self.us.battles.keys())
        fut = asyncio.run_coroutine_threadsafe(
            self.us.battle_against(self.foe, n_battles=1), self.us.ps_client.loop,
        )
        try:
            fut.result(timeout=timeout_s)
        except Exception as exc:
            if type(exc).__name__ in ('TimeoutError', 'CancelledError'):
                fut.cancel()
                diag = self.diagnose(before, match, timeout_s)
                _log_hang(diag)
                raise RuntimeError('BO1_TIMEOUT ' + json.dumps(diag, default=str)) from exc
            raise
        new = [b for tag, b in self.us.battles.items() if tag not in before]
        if not new:
            raise RuntimeError('el BO1 terminó sin batalla registrada')
        battle = new[-1]
        tag = battle.battle_tag
        capped = tag in getattr(self.us, '_turn_capped', set()) or tag in getattr(
            self.foe, '_turn_capped', set())
        if capped:
            reward = 0.0
        elif battle.won is True:
            reward = 1.0
        elif battle.won is False:
            reward = -1.0
        else:
            reward = 0.0
        turns = int(getattr(battle, 'turn', 0) or 0)
        # No acumular batallas terminadas en memoria durante miles de episodios.
        try:
            self.us.reset_battles()
            self.foe.reset_battles()
        except Exception:
            pass
        return {'reward': reward, 'turns': turns, 'capped': bool(capped)}

    def diagnose(self, before, match: PreviewMatch, timeout_s: float) -> Dict[str, Any]:
        """Foto del estado de ambos lados de un BO1 colgado (para encontrar la causa)."""

        def side(player) -> Dict[str, Any]:
            out: Dict[str, Any] = {'username': getattr(player, 'username', None)}
            try:
                tags = [t for t in player.battles if t not in before] or list(player.battles)
                b = player.battles[tags[-1]]
            except Exception as exc:
                out['error'] = f'sin batalla: {exc!r}'
                return out

            def get(name, fn):
                try:
                    out[name] = fn()
                except Exception as exc:
                    out[name] = f'<{type(exc).__name__}>'

            get('room', lambda: b.battle_tag)
            get('turn', lambda: b.turn)
            get('finished', lambda: b.finished)
            get('won', lambda: b.won)
            get('force_switch', lambda: b.force_switch)
            get('wait', lambda: getattr(b, '_wait', None))
            get('teampreview', lambda: getattr(b, 'teampreview', None))
            get('trapped', lambda: b.trapped)
            get('avail_moves', lambda: [[m.id for m in ms] for ms in b.available_moves])
            get('avail_switches', lambda: [[m.species for m in sw] for sw in b.available_switches])
            get('active', lambda: [m.species if m else None for m in b.active_pokemon])
            get('hp', lambda: [m.current_hp_fraction if m else None for m in b.active_pokemon])
            get('request_keys', lambda: sorted((b.last_request or {}).keys()))
            get('side_request', lambda: bool((b.last_request or {}).get('wait')))
            get('order_queue', lambda: player.order_queue.qsize())
            get('battle_queue', lambda: player.battle_queue.qsize())
            get('choice_pending', lambda: len(getattr(player, '_waiting', []) or []))
            return out

        return {
            'timeout_s': timeout_s,
            'our_order': match.our_order,
            'foe_order': match.foe_order,
            'port': self.port,
            'us': side(self.us),
            'foe': side(self.foe),
            'seed_env': os.environ.get('PS_BATTLE_SEED'),
        }

    def close(self) -> None:
        import asyncio

        from showdown.inference_sim.ps_client_shutdown import stop_player_client

        for p in (self.us, self.foe):
            try:
                asyncio.run_coroutine_threadsafe(
                    stop_player_client(p), p.ps_client.loop,
                ).result(timeout=10)
            except Exception:
                pass


def _log_hang(diag: Dict[str, Any]) -> None:
    """Añade el diagnóstico de un BO1 colgado a ``PS_HANG_LOG`` (si está definido)."""
    path = os.environ.get('PS_HANG_LOG', '').strip()
    if not path:
        return
    try:
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(diag, default=str) + chr(10))
    except OSError:
        pass


def _worker_main(conn, port, worker_id, pilot, max_turns, battle_format, start_ps,
                 timeout_s) -> None:
    from showdown.rl.parallel import limit_worker_threads, raise_nofile_limit

    limit_worker_threads(1)
    raise_nofile_limit()
    runner = None
    try:
        runner = _MatchRunner(port=port, worker_id=worker_id, pilot=pilot,
                              max_turns=max_turns, battle_format=battle_format,
                              start_ps=start_ps)
        conn.send(('ready', None))
        while True:
            cmd, payload = conn.recv()
            if cmd == 'close':
                conn.send(('ok', None))
                break
            if cmd != 'play':
                conn.send(('error', f'comando desconocido {cmd!r}'))
                continue
            try:
                conn.send(('ok', runner.play(payload, timeout_s)))
            except Exception:
                err = traceback.format_exc()
                # Tras un cuelgue el desafío sigue abierto en el servidor: reiniciar.
                conn.send(('fatal' if 'BO1_TIMEOUT' in err else 'error', err))
                if 'BO1_TIMEOUT' in err:
                    break
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    except Exception:
        try:
            conn.send(('fatal', traceback.format_exc()))
        except Exception:
            pass
    finally:
        if runner is not None:
            runner.close()
        try:
            conn.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Pool
# --------------------------------------------------------------------------- #

class LivePreviewPool:
    """Pool de BO1 live. ``run(matches)`` devuelve la Cand. 1 de cada uno."""

    def __init__(
        self,
        num_workers: int,
        *,
        envs_per_server: int = 4,
        base_port: Optional[int] = None,
        start_ps: bool = False,
        pilot: str = 'greedy_joint',
        max_turns: int = 40,
        battle_format: Optional[str] = None,
        battle_timeout_s: Optional[float] = None,
        start_timeout_s: float = 240.0,
        max_retries: int = 2,
        max_restarts_per_worker: int = 30,
    ):
        from showdown.rl.parallel import plan_env_layout

        if pilot not in PILOTS:
            raise ValueError(f'piloto inválido {pilot!r}; opciones: {", ".join(PILOTS)}')
        self.num_workers = max(1, int(num_workers))
        self.pilot = pilot
        self.max_turns = int(max_turns)
        self.battle_format = battle_format
        self.battle_timeout_s = float(
            battle_timeout_s if battle_timeout_s is not None else default_battle_timeout_s(pilot)
        )
        self.start_timeout_s = float(start_timeout_s)
        self.max_retries = int(max_retries)
        self.max_restarts = int(max_restarts_per_worker)
        self.start_ps = bool(start_ps)
        layout = plan_env_layout(self.num_workers, envs_per_server=envs_per_server,
                                 base_port=base_port)
        self.ports = layout.env_ports
        from showdown.inference_sim.ps_server import ensure_ps_ports, ps_start_hint
        if not ensure_ps_ports(layout.ports, start=start_ps):
            hints = '; '.join(ps_start_hint(p) for p in layout.ports)
            raise RuntimeError(f'Showdown no responde en {layout.ports}. Arranca: {hints}')
        self._ctx = mp.get_context('spawn')
        self._conns: List[Any] = [None] * self.num_workers
        self._procs: List[Any] = [None] * self.num_workers
        self.restarts = [0] * self.num_workers
        self.stats = {'battles': 0, 'wins': 0, 'losses': 0, 'ties': 0, 'capped': 0,
                      'failures': 0, 'retries': 0, 'turns_sum': 0, 'seconds': 0.0}
        self._closed = False
        try:
            for i in range(self.num_workers):
                self._launch(i)
            for i in range(self.num_workers):
                self._await_ready(i)
        except Exception:
            self.close()
            raise

    # --- workers ------------------------------------------------------------

    def _launch(self, i: int) -> None:
        parent, child = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(
            target=_worker_main,
            args=(child, self.ports[i], i + 1000 * self.restarts[i], self.pilot,
                  self.max_turns, self.battle_format, self.start_ps, self.battle_timeout_s),
            daemon=True,
            name=f'preview-bo1-{i}',
        )
        proc.start()
        child.close()
        self._conns[i] = parent
        self._procs[i] = proc

    def _await_ready(self, i: int) -> None:
        conn = self._conns[i]
        if not conn.poll(self.start_timeout_s):
            raise RuntimeError(f'worker preview {i} no arrancó en {self.start_timeout_s:.0f}s')
        status, payload = conn.recv()
        if status != 'ready':
            raise RuntimeError(f'worker preview {i} no arrancó: {status} {payload}')

    def _kill(self, i: int) -> None:
        conn, proc = self._conns[i], self._procs[i]
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        self._conns[i] = None
        self._procs[i] = None

    def _restart(self, i: int) -> None:
        self.restarts[i] += 1
        if self.restarts[i] > self.max_restarts:
            raise RuntimeError(f'worker preview {i} superó {self.max_restarts} reinicios')
        self._kill(i)
        self._launch(i)
        self._await_ready(i)

    # --- API ----------------------------------------------------------------

    def run(self, matches: Sequence[PreviewMatch]) -> np.ndarray:
        """Juega todos los BO1 en paralelo; devuelve Cand. 1 en el mismo orden."""
        from multiprocessing.connection import wait as _wait

        n = len(matches)
        rewards = np.zeros(n, dtype=np.float32)
        if n == 0:
            return rewards
        t0 = time.monotonic()
        pending = list(range(n))
        attempts = [0] * n
        busy: Dict[int, Tuple[int, float]] = {}  # worker -> (job, t_inicio)
        idle = list(range(self.num_workers))
        while pending or busy:
            while pending and idle:
                w = idle.pop()
                j = pending.pop(0)
                try:
                    self._conns[w].send(('play', matches[j]))
                    busy[w] = (j, time.monotonic())
                except (BrokenPipeError, OSError, AttributeError):
                    pending.insert(0, j)
                    self._restart(w)
                    idle.append(w)
            if not busy:
                continue
            conns = {self._conns[w]: w for w in busy}
            ready = _wait(list(conns), timeout=5.0)
            now = time.monotonic()
            done_workers = []
            for conn in ready:
                w = conns[conn]
                j, _t = busy[w]
                try:
                    status, payload = conn.recv()
                except (EOFError, OSError):
                    status, payload = 'fatal', 'conexión perdida'
                if status == 'ok':
                    self._record(payload)
                    rewards[j] = float(payload['reward'])
                    done_workers.append((w, False))
                else:
                    self._retry_or_fail(j, attempts, pending, rewards, str(payload))
                    done_workers.append((w, status == 'fatal'))
            for w, (j, t_start) in list(busy.items()):
                if any(w == dw for dw, _ in done_workers):
                    continue
                if now - t_start > self.battle_timeout_s + 30.0:
                    self._retry_or_fail(j, attempts, pending, rewards,
                                        f'worker {w} sin respuesta')
                    done_workers.append((w, True))
            for w, needs_restart in done_workers:
                busy.pop(w, None)
                if needs_restart:
                    self._restart(w)
                idle.append(w)
        self.stats['seconds'] += time.monotonic() - t0
        return rewards

    def _retry_or_fail(self, j, attempts, pending, rewards, err) -> None:
        attempts[j] += 1
        last = err.strip().splitlines()[-1] if err.strip() else err
        if attempts[j] <= self.max_retries:
            self.stats['retries'] += 1
            print(f'[preview_live] BO1 #{j} intento {attempts[j]} falló ({last[:120]}); reintento',
                  flush=True)
            pending.append(j)
        else:
            self.stats['failures'] += 1
            rewards[j] = 0.0
            print(f'[preview_live] BO1 #{j} falló tras {attempts[j]} intentos: '
                  f'{err.strip().splitlines()[-1] if err.strip() else err}', flush=True)

    def _record(self, payload: Dict[str, Any]) -> None:
        s = self.stats
        s['battles'] += 1
        s['turns_sum'] += int(payload.get('turns', 0))
        if payload.get('capped'):
            s['capped'] += 1
        r = float(payload.get('reward', 0.0))
        if r > 0:
            s['wins'] += 1
        elif r < 0:
            s['losses'] += 1
        else:
            s['ties'] += 1

    def summary(self) -> Dict[str, Any]:
        s = dict(self.stats)
        b = max(1, s['battles'])
        s['mean_turns'] = s['turns_sum'] / b
        s['battles_per_s'] = s['battles'] / s['seconds'] if s['seconds'] > 0 else 0.0
        s['workers'] = self.num_workers
        s['pilot'] = self.pilot
        s['worker_restarts'] = int(sum(self.restarts))
        return s

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for i in range(self.num_workers):
            conn = self._conns[i]
            if conn is None:
                continue
            try:
                conn.send(('close', None))
                if conn.poll(15):
                    conn.recv()
            except Exception:
                pass
            self._kill(i)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
# Ayudantes para los entrenadores de preview (PPO, Nash, QD-PGA)
# --------------------------------------------------------------------------- #

def add_live_cli_args(parser) -> None:
    """Opciones comunes: backend mock|live y recursos del pool de BO1."""
    parser.add_argument('--backend', choices=('mock', 'live'), default='mock',
                        help='mock = proxy de cobertura; live = BO1 reales con π_c congelado')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='BO1 en paralelo (live). 0 = automático según núcleos')
    parser.add_argument('--envs-per-server', type=int, default=4,
                        help='Pilotos por servidor Showdown (live)')
    parser.add_argument('--base-port', type=int, default=None,
                        help='Primer puerto Showdown (default PS_PORT)')
    parser.add_argument('--start-ps', action='store_true', help='Arrancar Showdown si falta')
    parser.add_argument('--pilot', choices=PILOTS, default='greedy_joint',
                        help='π_c congelado (spec: greedy_joint; ablación expectimax_bayesian)')
    parser.add_argument('--max-turns', type=int, default=40,
                        help='Tope de turnos del BO1 (superarlo cuenta como empate)')


def live_options_from_args(args) -> Dict[str, Any]:
    return {
        'backend': getattr(args, 'backend', 'mock'),
        'num_workers': int(getattr(args, 'num_workers', 0) or 0),
        'envs_per_server': int(getattr(args, 'envs_per_server', 4) or 4),
        'base_port': getattr(args, 'base_port', None),
        'start_ps': bool(getattr(args, 'start_ps', False)),
        'pilot': getattr(args, 'pilot', 'greedy_joint'),
        'max_turns': int(getattr(args, 'max_turns', 40) or 40),
    }


def make_preview_env(seed: int, live: Optional[Dict[str, Any]] = None):
    """Devuelve ``(env, pool)``; ``pool`` es None con backend mock. Cerrar con ``pool.close()``."""
    from showdown.rl.parallel import default_num_envs, raise_nofile_limit
    from showdown.rl.team_preview_env import TeamPreviewEnv

    opts = dict(live or {})
    if opts.get('backend', 'mock') != 'live':
        return TeamPreviewEnv(seed=seed, backend='mock'), None
    raise_nofile_limit()
    n = int(opts.get('num_workers') or 0) or default_num_envs()
    pool = LivePreviewPool(
        n,
        envs_per_server=int(opts.get('envs_per_server', 4)),
        base_port=opts.get('base_port'),
        start_ps=bool(opts.get('start_ps', False)),
        pilot=str(opts.get('pilot', 'greedy_joint')),
        max_turns=int(opts.get('max_turns', 40)),
    )
    try:
        env = TeamPreviewEnv(seed=seed, backend='live', live_pool=pool)
    except Exception:
        pool.close()
        raise
    print(f'[preview] backend=live pilot={pool.pilot} workers={pool.num_workers} '
          f'max_turns={pool.max_turns}', flush=True)
    return env, pool


def provenance_meta(env, pool) -> Dict[str, Any]:
    """Metadatos de procedencia para train_meta.json / reporte.txt (exigidos por la spec)."""
    meta: Dict[str, Any] = {'provenance': getattr(env, 'provenance', 'mock')}
    if pool is not None:
        meta['live'] = pool.summary()
    return meta


def provenance_lines(meta: Dict[str, Any]) -> str:
    lines = [f"provenance: {meta.get('provenance', 'mock')}"]
    live = meta.get('live')
    if live:
        lines.append(
            f"live: pilot={live['pilot']} workers={live['workers']} battles={live['battles']} "
            f"W/L/T={live['wins']}/{live['losses']}/{live['ties']} capped={live['capped']} "
            f"failures={live['failures']} retries={live['retries']} "
            f"mean_turns={live['mean_turns']:.1f} battles_per_s={live['battles_per_s']:.2f}"
        )
    return '\n'.join(lines) + '\n'
