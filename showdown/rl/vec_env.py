"""Entornos VGC vectorizados: N batallas en paralelo, una por proceso worker.

El learner (DDQN / PPO) decide en lote para los N entornos y cada worker avanza
su batalla contra su servidor Showdown. Diseño:

- Cada worker construye su propio ``VGCEnv`` desde un ``EnvSpec`` picklable
  (``spawn``: igual en Linux y Windows, sin heredar hilos/event loops).
- Auto-reset: al terminar un episodio el worker reinicia solo y devuelve la
  observación final (``final_obs``) más la del episodio nuevo.
- Cambio de rival por episodio (Ω-expand): el learner adjunta en cada ``step``
  el rival que debe usarse *si* ese paso termina el episodio.
- Robustez: si un worker muere o no responde en ``step_timeout_s`` se mata y se
  relanza; el paso se reporta como ``trunc_reason='crash'`` (no entra al buffer).

``SyncVecVGCEnv`` ofrece la misma API en el mismo proceso (tests / 1 entorno).
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

OBS_KINDS = ('ppo', 'ddqn')
MASK_KINDS = ('slot', 'joint')


@dataclass(frozen=True)
class EnvSpec:
    """Receta picklable de un entorno (se reconstruye dentro del worker)."""

    opponent: str = 'mock'
    port: Optional[int] = None
    start_ps: bool = False
    account_prefix: str = 'RlVec'
    combat_family: str = 'ddqn'
    reward: str = 'candidate_2'  # RewardCandidate.DENSE.value
    max_episode_turns: int = 40
    obs_kind: str = 'ppo'
    mask_kind: str = 'slot'
    mock_max_turns: int = 40
    extra: Dict[str, Any] = field(default_factory=dict)

    def with_opponent(self, opponent: str) -> 'EnvSpec':
        return replace(self, opponent=str(opponent))


# --------------------------------------------------------------------------- #
# Construcción de entornos y helpers de observación / máscaras
# --------------------------------------------------------------------------- #

def make_env(spec: EnvSpec):
    from showdown.rl.env import MockBattleBackend, VGCEnv
    from showdown.rl.reward import RewardCandidate

    if spec.opponent == 'mock':
        backend = MockBattleBackend(max_turns=int(spec.mock_max_turns))
    else:
        from showdown.rl.pokeenv_backend import PokeEnvBattleBackend
        backend = PokeEnvBattleBackend(
            opponent=spec.opponent,
            start_ps=bool(spec.start_ps),
            account_prefix=spec.account_prefix,
            combat_family=spec.combat_family,
            port=spec.port,
        )
    return VGCEnv(
        reward_candidate=RewardCandidate(spec.reward),
        backend=backend,
        max_episode_turns=int(spec.max_episode_turns),
    )


def _observe(env, obs_kind: str) -> np.ndarray:
    from showdown.rl.observation import build_observation_ddqn, build_observation_ppo

    assert env._snap is not None
    if obs_kind == 'ddqn':
        return np.asarray(build_observation_ddqn(env._snap), dtype=np.float32)
    return np.asarray(build_observation_ppo(env._snap), dtype=np.float32)


def _mask(env, mask_kind: str) -> np.ndarray:
    if mask_kind == 'joint':
        from showdown.rl.ddqn.train import _joint_masks
        return np.asarray(_joint_masks(env), dtype=np.float32)
    from showdown.rl.ppo.train import _slot_masks
    return np.asarray(_slot_masks(env)[0], dtype=np.float32)


def _mask2(env, a1: int) -> np.ndarray:
    from showdown.rl.ppo.train import _m2_fn
    return np.asarray(_m2_fn(env)(int(a1)), dtype=np.float32)


def _set_opponent(env, spec: EnvSpec, opponent: str):
    """Cambia el rival; devuelve (env, spec) (puede reconstruir el entorno)."""
    from showdown.rl.env import MockBattleBackend

    opponent = str(opponent)
    if opponent == spec.opponent and env is not None:
        return env, spec
    new_spec = spec.with_opponent(opponent)
    backend = getattr(env, 'backend', None)
    if (
        opponent != 'mock'
        and backend is not None
        and not isinstance(backend, MockBattleBackend)
        and hasattr(backend, 'set_opponent')
    ):
        backend.set_opponent(opponent)
        return env, new_spec
    if env is not None:
        env.close()
    return make_env(new_spec), new_spec


def _lite_info(info: Dict[str, Any], env=None) -> Dict[str, Any]:
    eng = info.get('engine') or {}
    tel = info.get('telemetry') or {}
    eligible = bool(tel.get('archive_eligible'))
    bd = getattr(env, '_bd', None) if env is not None else None
    if not eligible and bd is not None:
        try:
            eligible = bool(bd.archive_eligible())
        except Exception:
            eligible = False
    bd_raw = info.get('bd_raw') or info.get('bd')
    if bd_raw is None and bd is not None:
        try:
            bd_raw = bd.bd_vector()
        except Exception:
            bd_raw = None
    return {
        'skip_buffer': bool(info.get('skip_buffer')),
        'trunc_reason': info.get('trunc_reason'),
        'used_fallback': bool(isinstance(eng, dict) and eng.get('used_fallback')),
        'episode_return': float(info.get('episode_return', 0.0) or 0.0),
        'turn': int(info.get('turn', 0) or 0),
        'error': info.get('error'),
        # Descriptor de comportamiento (QD): válido al terminar el episodio.
        'archive_eligible': eligible,
        'bd_raw': None if bd_raw is None else (float(bd_raw[0]), float(bd_raw[1])),
    }


class _EnvRunner:
    """Lógica de un entorno (compartida por el worker y la versión síncrona)."""

    def __init__(self, spec: EnvSpec, seed: int):
        self.spec = spec
        self.seed = int(seed)
        self.env = make_env(spec)
        self._episodes = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        s = self.seed + 7919 * self._episodes if seed is None else int(seed)
        self._episodes += 1
        self.env.reset(seed=s)
        return _observe(self.env, self.spec.obs_kind), _mask(self.env, self.spec.mask_kind)

    def step(self, action: Tuple[int, int], on_done_opponent: Optional[str]) -> Dict[str, Any]:
        _obs, reward, terminated, truncated, info = self.env.step(tuple(int(a) for a in action))
        obs = _observe(self.env, self.spec.obs_kind)
        lite = _lite_info(info, self.env)
        out: Dict[str, Any] = {
            'reward': float(reward),
            'terminated': bool(terminated),
            'truncated': bool(truncated),
            'info': lite,
            'opponent': self.spec.opponent,
        }
        done = bool(terminated or truncated)
        if not done:
            out['obs'] = obs
            out['mask'] = _mask(self.env, self.spec.mask_kind)
            return out
        out['final_obs'] = obs
        if on_done_opponent and on_done_opponent != self.spec.opponent:
            self.env, self.spec = _set_opponent(self.env, self.spec, on_done_opponent)
        out['obs'], out['mask'] = self.reset()
        out['next_opponent'] = self.spec.opponent
        return out

    def mask2(self, a1: int) -> np.ndarray:
        return _mask2(self.env, a1)

    def set_opponent(self, opponent: str) -> Tuple[np.ndarray, np.ndarray]:
        self.env, self.spec = _set_opponent(self.env, self.spec, opponent)
        return self.reset()

    def set_reward(self, candidate: str) -> None:
        """Cambia el candidato de recompensa (Cand. 1 evaluar / Cand. 2 mejorar)."""
        from showdown.rl.reward import RewardCandidate

        value = RewardCandidate(candidate).value
        self.env.reward_candidate = RewardCandidate(value)
        self.spec = replace(self.spec, reward=value)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

def _worker_main(conn, spec: EnvSpec, seed: int, threads: int) -> None:
    from showdown.rl.parallel import limit_worker_threads, raise_nofile_limit

    limit_worker_threads(threads)
    raise_nofile_limit()
    runner: Optional[_EnvRunner] = None
    try:
        runner = _EnvRunner(spec, seed)
        conn.send(('ready', None))
        while True:
            cmd, payload = conn.recv()
            try:
                if cmd == 'reset':
                    conn.send(('ok', runner.reset(payload)))
                elif cmd == 'step':
                    action, on_done = payload
                    conn.send(('ok', runner.step(action, on_done)))
                elif cmd == 'mask2':
                    conn.send(('ok', runner.mask2(payload)))
                elif cmd == 'set_opponent':
                    conn.send(('ok', runner.set_opponent(payload)))
                elif cmd == 'set_reward':
                    runner.set_reward(payload)
                    conn.send(('ok', None))
                elif cmd == 'close':
                    conn.send(('ok', None))
                    break
                else:
                    conn.send(('error', f'comando desconocido: {cmd!r}'))
            except Exception:
                conn.send(('error', traceback.format_exc()))
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


class WorkerError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# API vectorizada
# --------------------------------------------------------------------------- #

class _VecBase:
    num_envs: int
    specs: List[EnvSpec]

    def opponents(self) -> List[str]:
        return [s.opponent for s in self.specs]

    def _crash_result(self, i: int, err: str, obs: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        return {
            'reward': 0.0,
            'terminated': False,
            'truncated': True,
            'info': {
                'skip_buffer': True,
                'trunc_reason': 'crash',
                'used_fallback': False,
                'episode_return': 0.0,
                'turn': 0,
                'error': err,
            },
            'opponent': self.specs[i].opponent,
            'final_obs': obs,
            'obs': obs,
            'mask': mask,
            'next_opponent': self.specs[i].opponent,
            'worker_restarted': True,
        }


class SyncVecVGCEnv(_VecBase):
    """Misma API que ``SubprocVecVGCEnv`` pero en el proceso actual."""

    def __init__(self, specs: Sequence[EnvSpec], *, seed: int = 0):
        self.specs = list(specs)
        self.num_envs = len(self.specs)
        self._runners = [
            _EnvRunner(s, seed=int(seed) + 1000 * i) for i, s in enumerate(self.specs)
        ]

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        pairs = [r.reset() for r in self._runners]
        return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])

    def step(
        self,
        actions: Sequence[Tuple[int, int]],
        on_done_opponents: Optional[Sequence[Optional[str]]] = None,
    ) -> List[Dict[str, Any]]:
        on_done = list(on_done_opponents or [None] * self.num_envs)
        out = []
        for i, r in enumerate(self._runners):
            res = r.step(actions[i], on_done[i])
            self.specs[i] = r.spec
            out.append(res)
        return out

    def mask2(self, a1s: Sequence[int]) -> np.ndarray:
        return np.stack([r.mask2(int(a)) for r, a in zip(self._runners, a1s)])

    # --- API asíncrona (misma semántica que SubprocVecVGCEnv) ---------------

    def mask2_subset(self, indices: Sequence[int], a1s: Sequence[int]) -> np.ndarray:
        return np.stack([self._runners[i].mask2(int(a)) for i, a in zip(indices, a1s)])

    def step_async(
        self,
        indices: Sequence[int],
        actions: Sequence[Tuple[int, int]],
        on_done_opponents: Optional[Sequence[Optional[str]]] = None,
    ) -> None:
        on_done = list(on_done_opponents or [None] * len(indices))
        pend = getattr(self, '_pending', [])
        pend.extend(zip(indices, actions, on_done))
        self._pending = pend

    def in_flight(self) -> List[int]:
        return [i for i, _a, _o in getattr(self, '_pending', [])]

    def step_wait_any(
        self, *, timeout: Optional[float] = None, gather_s: float = 0.0,
    ) -> List[Tuple[int, Dict[str, Any]]]:
        pend, self._pending = getattr(self, '_pending', []), []
        out = []
        for i, action, on_done in pend:
            res = self._runners[i].step(action, on_done)
            self.specs[i] = self._runners[i].spec
            out.append((i, res))
        return out

    def drain(self) -> List[Tuple[int, Dict[str, Any]]]:
        return self.step_wait_any()

    def set_opponent(self, opponent: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.set_opponents([opponent] * self.num_envs)

    def set_opponents(self, opponents: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Un rival por entorno (p. ej. mosaico QD); reinicia todos los episodios."""
        pairs = []
        for i, (r, opp) in enumerate(zip(self._runners, opponents)):
            pairs.append(r.set_opponent(opp))
            self.specs[i] = r.spec
        return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])

    def set_reward(self, candidate: str) -> None:
        for i, r in enumerate(self._runners):
            r.set_reward(candidate)
            self.specs[i] = r.spec

    def close(self) -> None:
        for r in self._runners:
            r.close()


class SubprocVecVGCEnv(_VecBase):
    """N entornos en N procesos ``spawn`` comunicados por ``Pipe``.

    Los comandos se envían a todos los workers y luego se recogen: las N
    batallas avanzan en paralelo.
    """

    def __init__(
        self,
        specs: Sequence[EnvSpec],
        *,
        seed: int = 0,
        worker_threads: int = 1,
        start_timeout_s: float = 180.0,
        step_timeout_s: float = 600.0,
        max_restarts_per_worker: int = 20,
    ):
        self.specs = list(specs)
        self.num_envs = len(self.specs)
        self.seed = int(seed)
        self.worker_threads = int(worker_threads)
        self.start_timeout_s = float(start_timeout_s)
        self.step_timeout_s = float(step_timeout_s)
        self.max_restarts = int(max_restarts_per_worker)
        self.restarts = [0] * self.num_envs
        self._ctx = mp.get_context('spawn')
        self._conns: List[Any] = [None] * self.num_envs
        self._procs: List[Any] = [None] * self.num_envs
        self._last_obs: List[Optional[np.ndarray]] = [None] * self.num_envs
        self._last_mask: List[Optional[np.ndarray]] = [None] * self.num_envs
        self._closed = False
        try:
            for i in range(self.num_envs):
                self._launch(i)
            for i in range(self.num_envs):
                self._await_ready(i)
        except Exception:
            self.close()
            raise

    # --- ciclo de vida de workers -------------------------------------------

    def _worker_seed(self, i: int) -> int:
        return self.seed + 1000 * i + 100_000 * self.restarts[i]

    def _launch(self, i: int) -> None:
        parent, child = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(
            target=_worker_main,
            args=(child, self.specs[i], self._worker_seed(i), self.worker_threads),
            daemon=True,
            name=f'vgc-env-{i}',
        )
        proc.start()
        child.close()
        self._conns[i] = parent
        self._procs[i] = proc

    def _await_ready(self, i: int) -> None:
        status, payload = self._recv(i, self.start_timeout_s)
        if status != 'ready':
            raise WorkerError(f'worker {i} no arrancó: {status} {payload}')

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

    def _restart(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        self.restarts[i] += 1
        if self.restarts[i] > self.max_restarts:
            raise WorkerError(
                f'worker {i} superó {self.max_restarts} reinicios (puerto={self.specs[i].port})'
            )
        self._kill(i)
        self._launch(i)
        self._await_ready(i)
        obs, mask = self._call(i, 'reset', None)
        self._last_obs[i], self._last_mask[i] = obs, mask
        return obs, mask

    # --- comunicación --------------------------------------------------------

    def _recv(self, i: int, timeout: float):
        conn = self._conns[i]
        if conn is None:
            raise WorkerError(f'worker {i} sin conexión')
        try:
            if not conn.poll(timeout):
                raise WorkerError(f'worker {i} sin respuesta en {timeout:.0f}s')
            msg = conn.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise WorkerError(f'worker {i} cayó: {exc!r}') from exc
        if not isinstance(msg, tuple) or len(msg) != 2:
            raise WorkerError(f'worker {i} respuesta inválida: {msg!r}')
        return msg

    def _send(self, i: int, cmd: str, payload: Any) -> None:
        conn = self._conns[i]
        if conn is None:
            raise WorkerError(f'worker {i} sin conexión')
        try:
            conn.send((cmd, payload))
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise WorkerError(f'worker {i} cayó al enviar: {exc!r}') from exc

    def _unwrap(self, i: int, msg) -> Any:
        status, payload = msg
        if status == 'ok':
            return payload
        raise WorkerError(f'worker {i} {status}: {payload}')

    def _call(self, i: int, cmd: str, payload: Any) -> Any:
        self._send(i, cmd, payload)
        return self._unwrap(i, self._recv(i, self.step_timeout_s))

    def _broadcast(self, cmd: str, payloads: Sequence[Any]) -> List[Tuple[Optional[Any], Optional[str]]]:
        """Envía a todos y recoge (resultado, error) por worker."""
        sent = [False] * self.num_envs
        errors: List[Optional[str]] = [None] * self.num_envs
        for i in range(self.num_envs):
            try:
                self._send(i, cmd, payloads[i])
                sent[i] = True
            except WorkerError as exc:
                errors[i] = str(exc)
        results: List[Optional[Any]] = [None] * self.num_envs
        for i in range(self.num_envs):
            if not sent[i]:
                continue
            try:
                results[i] = self._unwrap(i, self._recv(i, self.step_timeout_s))
            except WorkerError as exc:
                errors[i] = str(exc)
        return list(zip(results, errors))

    # --- API -----------------------------------------------------------------

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        out = self._broadcast('reset', [None] * self.num_envs)
        for i, (res, err) in enumerate(out):
            if err is not None:
                res = self._restart(i)
            self._last_obs[i], self._last_mask[i] = res
        return np.stack(self._last_obs), np.stack(self._last_mask)

    def step(
        self,
        actions: Sequence[Tuple[int, int]],
        on_done_opponents: Optional[Sequence[Optional[str]]] = None,
    ) -> List[Dict[str, Any]]:
        on_done = list(on_done_opponents or [None] * self.num_envs)
        payloads = [((int(a[0]), int(a[1])), on_done[i]) for i, a in enumerate(actions)]
        out = self._broadcast('step', payloads)
        results: List[Dict[str, Any]] = []
        for i, (res, err) in enumerate(out):
            if err is not None:
                obs, mask = self._restart(i)
                results.append(self._crash_result(i, err, obs, mask))
                continue
            if res.get('next_opponent'):
                self.specs[i] = self.specs[i].with_opponent(res['next_opponent'])
            self._last_obs[i], self._last_mask[i] = res['obs'], res['mask']
            results.append(res)
        return results

    def mask2(self, a1s: Sequence[int]) -> np.ndarray:
        return self.mask2_subset(list(range(self.num_envs)), a1s)

    @staticmethod
    def _pass_only_mask() -> np.ndarray:
        from showdown.rl.constants import PASS_ID, SLOT_ACTION_N
        m = np.zeros(SLOT_ACTION_N, dtype=np.float32)
        m[PASS_ID] = 1.0
        return m

    def mask2_subset(self, indices: Sequence[int], a1s: Sequence[int]) -> np.ndarray:
        """Máscaras Slot2 de un subconjunto de workers (que no estén en vuelo)."""
        idx = [int(i) for i in indices]
        ok = {}
        for i, a in zip(idx, a1s):
            try:
                self._send(i, 'mask2', int(a))
                ok[i] = True
            except WorkerError:
                ok[i] = False
        masks = []
        for i in idx:
            m = None
            if ok[i]:
                try:
                    m = self._unwrap(i, self._recv(i, self.step_timeout_s))
                except WorkerError:
                    m = None
            # Sin máscara fiable: solo PASS (el paso fallará y se marcará crash).
            masks.append(self._pass_only_mask() if m is None else np.asarray(m, dtype=np.float32))
        return np.stack(masks)

    # --- API asíncrona -------------------------------------------------------
    #
    # step_async() envía el paso a algunos workers y step_wait_any() devuelve
    # los que ya respondieron. Así el learner no espera al entorno más lento
    # (p. ej. uno que está reiniciando batalla / team preview).

    def in_flight(self) -> List[int]:
        return sorted(getattr(self, '_inflight', {}).keys())

    def step_async(
        self,
        indices: Sequence[int],
        actions: Sequence[Tuple[int, int]],
        on_done_opponents: Optional[Sequence[Optional[str]]] = None,
    ) -> None:
        import time as _time

        inflight = getattr(self, '_inflight', None)
        if inflight is None:
            inflight = self._inflight = {}
        on_done = list(on_done_opponents or [None] * len(indices))
        failed = getattr(self, '_failed_sends', None)
        if failed is None:
            failed = self._failed_sends = {}
        for k, i in enumerate(indices):
            i = int(i)
            a = actions[k]
            try:
                self._send(i, 'step', ((int(a[0]), int(a[1])), on_done[k]))
                inflight[i] = _time.monotonic()
            except WorkerError as exc:
                failed[i] = str(exc)

    def step_wait_any(
        self,
        *,
        timeout: Optional[float] = None,
        gather_s: float = 0.002,
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """Espera hasta que al menos un worker responda y devuelve todos los listos.

        Workers que excedan ``step_timeout_s`` se reinician y devuelven un paso
        ``crash``. ``gather_s``: pequeña ventana para juntar más respuestas en
        el mismo lote (mejor uso de la GPU sin añadir latencia apreciable).
        """
        import time as _time
        from multiprocessing.connection import wait as _wait

        inflight: Dict[int, float] = getattr(self, '_inflight', {})
        failed: Dict[int, str] = getattr(self, '_failed_sends', {})
        out: List[Tuple[int, Dict[str, Any]]] = []

        for i, err in list(failed.items()):
            failed.pop(i, None)
            obs, mask = self._restart(i)
            out.append((i, self._crash_result(i, err, obs, mask)))
        if out or not inflight:
            return out

        def _collect(conns_ready) -> None:
            for conn in conns_ready:
                i = self._conns.index(conn)
                inflight.pop(i, None)
                try:
                    res = self._unwrap(i, self._recv(i, 1.0))
                except WorkerError as exc:
                    obs, mask = self._restart(i)
                    out.append((i, self._crash_result(i, str(exc), obs, mask)))
                    continue
                if res.get('next_opponent'):
                    self.specs[i] = self.specs[i].with_opponent(res['next_opponent'])
                self._last_obs[i], self._last_mask[i] = res['obs'], res['mask']
                out.append((i, res))

        deadline = None if timeout is None else _time.monotonic() + float(timeout)
        while not out and inflight:
            now = _time.monotonic()
            # Workers colgados: reinicio + crash.
            for i, t0 in list(inflight.items()):
                if now - t0 > self.step_timeout_s:
                    inflight.pop(i, None)
                    obs, mask = self._restart(i)
                    out.append((i, self._crash_result(
                        i, f'worker {i} sin respuesta en {self.step_timeout_s:.0f}s', obs, mask,
                    )))
            if out:
                break
            wait_for = 1.0
            if deadline is not None:
                wait_for = max(0.0, min(wait_for, deadline - now))
            conns = [self._conns[i] for i in inflight if self._conns[i] is not None]
            ready = _wait(conns, timeout=wait_for) if conns else []
            _collect(ready)
            if deadline is not None and _time.monotonic() >= deadline:
                break
        if out and inflight and gather_s > 0:
            conns = [self._conns[i] for i in inflight if self._conns[i] is not None]
            _collect(_wait(conns, timeout=gather_s) if conns else [])
        return out

    def drain(self) -> List[Tuple[int, Dict[str, Any]]]:
        """Espera a todos los workers en vuelo (antes de una actualización PPO)."""
        out: List[Tuple[int, Dict[str, Any]]] = []
        while getattr(self, '_inflight', {}) or getattr(self, '_failed_sends', {}):
            out.extend(self.step_wait_any(gather_s=0.0))
        return out

    def set_opponent(self, opponent: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.set_opponents([opponent] * self.num_envs)

    def set_opponents(self, opponents: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Un rival por entorno (p. ej. mosaico QD); reinicia todos los episodios."""
        opps = [str(o) for o in opponents]
        out = self._broadcast('set_opponent', opps)
        for i, (res, err) in enumerate(out):
            self.specs[i] = self.specs[i].with_opponent(opps[i])
            if err is not None:
                res = self._restart(i)
            self._last_obs[i], self._last_mask[i] = res
        return np.stack(self._last_obs), np.stack(self._last_mask)

    def set_reward(self, candidate: str) -> None:
        out = self._broadcast('set_reward', [str(candidate)] * self.num_envs)
        for i, (_res, err) in enumerate(out):
            self.specs[i] = replace(self.specs[i], reward=str(candidate))
            if err is not None:
                self._restart(i)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for i in range(self.num_envs):
            if self._conns[i] is None:
                continue
            try:
                self._send(i, 'close', None)
                self._recv(i, 15.0)
            except Exception:
                pass
            self._kill(i)


def make_vec_env(
    base_spec: EnvSpec,
    *,
    num_envs: int,
    seed: int = 0,
    envs_per_server: int = 4,
    base_port: Optional[int] = None,
    start_ps: bool = False,
    subprocess: Optional[bool] = None,
    step_timeout_s: float = 600.0,
):
    """Construye el entorno vectorizado con puertos/nombres únicos por worker.

    - Rival ``mock``: no usa Showdown.
    - Rival live: arranca (si ``start_ps``) los servidores necesarios *antes* de
      lanzar workers, para que no compitan arrancando el mismo puerto.
    """
    from showdown.rl.parallel import plan_env_layout

    n = max(1, int(num_envs))
    live = base_spec.opponent != 'mock'
    layout = plan_env_layout(n, envs_per_server=envs_per_server, base_port=base_port)
    if live:
        from showdown.inference_sim.ps_server import ensure_ps_ports, ps_start_hint
        if not ensure_ps_ports(layout.ports, start=start_ps):
            hints = '; '.join(ps_start_hint(p) for p in layout.ports)
            raise RuntimeError(f'Showdown no responde en {layout.ports}. Arranca: {hints}')
    specs = [
        replace(
            base_spec,
            port=layout.env_ports[i] if live else None,
            start_ps=bool(start_ps),
            account_prefix=f'{base_spec.account_prefix}{i}',
        )
        for i in range(n)
    ]
    use_subproc = (n > 1) if subprocess is None else bool(subprocess)
    if use_subproc:
        return SubprocVecVGCEnv(specs, seed=seed, step_timeout_s=step_timeout_s)
    return SyncVecVGCEnv(specs, seed=seed)
