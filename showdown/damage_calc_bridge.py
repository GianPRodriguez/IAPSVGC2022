"""
Bridge al damage-calc Champions (gen 0), afinado para muchos agentes.

Prioridad de transporte:
  1) Servidor TCP compartido (DAMAGE_CALC_HOST:PORT, default 127.0.0.1:8765)
     -> un solo Node para 50+ agentes Python.
  2) Worker stdio local (fallback si no hay server).

Ademas:
  - Batch de requests en un solo round-trip.
  - LRU en memoria + caché SQLite compartida entre procesos.
  - Mock inyectable para tests.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _node_bin() -> str:
    from showdown.inference_sim.paths import node_executable
    return node_executable()

_DEFAULT_CALC_ROOT = Path(__file__).resolve().parents[2] / 'damage-calc'
_TOOLS = Path(__file__).resolve().parent / 'tools'
_WORKER_JS = _TOOLS / 'calc_worker.js'
_SERVER_JS = _TOOLS / 'calc_server.js'
_LOCK_FILE = Path(os.environ.get('DAMAGE_CALC_LOCK', str(_TOOLS / '.calc_server.lock')))
_DISK_CACHE = Path(os.environ.get(
    'DAMAGE_CALC_CACHE',
    str(Path(__file__).resolve().parents[1] / 'logs' / 'calc_cache.sqlite'),
))

_CACHE_MAX = int(os.environ.get('DAMAGE_CALC_LRU', '8192'))
_DEFAULT_HOST = os.environ.get('DAMAGE_CALC_HOST', '127.0.0.1')
_DEFAULT_PORT = int(os.environ.get('DAMAGE_CALC_PORT', '8765'))
# Misses hacia Node se parten en trozos (menos JSON gigante / timeouts 30s).
_BATCH_CHUNK = max(1, int(os.environ.get('DAMAGE_CALC_BATCH_CHUNK', '32')))


# Mock: callable(request) -> (min,max)|None  OR  callable(list[request]) -> list
_damage_range_override = None


def set_damage_range_override(fn) -> None:
    global _damage_range_override
    _damage_range_override = fn


def clear_damage_cache() -> None:
    DamageCalcBridge.instance().clear_cache()


def shutdown_damage_calc() -> None:
    DamageCalcBridge.instance().shutdown()


def damage_range(request: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    results = damage_ranges_batch([request])
    if not results:
        return None
    return results[0]


def damage_ranges_batch(
    requests: Sequence[Dict[str, Any]],
    *,
    chunk_size: Optional[int] = None,
    deadline_monotonic: Optional[float] = None,
) -> List[Optional[Tuple[int, int]]]:
    """
    Calcula rangos de daño. Los misses a Node se envían en chunks.

    deadline_monotonic: si se pasa (time.perf_counter()), entre chunks se para
    y los requests pendientes quedan None.
    """
    if not requests:
        return []
    if _damage_range_override is not None:
        # Override batch-aware o de un solo request (tests).
        try:
            batched = _damage_range_override(list(requests))
            if isinstance(batched, list) and len(batched) == len(requests):
                return list(batched)
        except Exception:
            pass
        return [_damage_range_override(req) for req in requests]
    return DamageCalcBridge.instance().damage_ranges_batch(
        list(requests),
        chunk_size=chunk_size,
        deadline_monotonic=deadline_monotonic,
    )


class _DiskCache:
    """SQLite compartido (WAL) entre procesos Python."""

    def __init__(self, path: Path):
        self.path = path
        self._local = threading.local()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = self._conn()
            conn.execute(
                'CREATE TABLE IF NOT EXISTS rolls ('
                'k TEXT PRIMARY KEY, min_r INTEGER, max_r INTEGER)'
            )
            conn.commit()
        except Exception as error:
            print(f'[STAT-INF] disk cache deshabilitada: {error}')
            self.path = None

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=2.0, check_same_thread=False)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            self._local.conn = conn
        return conn

    def get(self, key: str) -> Optional[Tuple[int, int]]:
        if self.path is None:
            return None
        try:
            row = self._conn().execute(
                'SELECT min_r, max_r FROM rolls WHERE k=?', (key,)
            ).fetchone()
            if row:
                return int(row[0]), int(row[1])
        except Exception:
            return None
        return None

    def put_many(self, items: Sequence[Tuple[str, Tuple[int, int]]]) -> None:
        if self.path is None or not items:
            return
        try:
            conn = self._conn()
            conn.executemany(
                'INSERT OR REPLACE INTO rolls(k, min_r, max_r) VALUES (?,?,?)',
                [(k, v[0], v[1]) for k, v in items],
            )
            conn.commit()
        except Exception:
            pass


class DamageCalcBridge:
    _singleton = None
    _singleton_lock = threading.Lock()

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._server_proc: Optional[subprocess.Popen] = None
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, Tuple[int, int]] = OrderedDict()
        self._failed = False
        self._mode: Optional[str] = None  # 'tcp' | 'stdio'
        self._disk = _DiskCache(_DISK_CACHE)

    @classmethod
    def instance(cls) -> 'DamageCalcBridge':
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def shutdown(self) -> None:
        with self._lock:
            self._close_sock()
            self._stop_stdio()
            # No matamos el server TCP compartido: otros agentes lo usan.
            self._cache.clear()
            self._failed = False
            self._mode = None

    def _close_sock(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _stop_stdio(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    def _calc_root(self) -> Path:
        env = os.environ.get('DAMAGE_CALC_ROOT')
        return Path(env) if env else _DEFAULT_CALC_ROOT

    def _cache_key(self, request: Dict[str, Any]) -> str:
        return json.dumps(request, sort_keys=True, separators=(',', ':'))

    def _remember(self, key: str, value: Tuple[int, int]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)

    def _try_tcp_connect(self) -> bool:
        host = _DEFAULT_HOST
        port = _DEFAULT_PORT
        try:
            sock = socket.create_connection((host, port), timeout=0.4)
            sock.settimeout(30.0)
            # ping
            sock.sendall(b'{"cmd":"ping"}\n')
            data = self._recv_line(sock)
            if not data or not data.get('ok'):
                sock.close()
                return False
            self._sock = sock
            self._mode = 'tcp'
            return True
        except Exception:
            self._close_sock()
            return False

    def _recv_line(self, sock: socket.socket) -> Optional[Dict[str, Any]]:
        buf = b''
        while b'\n' not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                return None
            buf += chunk
        line, _sep, _rest = buf.partition(b'\n')
        try:
            return json.loads(line.decode('utf-8'))
        except Exception:
            return None

    def _ensure_shared_server(self) -> bool:
        """Arranca calc_server.js una sola vez (lock file) si el puerto esta libre."""
        if self._try_tcp_connect():
            return True

        calc_root = self._calc_root()
        dist = calc_root / 'calc'
        if not _SERVER_JS.is_file():
            return False

        # Intento de ser el proceso que levanta el server
        try:
            _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
            if not _LOCK_FILE.exists():
                _LOCK_FILE.write_text(str(os.getpid()), encoding='utf-8')
                env = os.environ.copy()
                env['DAMAGE_CALC_DIST'] = str(dist)
                env['DAMAGE_CALC_HOST'] = _DEFAULT_HOST
                env['DAMAGE_CALC_PORT'] = str(_DEFAULT_PORT)
                self._server_proc = subprocess.Popen(
                    [_node_bin(), str(_SERVER_JS)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    cwd=str(calc_root),
                )
        except Exception:
            pass

        for _ in range(20):
            time.sleep(0.05)
            if self._try_tcp_connect():
                return True
        return False

    def _ensure_stdio_worker(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            self._mode = 'stdio'
            return True
        calc_root = self._calc_root()
        dist = calc_root / 'calc'
        if not _WORKER_JS.is_file():
            print(f'[STAT-INF] calc_worker.js no encontrado: {_WORKER_JS}')
            return False
        env = os.environ.copy()
        env['DAMAGE_CALC_DIST'] = str(dist)
        try:
            self._proc = subprocess.Popen(
                [_node_bin(), str(_WORKER_JS)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(calc_root),
            )
            self._proc.stdin.write(json.dumps({'cmd': 'ping'}) + '\n')
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            data = json.loads(line) if line else {}
            if not data.get('ok'):
                self._stop_stdio()
                return False
            self._mode = 'stdio'
            return True
        except Exception as error:
            print(f'[STAT-INF] No se pudo arrancar calc worker: {error}')
            self._stop_stdio()
            return False

    def _ensure_transport(self) -> bool:
        if self._failed:
            return False
        if self._mode == 'tcp' and self._sock is not None:
            return True
        if self._mode == 'stdio' and self._proc is not None and self._proc.poll() is None:
            return True

        # Preferir TCP compartido (escala a 50+ agentes)
        force_local = os.environ.get('DAMAGE_CALC_FORCE_LOCAL', '').strip() in ('1', 'true', 'yes')
        if not force_local:
            if self._ensure_shared_server():
                return True
        if self._ensure_stdio_worker():
            return True
        self._failed = True
        return False

    def _send_recv(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw = json.dumps(payload, separators=(',', ':')) + '\n'
        if self._mode == 'tcp' and self._sock is not None:
            try:
                self._sock.sendall(raw.encode('utf-8'))
                return self._recv_line(self._sock)
            except Exception as error:
                print(f'[STAT-INF] TCP calc error: {error}')
                self._close_sock()
                self._mode = None
                return None
        if self._mode == 'stdio' and self._proc is not None:
            try:
                self._proc.stdin.write(raw)
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
                if not line:
                    self._stop_stdio()
                    self._mode = None
                    return None
                return json.loads(line)
            except Exception as error:
                print(f'[STAT-INF] stdio calc error: {error}')
                self._stop_stdio()
                self._mode = None
                return None
        return None

    def _fill_miss_chunk(
        self,
        *,
        out: List[Optional[Tuple[int, int]]],
        miss_idx: List[int],
        miss_reqs: List[Dict[str, Any]],
        miss_keys: List[str],
    ) -> None:
        """Resuelve un trozo de misses (un round-trip batch o fallback 1-a-1)."""
        if not miss_reqs:
            return
        data = self._send_recv({'cmd': 'batch', 'items': miss_reqs})
        if not data or not data.get('ok') or not isinstance(data.get('results'), list):
            for j, req in enumerate(miss_reqs):
                one = self._send_recv(req)
                if one and one.get('ok'):
                    try:
                        val = (int(one['min']), int(one['max']))
                    except (KeyError, TypeError, ValueError):
                        continue
                    out[miss_idx[j]] = val
                    self._remember(miss_keys[j], val)
            return

        disk_writes = []
        for j, res in enumerate(data['results']):
            if j >= len(miss_idx):
                break
            if not res or not res.get('ok'):
                continue
            try:
                val = (int(res['min']), int(res['max']))
            except (KeyError, TypeError, ValueError):
                continue
            idx = miss_idx[j]
            out[idx] = val
            self._remember(miss_keys[j], val)
            disk_writes.append((miss_keys[j], val))
        self._disk.put_many(disk_writes)

    def damage_ranges_batch(
        self,
        requests: List[Dict[str, Any]],
        *,
        chunk_size: Optional[int] = None,
        deadline_monotonic: Optional[float] = None,
    ) -> List[Optional[Tuple[int, int]]]:
        n = len(requests)
        out: List[Optional[Tuple[int, int]]] = [None] * n
        if n == 0:
            return out

        keys = [self._cache_key(req) for req in requests]
        miss_idx: List[int] = []
        miss_reqs: List[Dict[str, Any]] = []
        miss_keys: List[str] = []

        with self._lock:
            for i, key in enumerate(keys):
                hit = self._cache.get(key)
                if hit is not None:
                    self._cache.move_to_end(key)
                    out[i] = hit
                    continue
                disk_hit = self._disk.get(key)
                if disk_hit is not None:
                    self._remember(key, disk_hit)
                    out[i] = disk_hit
                    continue
                miss_idx.append(i)
                miss_reqs.append(requests[i])
                miss_keys.append(key)

            if not miss_reqs:
                return out

            if not self._ensure_transport():
                return out

            cs = int(chunk_size) if chunk_size is not None else _BATCH_CHUNK
            cs = max(1, cs)
            total_miss = len(miss_reqs)
            for start in range(0, total_miss, cs):
                if (
                    deadline_monotonic is not None
                    and time.perf_counter() >= float(deadline_monotonic)
                ):
                    break
                end = min(start + cs, total_miss)
                self._fill_miss_chunk(
                    out=out,
                    miss_idx=miss_idx[start:end],
                    miss_reqs=miss_reqs[start:end],
                    miss_keys=miss_keys[start:end],
                )

        return out
