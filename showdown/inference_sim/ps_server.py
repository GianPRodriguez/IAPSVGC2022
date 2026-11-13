"""Healthcheck / ensure del servidor Pokémon Showdown local (1+ puertos)."""
from __future__ import annotations

import atexit
import os
import re
import signal
import socket
import subprocess
import time
from typing import Dict, List, Optional, Set

from showdown.inference_sim.paths import POKEMON_SHOWDOWN_ROOT, PS_HOST, PS_PORT

_ps_procs: Dict[int, subprocess.Popen] = {}

# poke-env recomienda --no-security en desarrollo local (sin rate limits).
# Opt-out: PS_NO_SECURITY=0|false|no
_NO_SECURITY_DEFAULT = True


def ps_no_security_enabled() -> bool:
    raw = os.environ.get('PS_NO_SECURITY', '').strip().lower()
    if raw in ('0', 'false', 'no', 'off'):
        return False
    if raw in ('1', 'true', 'yes', 'on'):
        return True
    return _NO_SECURITY_DEFAULT


def ps_start_argv(port: Optional[int] = None) -> List[str]:
    """Comando canónico para arrancar PS Champions local.

    El puerto se pasa por env PSPORT al spawn (varios listeners en paralelo).
    El argv opcionalmente incluye el puerto numérico por compatibilidad CLI.
    """
    from showdown.inference_sim.paths import REPO_ROOT, node_executable

    if battle_seed_enabled():
        # Lanzador propio: fija la semilla de cada batalla (ver ps_seeded_start.js).
        # Exige dist/ ya compilado (no ejecuta `node build`).
        argv = [node_executable(), str(REPO_ROOT / 'showdown' / 'tools' / 'ps_seeded_start.js')]
        if ps_no_security_enabled():
            argv.append('--no-security')
        if port is not None:
            argv.append(str(int(port)))
        return argv
    argv = [node_executable(), 'pokemon-showdown', 'start']
    if ps_is_built():
        # Sin esto el lanzador ejecuta `node build` en cada arranque: con muchos
        # servidores a la vez son builds concurrentes sobre el mismo dist/.
        argv.append('--skip-build')
    if ps_no_security_enabled():
        argv.append('--no-security')
    if port is not None:
        argv.append(str(int(port)))
    return argv


def battle_seed_enabled() -> bool:
    """True si hay semilla base de batallas (``PS_BATTLE_SEED``) y el servidor está compilado."""
    return bool(os.environ.get('PS_BATTLE_SEED', '').strip()) and ps_is_built()


def set_battle_seed(seed: Optional[int], *, log_path: Optional[str] = None) -> None:
    """Fija la semilla base de las batallas de los servidores que lance este proceso.

    Debe llamarse ANTES de arrancar Showdown; un servidor que ya estaba encendido
    conserva su azar (``ensure_ps`` lo avisa). Los procesos hijos la heredan.
    """
    if seed is None:
        return
    os.environ['PS_BATTLE_SEED'] = str(int(seed))
    if log_path:
        os.environ['PS_SEED_LOG'] = str(log_path)
    # Python aleatoriza por proceso el orden de set/dict con cadenas, y los bots
    # desempatan por ese orden. Los procesos hijos (workers, shards) lo heredan.
    os.environ.setdefault('PYTHONHASHSEED', '0')


def ensure_fixed_hash_seed() -> None:
    """Re-ejecuta este proceso con ``PYTHONHASHSEED`` fijo si aún no lo tiene.

    Solo hace falta cuando el propio proceso juega las batallas (``--procs 1``);
    los procesos hijos ya reciben la variable al crearse.
    """
    import sys

    if os.environ.get('PYTHONHASHSEED') or not getattr(sys.flags, 'hash_randomization', 0):
        return
    os.environ['PYTHONHASHSEED'] = '0'
    orig = getattr(sys, 'orig_argv', None)
    if not orig:
        return
    os.execv(sys.executable, [sys.executable, *orig[1:]])


def seed_python_rngs(seed: int) -> None:
    """Siembra ``random`` y numpy (jugador random, desempates de los bots)."""
    import random

    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed) % (2 ** 32))
    except ImportError:
        pass


def ps_is_built() -> bool:
    return (POKEMON_SHOWDOWN_ROOT / 'dist' / 'server' / 'index.js').is_file()


def ensure_ps_build(timeout_s: float = 900.0) -> bool:
    """Compila pokemon-showdown una vez si falta ``dist/`` (antes de lanzar servidores)."""
    if ps_is_built():
        return True
    from showdown.inference_sim.paths import node_executable

    try:
        subprocess.run(
            [node_executable(), 'build'],
            cwd=str(POKEMON_SHOWDOWN_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
    except Exception:
        return False
    return ps_is_built()


def _start_parallelism() -> int:
    raw = os.environ.get('PS_START_PARALLEL', '').strip()
    try:
        return max(1, int(raw)) if raw else 8
    except ValueError:
        return 8


def ps_start_hint(port: Optional[int] = None) -> str:
    p = PS_PORT if port is None else int(port)
    return ' '.join(ps_start_argv(p))


def parse_ps_ports(*, workers: int = 2) -> List[int]:
    """Lista de puertos: PS_PORTS=8000,8001 o PS_PORT..PS_PORT+workers-1."""
    raw = os.environ.get('PS_PORTS', '').strip()
    if raw:
        ports: List[int] = []
        for part in raw.replace(';', ',').split(','):
            part = part.strip()
            if part:
                ports.append(int(part))
        if ports:
            return ports
    n = max(1, int(workers))
    base = int(PS_PORT)
    return [base + i for i in range(n)]


def is_ps_up(host: str = PS_HOST, port: int = PS_PORT, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _pids_listening_on_port(port: int = PS_PORT) -> Set[int]:
    """PIDs que escuchan el puerto (Windows: netstat; Linux/macOS: ss)."""
    if os.name != 'nt':
        return _pids_listening_posix(port)
    pids: Set[int] = set()
    try:
        out = subprocess.check_output(
            ['netstat', '-ano', '-p', 'tcp'],
            text=True,
            errors='replace',
            timeout=10,
        )
        needle = f':{int(port)}'
        for line in out.splitlines():
            if needle not in line or 'LISTENING' not in line.upper():
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                pids.add(int(parts[-1]))
            except ValueError:
                continue
    except Exception:
        pass
    return {p for p in pids if p > 0}


def _pids_listening_posix(port: int) -> Set[int]:
    pids: Set[int] = set()
    try:
        out = subprocess.check_output(
            ['ss', '-ltnpH', f'sport = :{int(port)}'],
            text=True, errors='replace', timeout=10,
        )
        for m in re.finditer(r'pid=(\d+)', out):
            pids.add(int(m.group(1)))
    except Exception:
        pass
    return {p for p in pids if p > 0}


def stop_ps_listeners(port: int = PS_PORT) -> int:
    """Mata procesos que escuchan el puerto PS. Retorna cuántos intentó matar."""
    stop_started_ps(port=port)
    killed = 0
    for pid in sorted(_pids_listening_on_port(port)):
        try:
            if os.name == 'nt':
                subprocess.run(
                    ['taskkill', '/PID', str(pid), '/T', '/F'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                os.kill(pid, 15)
            killed += 1
        except Exception:
            pass
    deadline = time.time() + 15
    while time.time() < deadline and is_ps_up(port=port):
        time.sleep(0.2)
    return killed


def _spawn_ps(port: int = PS_PORT) -> subprocess.Popen:
    if not POKEMON_SHOWDOWN_ROOT.is_dir():
        raise FileNotFoundError(f'No existe pokemon-showdown en {POKEMON_SHOWDOWN_ROOT}')
    cfg = POKEMON_SHOWDOWN_ROOT / 'config' / 'config.js'
    example = POKEMON_SHOWDOWN_ROOT / 'config' / 'config-example.js'
    if not cfg.exists() and example.exists():
        cfg.write_text(example.read_text(encoding='utf-8'), encoding='utf-8')
    # Resolver node antes de copiar el entorno: añade nvm al PATH si hace falta.
    from showdown.inference_sim.paths import node_executable
    node_executable()
    env = os.environ.copy()
    env['PSPORT'] = str(int(port))
    _register_cleanup()
    kwargs = {}
    if os.name == 'nt':
        kwargs['creationflags'] = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    else:
        # Grupo propio: al apagar se cierran también los subprocesos de PS
        # (sockets, room-battle, validator...), no solo el lanzador.
        kwargs['start_new_session'] = True
    return subprocess.Popen(
        ps_start_argv(port),
        cwd=str(POKEMON_SHOWDOWN_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        **kwargs,
    )


_cleanup_registered = False


def _register_cleanup() -> None:
    """Apaga al salir los servidores que arrancó este proceso.

    En un servidor compartido no deben quedar decenas de PS ociosos tras un
    entrenamiento. ``PS_KEEP_SERVERS=1`` desactiva el apagado (para reutilizar
    servidores entre corridas seguidas).
    """
    global _cleanup_registered
    if _cleanup_registered:
        return
    _cleanup_registered = True
    if os.environ.get('PS_KEEP_SERVERS', '').strip().lower() in ('1', 'true', 'yes'):
        return
    atexit.register(stop_started_ps)


def _kill_process_tree(proc: subprocess.Popen, timeout: float = 8.0) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == 'nt':
            subprocess.run(
                ['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=timeout)
    except Exception:
        try:
            if os.name != 'nt':
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


_seed_warned: Set[int] = set()


def _warn_unseeded_server(port: int) -> None:
    """Un servidor ya encendido no cambia su azar: avisa una vez por puerto."""
    if not os.environ.get('PS_BATTLE_SEED', '').strip() or port in _ps_procs:
        return
    if port in _seed_warned:
        return
    _seed_warned.add(port)
    print(
        f'[ps_server] AVISO: el servidor del puerto {port} ya estaba encendido; '
        'PS_BATTLE_SEED no se aplica a sus batallas. Usa --restart-ps o ciérralo.',
        flush=True,
    )


def ensure_ps(
    start: bool = False,
    wait_s: float = 60.0,
    *,
    restart: bool = False,
    port: Optional[int] = None,
) -> bool:
    """True si el servidor responde en el puerto.

    - start=True: arranca si está caído (con --no-security por defecto)
    - restart=True: mata listeners del puerto y arranca de nuevo
    No se llama por cada BO3: el proceso vive durante todo el batch.
    """
    p = int(PS_PORT if port is None else port)
    if restart:
        stop_ps_listeners(p)
        start = True
    if is_ps_up(port=p) and not restart:
        _warn_unseeded_server(p)
        return True
    if not start:
        return False
    ensure_ps_build()
    _ps_procs[p] = _spawn_ps(p)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if is_ps_up(port=p):
            return True
        time.sleep(0.25)
    return False


def ensure_ps_ports(
    ports: List[int],
    *,
    start: bool = False,
    restart: bool = False,
    wait_s: float = 60.0,
) -> bool:
    """Asegura todos los puertos (paralelismo horizontal).

    Con ``start``, lanza a la vez todos los servidores caídos y luego espera
    (24 servidores arrancan en segundos, no uno tras otro).
    """
    ports = [int(p) for p in ports]
    if restart:
        for p in ports:
            stop_ps_listeners(p)
        start = True
    down = [p for p in ports if not is_ps_up(port=p)]
    for p in ports:
        if p not in down:
            _warn_unseeded_server(p)
    if not down:
        return True
    if not start:
        return False
    ensure_ps_build()
    # Por tandas (PS_START_PARALLEL, default 8): cada servidor lanza ~10
    # procesos node; decenas a la vez saturan el arranque.
    wave = _start_parallelism()
    for k in range(0, len(down), wave):
        group = down[k:k + wave]
        for p in group:
            _ps_procs[p] = _spawn_ps(p)
        deadline = time.time() + wait_s
        pending = set(group)
        while pending and time.time() < deadline:
            pending = {p for p in pending if not is_ps_up(port=p)}
            if pending:
                time.sleep(0.25)
        if pending:
            return False
    return True


def stop_started_ps(port: Optional[int] = None) -> None:
    if port is None:
        ports = list(_ps_procs.keys())
    else:
        ports = [int(port)]
    for p in ports:
        proc = _ps_procs.pop(p, None)
        if proc is None:
            continue
        _kill_process_tree(proc)
