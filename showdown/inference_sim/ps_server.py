"""Healthcheck / ensure del servidor Pokémon Showdown local (1+ puertos)."""
from __future__ import annotations

import os
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
    argv = ['node', 'pokemon-showdown', 'start']
    if ps_no_security_enabled():
        argv.append('--no-security')
    if port is not None:
        argv.append(str(int(port)))
    return argv


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
    """PIDs que escuchan el puerto (Windows netstat)."""
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
    env = os.environ.copy()
    env['PSPORT'] = str(int(port))
    return subprocess.Popen(
        ps_start_argv(port),
        cwd=str(POKEMON_SHOWDOWN_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
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
        return True
    if not start:
        return False
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
    """Asegura todos los puertos (paralelismo horizontal)."""
    ok = True
    for p in ports:
        if not ensure_ps(start=start, restart=restart, wait_s=wait_s, port=p):
            ok = False
    return ok


def stop_started_ps(port: Optional[int] = None) -> None:
    if port is None:
        ports = list(_ps_procs.keys())
    else:
        ports = [int(port)]
    for p in ports:
        proc = _ps_procs.pop(p, None)
        if proc is None:
            continue
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
