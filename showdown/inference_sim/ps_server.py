"""Healthcheck / ensure del servidor Pokémon Showdown local."""
from __future__ import annotations

import os
import socket
import subprocess
import time
from typing import List, Optional, Set

from showdown.inference_sim.paths import POKEMON_SHOWDOWN_ROOT, PS_HOST, PS_PORT

_ps_proc: Optional[subprocess.Popen] = None

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


def ps_start_argv() -> List[str]:
    """Comando canónico para arrancar PS Champions local."""
    argv = ['node', 'pokemon-showdown', 'start']
    if ps_no_security_enabled():
        argv.append('--no-security')
    return argv


def ps_start_hint() -> str:
    return ' '.join(ps_start_argv())


def is_ps_up(host: str = PS_HOST, port: int = PS_PORT, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _pids_listening_on_port(port: int = PS_PORT) -> Set[int]:
    """PIDs que escuchan el puerto (Windows netstat / Unix lsof-ish via netstat)."""
    pids: Set[int] = set()
    try:
        # Windows: netstat -ano | findstr :8000
        out = subprocess.check_output(
            ['netstat', '-ano', '-p', 'tcp'],
            text=True,
            errors='replace',
            timeout=10,
        )
        needle = f':{port}'
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
    global _ps_proc
    stop_started_ps()
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
    # Espera a que el puerto quede libre
    deadline = time.time() + 15
    while time.time() < deadline and is_ps_up(port=port):
        time.sleep(0.2)
    return killed


def _spawn_ps() -> subprocess.Popen:
    if not POKEMON_SHOWDOWN_ROOT.is_dir():
        raise FileNotFoundError(f'No existe pokemon-showdown en {POKEMON_SHOWDOWN_ROOT}')
    cfg = POKEMON_SHOWDOWN_ROOT / 'config' / 'config.js'
    example = POKEMON_SHOWDOWN_ROOT / 'config' / 'config-example.js'
    if not cfg.exists() and example.exists():
        cfg.write_text(example.read_text(encoding='utf-8'), encoding='utf-8')
    return subprocess.Popen(
        ps_start_argv(),
        cwd=str(POKEMON_SHOWDOWN_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ensure_ps(start: bool = False, wait_s: float = 60.0, *, restart: bool = False) -> bool:
    """True si el servidor responde.

    - start=True: arranca si está caído (con --no-security por defecto)
    - restart=True: mata listeners del puerto y arranca de nuevo con --no-security
    """
    global _ps_proc
    if restart:
        stop_ps_listeners()
        start = True
    if is_ps_up() and not restart:
        return True
    if not start:
        return False
    _ps_proc = _spawn_ps()
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if is_ps_up():
            return True
        time.sleep(0.25)
    return False


def stop_started_ps() -> None:
    global _ps_proc
    if _ps_proc is not None:
        try:
            _ps_proc.terminate()
            _ps_proc.wait(timeout=5)
        except Exception:
            try:
                _ps_proc.kill()
            except Exception:
                pass
        _ps_proc = None
