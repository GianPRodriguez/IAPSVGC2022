"""Healthcheck / ensure del servidor Pokémon Showdown local."""
from __future__ import annotations

import socket
import subprocess
import time
from typing import Optional

from showdown.inference_sim.paths import POKEMON_SHOWDOWN_ROOT, PS_HOST, PS_PORT

_ps_proc: Optional[subprocess.Popen] = None


def is_ps_up(host: str = PS_HOST, port: int = PS_PORT, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_ps(start: bool = False, wait_s: float = 60.0) -> bool:
    """True si el servidor responde. Si start=True y está caído, intenta arrancarlo."""
    global _ps_proc
    if is_ps_up():
        return True
    if not start:
        return False
    if not POKEMON_SHOWDOWN_ROOT.is_dir():
        raise FileNotFoundError(f'No existe pokemon-showdown en {POKEMON_SHOWDOWN_ROOT}')
    cfg = POKEMON_SHOWDOWN_ROOT / 'config' / 'config.js'
    example = POKEMON_SHOWDOWN_ROOT / 'config' / 'config-example.js'
    if not cfg.exists() and example.exists():
        cfg.write_text(example.read_text(encoding='utf-8'), encoding='utf-8')
    _ps_proc = subprocess.Popen(
        ['node', 'pokemon-showdown', 'start'],
        cwd=str(POKEMON_SHOWDOWN_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if is_ps_up():
            return True
        time.sleep(0.25)
    return False


def stop_started_ps() -> None:
    global _ps_proc
    if _ps_proc is not None:
        _ps_proc.terminate()
        try:
            _ps_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _ps_proc.kill()
        _ps_proc = None
