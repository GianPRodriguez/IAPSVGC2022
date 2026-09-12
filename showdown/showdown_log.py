"""Append-only logger for raw Pokémon Showdown websocket messages."""
from __future__ import annotations

import os
from datetime import datetime
from threading import Lock

_LOG_PATH = os.path.join('logs', 'logs_showdown.txt')
_LOCK = Lock()


def append_showdown_log(message: str, *, direction: str = 'recv') -> None:
    """Append one raw PS message to logs/logs_showdown.txt."""
    if message is None:
        return
    text = str(message)
    if not text.strip():
        return

    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    block = (
        f"\n{'=' * 72}\n"
        f"  {direction.upper()}  {stamp}\n"
        f"{'=' * 72}\n"
        f"{text.rstrip()}\n"
    )
    try:
        os.makedirs('logs', exist_ok=True)
        with _LOCK:
            with open(_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(block)
    except OSError as exc:
        print(f"[WARN] No se pudo escribir {_LOG_PATH}: {exc}")


def mark_showdown_battle(tag: str | None = None) -> None:
    """Separator when a new battle / BO3 game starts."""
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    label = tag or 'nueva batalla'
    block = (
        f"\n{'#' * 72}\n"
        f"  BATTLE START  {stamp}  |  {label}\n"
        f"{'#' * 72}\n"
    )
    try:
        os.makedirs('logs', exist_ok=True)
        with _LOCK:
            with open(_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(block)
    except OSError as exc:
        print(f"[WARN] No se pudo escribir {_LOG_PATH}: {exc}")


_REPORT_PATH = os.path.join('logs', 'reporte.txt')
_LATEST_PATH = os.path.join('logs', 'latest_state.txt')


def clear_battle_debug_logs() -> None:
    """Vacía reporte + logs Showdown al iniciar un nuevo BO3 (deja el set anterior revisable)."""
    os.makedirs('logs', exist_ok=True)
    with _LOCK:
        for path in (_REPORT_PATH, _LOG_PATH, _LATEST_PATH):
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write('')
            except OSError as exc:
                print(f"[WARN] No se pudo limpiar {path}: {exc}")
    print("[INFO]     Logs de depuración limpiados (reporte / logs_showdown / latest_state)")
