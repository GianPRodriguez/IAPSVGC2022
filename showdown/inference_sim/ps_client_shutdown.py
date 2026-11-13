"""Cierre limpio de PSClient: evita 'Task was destroyed but it is pending!'.

poke-env crea un Task por mensaje (`_handle_message`). `stop_listening` cierra
el websocket pero, en versiones sin drain, deja handlers en vuelo; al destruir
el loop asyncio avisa. Aquí cancelamos y esperamos `_active_tasks`.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional


async def drain_ps_client_tasks(
    client: Any,
    *,
    timeout_s: float = 8.0,
) -> int:
    """Cancela y await de `_active_tasks`. Devuelve cuántas tasks se drenaron.

    CancelledError en handlers es esperado (shutdown); gather(..., return_exceptions).
    """
    raw = getattr(client, '_active_tasks', None)
    if not raw:
        return 0
    pending = [t for t in list(raw) if t is not None and not t.done()]
    if not pending:
        return 0
    for t in pending:
        t.cancel()
    try:
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True),
            timeout=float(timeout_s),
        )
    except Exception:
        pass
    return len(pending)


async def stop_player_client(player: Any, *, timeout_s: float = 8.0) -> None:
    """stop_listening + drain de handlers (idempotente, traga errores)."""
    if player is None:
        return
    client = getattr(player, 'ps_client', None)
    if client is None:
        return
    try:
        await client.stop_listening()
    except Exception:
        pass
    try:
        await drain_ps_client_tasks(client, timeout_s=timeout_s)
    except Exception:
        pass


async def stop_players(
    *players: Any,
    timeout_s: float = 8.0,
) -> None:
    for pl in players:
        await stop_player_client(pl, timeout_s=timeout_s)
