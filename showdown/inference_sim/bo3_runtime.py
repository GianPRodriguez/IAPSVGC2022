"""Reset / drain del bookkeeping BO3 de poke-env (anti hang 600s).

Tras TimeoutError sin |win| en game-bestof*, `_battle_count_queue` queda con
un item sin task_done. La siguiente serie puede jugar los 3 games y aun así
`battle_against` nunca sale de `join()` → TimeoutError(600s) en cascada.
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Sequence, Tuple


def bo3_series_decided(games: Sequence[Tuple[int, bool]]) -> bool:
    """True si el BO3 ya tiene ganador (2 wins / 2 losses / 3 games)."""
    if not games:
        return False
    wins = sum(1 for _turns, won in games if won)
    losses = len(games) - wins
    return len(games) >= 3 or wins >= 2 or losses >= 2


def drain_battle_count_queue(player: Any) -> int:
    """Vacía `_battle_count_queue` marcando task_done. Devuelve items drenados.

    Solo usar al abandonar una serie (timeout/reject/cleanup). Drenar mientras
    `battle_against` espera `join()` hace que la serie parezca 'ok' incompleta
    (short_bo3).
    """
    q = getattr(player, '_battle_count_queue', None)
    if q is None:
        return 0
    n = 0
    while True:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break
        except Exception:
            break
        try:
            q.task_done()
        except ValueError:
            # task_done de más: ignorar
            pass
        n += 1
    return n


def reset_bo3_runtime_state(player: Any, *, drain_queue: bool = True) -> None:
    """Limpia estado BO3 entre series / tras timeout (idempotente)."""
    try:
        player._bestof_games = set()
    except Exception:
        pass
    if drain_queue:
        drain_battle_count_queue(player)
    # Batallas hijas huérfanas tras cancel: evitar referencias stale.
    try:
        battles = getattr(player, '_battles', None)
        if isinstance(battles, dict):
            finished = [
                k for k, b in list(battles.items())
                if getattr(b, 'finished', False)
            ]
            for k in finished:
                battles.pop(k, None)
    except Exception:
        pass


def reset_bo3_runtime_state_many(
    players: List[Any],
    *,
    drain_queue: bool = True,
) -> None:
    for pl in players:
        if pl is not None:
            reset_bo3_runtime_state(pl, drain_queue=drain_queue)
