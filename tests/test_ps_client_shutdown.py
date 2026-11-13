"""Shutdown limpio de PSClient (anti Task destroyed pending)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from showdown.inference_sim.ps_client_shutdown import (
    drain_ps_client_tasks,
    stop_player_client,
)


def test_drain_cancels_pending_handle_message_tasks():
    async def _run():
        started = asyncio.Event()
        block = asyncio.Event()

        async def _hang():
            started.set()
            await block.wait()

        t = asyncio.create_task(_hang())
        client = SimpleNamespace(_active_tasks={t})
        await started.wait()
        n = await drain_ps_client_tasks(client, timeout_s=2.0)
        assert n == 1
        assert t.done()

    asyncio.run(_run())


def test_stop_player_client_calls_stop_then_drain():
    async def _run():
        hang = asyncio.Event()

        async def _hang():
            await hang.wait()

        t = asyncio.create_task(_hang())
        client = SimpleNamespace(
            _active_tasks={t},
            stop_listening=AsyncMock(),
        )
        player = SimpleNamespace(ps_client=client)
        await stop_player_client(player, timeout_s=2.0)
        client.stop_listening.assert_awaited_once()
        assert t.done()

    asyncio.run(_run())
