"""BO3 runtime: drain de cola y detección de serie decidida."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from showdown.inference_sim.bo3_runtime import (
    bo3_series_decided,
    drain_battle_count_queue,
    reset_bo3_runtime_state,
)


def test_bo3_series_decided():
    assert not bo3_series_decided([])
    assert not bo3_series_decided([(5, True)])
    assert bo3_series_decided([(5, True), (6, True)])
    assert bo3_series_decided([(5, False), (6, False)])
    assert bo3_series_decided([(5, True), (6, False), (7, True)])
    assert not bo3_series_decided([(5, True), (6, False)])


def test_drain_battle_count_queue():
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait(None)
    q.put_nowait(None)
    player = SimpleNamespace(_battle_count_queue=q, _bestof_games={'x'}, _battles={})
    assert drain_battle_count_queue(player) == 2
    assert q.empty()
    assert drain_battle_count_queue(player) == 0


def test_reset_bo3_clears_bestof_and_queue():
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait(None)
    finished = SimpleNamespace(finished=True)
    live = SimpleNamespace(finished=False)
    player = SimpleNamespace(
        _battle_count_queue=q,
        _bestof_games={'game-bestof3-old'},
        _battles={'battle-a': finished, 'battle-b': live},
    )
    reset_bo3_runtime_state(player)
    assert player._bestof_games == set()
    assert q.empty()
    assert 'battle-a' not in player._battles
    assert 'battle-b' in player._battles
