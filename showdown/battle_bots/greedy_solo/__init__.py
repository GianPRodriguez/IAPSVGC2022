"""Greedy-Solo: política greedy myopic por mon (1-ply factored)."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ['BattleBot', 'GreedyDamagePlayer']


def __getattr__(name: str):
    if name == 'BattleBot':
        from showdown.battle_bots.greedy_solo.main import BattleBot
        return BattleBot
    if name == 'GreedyDamagePlayer':
        from showdown.battle_bots.greedy_solo.pokeenv_player import GreedyDamagePlayer
        return GreedyDamagePlayer
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


if TYPE_CHECKING:
    from showdown.battle_bots.greedy_solo.main import BattleBot as BattleBot
    from showdown.battle_bots.greedy_solo.pokeenv_player import (
        GreedyDamagePlayer as GreedyDamagePlayer,
    )
