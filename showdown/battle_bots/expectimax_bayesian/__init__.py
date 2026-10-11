"""Expectimax-Bayesian: 1-ply anytime con creencia MetaVGC + π_opp + soft P(KO)."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ['BattleBot', 'ExpectimaxBayesianPlayer']


def __getattr__(name: str):
    if name == 'BattleBot':
        from showdown.battle_bots.expectimax_bayesian.main import BattleBot
        return BattleBot
    if name == 'ExpectimaxBayesianPlayer':
        from showdown.battle_bots.expectimax_bayesian.pokeenv_player import (
            ExpectimaxBayesianPlayer,
        )
        return ExpectimaxBayesianPlayer
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


if TYPE_CHECKING:
    from showdown.battle_bots.expectimax_bayesian.main import BattleBot as BattleBot
    from showdown.battle_bots.expectimax_bayesian.pokeenv_player import (
        ExpectimaxBayesianPlayer as ExpectimaxBayesianPlayer,
    )
