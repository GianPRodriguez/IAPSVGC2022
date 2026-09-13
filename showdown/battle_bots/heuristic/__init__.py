"""Alias de compatibilidad: BATTLE_BOT=heuristic → Greedy-Solo."""

from showdown.battle_bots.greedy_solo.main import BattleBot
from showdown.battle_bots.greedy_solo.pokeenv_player import GreedyDamagePlayer

__all__ = ['BattleBot', 'GreedyDamagePlayer']
