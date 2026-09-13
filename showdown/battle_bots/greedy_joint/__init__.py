"""Greedy-Joint: 1-ply joint action search (A1 × A2) para VGC doubles."""

from showdown.battle_bots.greedy_joint.main import BattleBot
from showdown.battle_bots.greedy_joint.pokeenv_player import GreedyJointPlayer

# Alias para scripts que esperan el nombre legacy del player.
GreedyDamagePlayer = GreedyJointPlayer

__all__ = ['BattleBot', 'GreedyJointPlayer', 'GreedyDamagePlayer']
