"""Re-export poke-env DDQN players for battle_bots convenience."""
from showdown.rl.ddqn.player import DDQNPlayer, make_validated_ddqn_player_class

ValidatedDDQNPlayer = make_validated_ddqn_player_class()

__all__ = ['DDQNPlayer', 'ValidatedDDQNPlayer']
