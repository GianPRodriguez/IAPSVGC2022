"""Re-export poke-env PPO players for battle_bots convenience."""
from showdown.rl.ppo.player import PPOPlayer, make_validated_ppo_player_class

ValidatedPPOPlayer = make_validated_ppo_player_class()

__all__ = ['PPOPlayer', 'ValidatedPPOPlayer']
