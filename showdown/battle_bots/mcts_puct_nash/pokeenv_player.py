from showdown.rl.mcts_puct_nash.player import (
    MCTSPuctNashPlayer,
    make_validated_mcts_puct_nash_player_class,
)

ValidatedMCTSPuctNashPlayer = make_validated_mcts_puct_nash_player_class()

__all__ = ['MCTSPuctNashPlayer', 'ValidatedMCTSPuctNashPlayer']
