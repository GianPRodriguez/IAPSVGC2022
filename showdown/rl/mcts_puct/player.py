"""poke-env MCTS+PUCT player + Validated factory."""
from __future__ import annotations

import os
from typing import Optional

from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from showdown.rl.action_space import decode_joint_action
from showdown.rl.mcts_puct.agent import MCTSPuctAgent, MCTSPuctConfig
from showdown.rl.pe_adapter import joint_order_from_spec_actions, snapshot_from_battle


class MCTSPuctPlayer(Player):
    """Fixed Export Order `/team 1234` + MCTS+PUCT joint policy."""

    def __init__(self, *args, n_sims: Optional[int] = None, seed: int = 0, **kwargs):
        kwargs.setdefault('accept_open_team_sheet', True)
        super().__init__(*args, **kwargs)
        sims = n_sims
        if sims is None:
            sims = int(os.environ.get('MCTS_SIMS', '200'))
        self.agent = MCTSPuctAgent(
            MCTSPuctConfig(
                n_sims=int(sims),
                seed=int(os.environ.get('MCTS_SEED', seed)),
            )
        )
        self.last_joint = 0

    def teampreview(self, battle: AbstractBattle) -> str:
        return '/team 1234'

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)
        try:
            snap = snapshot_from_battle(battle)
            joint = self.agent.act(snap)
            self.last_joint = int(joint)
            a1, a2 = decode_joint_action(joint)
            order, _fb = joint_order_from_spec_actions(battle, (a1, a2))
            return order
        except Exception:
            return DefaultBattleOrder()


def make_validated_mcts_puct_player_class():
    from showdown.inference_sim.validated_greedy_player import ValidatedBattleMixin

    class ValidatedMCTSPuctPlayer(ValidatedBattleMixin, MCTSPuctPlayer):
        """MCTS+PUCT + IAP mirror sync / OTS."""

    return ValidatedMCTSPuctPlayer
