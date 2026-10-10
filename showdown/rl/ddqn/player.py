"""poke-env DDQNPlayer + ValidatedDDQNPlayer factory helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from showdown.rl.action_space import decode_joint_action
from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
from showdown.rl.observation import build_observation_ddqn
from showdown.rl.pe_adapter import (
    joint_order_from_spec_actions,
    legal_joint_mask,
    snapshot_from_battle,
)


class DDQNPlayer(Player):
    """
    poke-env player driven by a joint DDQN checkpoint.

    Team preview: Fixed Export Order `/team 1234` (DDQNSpec).
    """

    def __init__(self, *args, checkpoint: Optional[str] = None, epsilon: float = 0.0, **kwargs):
        kwargs.setdefault('accept_open_team_sheet', True)
        super().__init__(*args, **kwargs)
        self.epsilon = float(epsilon)
        self.agent = DDQNAgent(DDQNConfig(device=os.environ.get('DDQN_DEVICE', 'cpu')))
        ckpt = checkpoint or os.environ.get('DDQN_CHECKPOINT') or ''
        self.checkpoint_path = str(ckpt).strip()
        if self.checkpoint_path and Path(self.checkpoint_path).is_file():
            self.agent.load(self.checkpoint_path)
        self.last_joint: int = 0

    def teampreview(self, battle: AbstractBattle) -> str:
        # Fixed Export Order — first four slots
        return '/team 1234'

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)
        try:
            snap = snapshot_from_battle(battle)
            obs = build_observation_ddqn(snap)
            mask = legal_joint_mask(battle)
            joint = self.agent.act(obs, mask, epsilon=self.epsilon)
            self.last_joint = int(joint)
            a1, a2 = decode_joint_action(joint)
            order, _fb = joint_order_from_spec_actions(battle, (a1, a2))
            return order
        except Exception:
            return DefaultBattleOrder()


def make_validated_ddqn_player_class():
    """Lazy: ValidatedBattleMixin + DDQNPlayer (needs inference_sim)."""
    from showdown.inference_sim.validated_greedy_player import ValidatedBattleMixin

    class ValidatedDDQNPlayer(ValidatedBattleMixin, DDQNPlayer):
        """DDQN + IAP mirror sync / OTS."""

    return ValidatedDDQNPlayer
