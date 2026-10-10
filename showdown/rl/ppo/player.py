"""poke-env PPOPlayer + ValidatedPPOPlayer factory helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from showdown.rl.observation import build_observation_ppo
from showdown.rl.pe_adapter import (
    joint_order_from_spec_actions,
    pe_autoregressive_masks,
    snapshot_from_battle,
)
from showdown.rl.ppo.agent import PPOAgent, PPOConfig


class PPOPlayer(Player):
    """
    poke-env player driven by a Masked PPO checkpoint.

    Team preview: Fixed Export Order `/team 1234` (PPOSpec).
    """

    def __init__(self, *args, checkpoint: Optional[str] = None, greedy: bool = True, **kwargs):
        kwargs.setdefault('accept_open_team_sheet', True)
        super().__init__(*args, **kwargs)
        self.greedy = bool(greedy)
        self.agent = PPOAgent(PPOConfig(device=os.environ.get('PPO_DEVICE', 'cpu')))
        ckpt = checkpoint or os.environ.get('PPO_CHECKPOINT') or ''
        self.checkpoint_path = str(ckpt).strip()
        if self.checkpoint_path and Path(self.checkpoint_path).is_file():
            self.agent.load(self.checkpoint_path)
        self.last_actions = (0, 0)

    def teampreview(self, battle: AbstractBattle) -> str:
        return '/team 1234'

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)
        try:
            snap = snapshot_from_battle(battle)
            obs = build_observation_ppo(snap)
            m1, _m2u = pe_autoregressive_masks(battle, slot1_action_id=None)

            def m2_fn(a1: int):
                _m1, m2 = pe_autoregressive_masks(battle, slot1_action_id=a1)
                return m2

            a1, a2, _lp, _v = self.agent.act(obs, m1, m2_fn, greedy=self.greedy)
            self.last_actions = (int(a1), int(a2))
            order, _fb = joint_order_from_spec_actions(battle, (a1, a2))
            return order
        except Exception:
            return DefaultBattleOrder()


def make_validated_ppo_player_class():
    """Lazy: ValidatedBattleMixin + PPOPlayer."""
    from showdown.inference_sim.validated_greedy_player import ValidatedBattleMixin

    class ValidatedPPOPlayer(ValidatedBattleMixin, PPOPlayer):
        """PPO + IAP mirror sync / OTS."""

    return ValidatedPPOPlayer
