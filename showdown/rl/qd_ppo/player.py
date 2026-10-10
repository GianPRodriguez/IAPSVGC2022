"""poke-env QD-PPO player (π*) + Validated factory."""
from __future__ import annotations

import os
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
from showdown.rl.qd_ppo.agent import QDPPOAgent


class QDPPOPlayer(Player):
    """Fixed Export Order `/team 1234` + best-elite QD-PPO policy."""

    def __init__(
        self,
        *args,
        checkpoint: Optional[str] = None,
        archive_path: Optional[str] = None,
        greedy: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('accept_open_team_sheet', True)
        super().__init__(*args, **kwargs)
        self.greedy = bool(greedy)
        self.agent = QDPPOAgent(
            checkpoint=checkpoint or os.environ.get('QD_PPO_CHECKPOINT'),
            archive_path=archive_path or os.environ.get('QD_PPO_ARCHIVE'),
        )
        self.last_actions = (0, 0)

    def teampreview(self, battle: AbstractBattle) -> str:
        return '/team 1234'

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)
        try:
            snap = snapshot_from_battle(battle)
            obs = build_observation_ppo(snap)
            m1, _ = pe_autoregressive_masks(battle, slot1_action_id=None)

            def m2_fn(a1: int):
                _m1, m2 = pe_autoregressive_masks(battle, slot1_action_id=a1)
                return m2

            a1, a2, _lp, _v = self.agent.act(obs, m1, m2_fn, greedy=self.greedy)
            self.last_actions = (int(a1), int(a2))
            order, _fb = joint_order_from_spec_actions(battle, (a1, a2))
            return order
        except Exception:
            return DefaultBattleOrder()


def make_validated_qd_ppo_player_class():
    from showdown.inference_sim.validated_greedy_player import ValidatedBattleMixin

    class ValidatedQDPPOPlayer(ValidatedBattleMixin, QDPPOPlayer):
        """QD-PPO π* + IAP mirror sync / OTS."""

    return ValidatedQDPPOPlayer
