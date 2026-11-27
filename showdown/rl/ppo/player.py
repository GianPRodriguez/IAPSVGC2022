"""poke-env PPOPlayer + ValidatedPPOPlayer factory helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from showdown.rl.family_preview_mixin import FamilyPreviewMixin
from showdown.rl.observation import build_observation_ppo
from showdown.rl.pe_adapter import (
    joint_order_from_spec_actions,
    pe_autoregressive_masks,
    player_stat_beliefs,
    snapshot_from_battle,
)
from showdown.rl.ppo.agent import PPOAgent, PPOConfig


class PPOPlayer(FamilyPreviewMixin, Player):
    """
    poke-env player driven by a Masked PPO checkpoint.

    Team preview: frozen ``team_preview_ppo`` (fallback ``/team 1234``).
    """

    _combat_family_id = 'ppo'

    def __init__(
        self,
        *args,
        checkpoint: Optional[str] = None,
        greedy: bool = True,
        temperature: float = 1.0,
        preview_checkpoint: Optional[str] = None,
        preview_kind: Optional[str] = None,
        **kwargs,
    ):
        kwargs.setdefault('accept_open_team_sheet', True)
        super().__init__(*args, **kwargs)
        self.greedy = bool(greedy)
        # Solo al muestrear (greedy=False): PPO v2 contra personas usa T = 0,5 (memorizacion_equipos §19).
        self.temperature = float(temperature)
        self.agent = PPOAgent(PPOConfig(device=os.environ.get('PPO_DEVICE', 'cpu')))
        ckpt = checkpoint or os.environ.get('PPO_CHECKPOINT') or ''
        self.checkpoint_path = str(ckpt).strip()
        if self.checkpoint_path and Path(self.checkpoint_path).is_file():
            self.agent.load(self.checkpoint_path)
        if getattr(self.agent.cfg, 'mask_protect_streak', False):
            os.environ['RL_MASK_PROTECT_STREAK'] = '1'  # PPO v3: máscara del Protect consecutivo
        self.last_actions = (0, 0)
        self.init_family_preview(
            preview_checkpoint=preview_checkpoint,
            preview_kind=preview_kind,
        )
        from showdown.rl.decision_guard import DecisionStats
        self.decision_stats = DecisionStats(agent='ppo')

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)
        try:
            snap = snapshot_from_battle(battle, stat_beliefs=player_stat_beliefs(self))
            obs = build_observation_ppo(snap, abilities=bool(getattr(self.agent.cfg, 'obs_abilities', False)),
                                        memory=bool(getattr(self.agent.cfg, 'series_memory', False)))
            m1, _m2u = pe_autoregressive_masks(battle, slot1_action_id=None)

            def m2_fn(a1: int):
                _m1, m2 = pe_autoregressive_masks(battle, slot1_action_id=a1)
                return m2

            a1, a2, _lp, _v = self.agent.act(obs, m1, m2_fn, greedy=self.greedy,
                                              temperature=self.temperature)
            self.last_actions = (int(a1), int(a2))
            order, used_fallback = joint_order_from_spec_actions(battle, (a1, a2))
            self.decision_stats.record_ok(order_fallback=bool(used_fallback))
            return order
        except Exception as exc:
            # Como DDQN (rl_env_spec v1.22): se cuenta, se registra y con PS_RL_STRICT=1 se propaga.
            self.decision_stats.record_exception(exc)
            return DefaultBattleOrder()


def make_validated_ppo_player_class():
    """Lazy: ValidatedBattleMixin + PPOPlayer."""
    from showdown.inference_sim.validated_greedy_player import ValidatedBattleMixin

    class ValidatedPPOPlayer(ValidatedBattleMixin, PPOPlayer):
        """PPO + IAP mirror sync / OTS."""

    return ValidatedPPOPlayer
