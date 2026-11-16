"""poke-env DDQNPlayer + ValidatedDDQNPlayer factory helpers."""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from showdown.rl.action_space import decode_joint_action
from showdown.rl.ddqn.agent import DDQNAgent, DDQNConfig
from showdown.rl.decision_guard import DecisionStats
from showdown.rl.family_preview_mixin import FamilyPreviewMixin
from showdown.rl.observation import build_observation_ddqn
from showdown.rl.pe_adapter import (
    joint_order_from_spec_actions,
    legal_joint_mask,
    player_stat_beliefs,
    snapshot_from_battle,
)


class DDQNPlayer(FamilyPreviewMixin, Player):
    """
    poke-env player driven by a joint DDQN checkpoint.

    Team preview: frozen ``team_preview_ppo`` (fallback ``/team 1234``).
    """

    _combat_family_id = 'ddqn'

    def __init__(
        self,
        *args,
        checkpoint: Optional[str] = None,
        epsilon: float = 0.0,
        preview_checkpoint: Optional[str] = None,
        preview_kind: Optional[str] = None,
        required: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('accept_open_team_sheet', True)
        super().__init__(*args, **kwargs)
        self.epsilon = float(epsilon)
        self.agent = DDQNAgent(DDQNConfig(device=os.environ.get('DDQN_DEVICE', 'cpu')))
        ckpt = checkpoint or os.environ.get('DDQN_CHECKPOINT') or ''
        self.checkpoint_path = str(ckpt).strip()
        self.checkpoint_loaded = False
        if self.checkpoint_path and Path(self.checkpoint_path).is_file():
            self.agent.load(self.checkpoint_path)
            self.checkpoint_loaded = True
        if not self.checkpoint_loaded:
            msg = (f'DDQN sin checkpoint (ruta={self.checkpoint_path or "<vacía>"}): '
                   'jugará con pesos aleatorios de inicialización')
            if required or os.environ.get('DDQN_REQUIRED', '').strip().lower() in ('1', 'true', 'yes'):
                raise FileNotFoundError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        self.last_joint: int = 0
        self.decision_stats = DecisionStats(agent='ddqn')
        self.init_family_preview(
            preview_checkpoint=preview_checkpoint,
            preview_kind=preview_kind,
        )

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)
        try:
            snap = snapshot_from_battle(battle, stat_beliefs=player_stat_beliefs(self))
            obs = build_observation_ddqn(snap)
            mask = legal_joint_mask(battle)
            joint = self.agent.act(obs, mask, epsilon=self.epsilon)
            self.last_joint = int(joint)
            a1, a2 = decode_joint_action(joint)
            order, used_fallback = joint_order_from_spec_actions(battle, (a1, a2))
            self.decision_stats.record_ok(order_fallback=bool(used_fallback))
            return order
        except Exception as exc:
            # Ya no se traga en silencio: se cuenta, se registra y en PS_RL_STRICT=1 se propaga.
            self.decision_stats.record_exception(exc)
            return DefaultBattleOrder()


def make_validated_ddqn_player_class():
    """Lazy: ValidatedBattleMixin + DDQNPlayer (needs inference_sim)."""
    from showdown.inference_sim.validated_greedy_player import ValidatedBattleMixin

    class ValidatedDDQNPlayer(ValidatedBattleMixin, DDQNPlayer):
        """DDQN + IAP mirror sync / OTS."""

    return ValidatedDDQNPlayer
