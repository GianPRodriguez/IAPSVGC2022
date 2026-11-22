"""poke-env MCTS–PUCT/Nash player (agente H) + Validated factory."""
from __future__ import annotations

import logging
import os
from typing import Optional

from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from showdown.rl.action_space import decode_joint_action
from showdown.rl.family_preview_mixin import FamilyPreviewMixin
from showdown.rl.mcts_puct.player import MCTSPuctPlayer, _env_int, _record_anomaly
from showdown.rl.mcts_puct_nash.agent import MCTSPuctNashAgent, MCTSPuctNashConfig
from showdown.rl.pe_adapter import joint_order_from_spec_actions, snapshot_from_battle

_LOG = logging.getLogger(__name__)


class MCTSPuctNashPlayer(FamilyPreviewMixin, Player):
    """Preview por búsqueda (el de G 5.2) + SM-MCTS/Nash.

    ``MCTS_NASH_VERSION`` (2 por defecto = H 2.0: S3 fase 3c open-loop, docs/mcts_puct_nash_spec.md
    v2.0; 1 = SM-MCTS B1 original con S2). Preview: el mismo de G (``MCTS_PREVIEW``: ``search`` por
    defecto, ``greedy``, ``nash``); ``team_preview_nash`` nunca se entrenó y caía al orden fijo
    ``/team 1234``. Presupuesto ``MCTS_NASH_SIMS`` (fallback ``MCTS_SIMS``; 200 en v2, como G).
    Anomalías en ``MCTS_STATS_FILE`` (excepción, fallback del adaptador, calculadora, preview).
    """

    _combat_family_id = 'mcts_puct_nash'

    # Preview por búsqueda de G (spec de G §4.6, confirmado en protocolo §5.5).
    teampreview = MCTSPuctPlayer.teampreview
    _search_preview_async = MCTSPuctPlayer._search_preview_async
    _search_preview = MCTSPuctPlayer._search_preview
    _root_joints = MCTSPuctPlayer._root_joints

    def __init__(
        self,
        *args,
        n_sims: Optional[int] = None,
        seed: int = 0,
        preview_checkpoint: Optional[str] = None,
        preview_kind: Optional[str] = None,
        **kwargs,
    ):
        kwargs.setdefault('accept_open_team_sheet', True)
        super().__init__(*args, **kwargs)
        version = _env_int('MCTS_NASH_VERSION', 2)
        sims = n_sims
        if sims is None:
            sims = int(os.environ.get('MCTS_NASH_SIMS', os.environ.get('MCTS_SIMS', '200' if version >= 2 else '50')))
        self.agent = MCTSPuctNashAgent(
            MCTSPuctNashConfig(
                n_sims=int(sims),
                seed=int(os.environ.get('MCTS_NASH_SEED', os.environ.get('MCTS_SEED', seed))),
                version=version,
            )
        )
        self.last_joint = 0
        self.mcts_stats = {
            'decisions': 0, 'exceptions': 0, 'adapter_fallbacks': 0,
            'trivial': 0, 'search_sec': 0.0,
            'calc_requests': 0, 'calc_missing': 0, 'calc_errors': 0, 'calc_sec': 0.0,
            'foe_no_moves': 0,
        }
        self.init_family_preview(
            preview_checkpoint=preview_checkpoint or os.environ.get('MCTS_PREVIEW_CKPT') or None,
            preview_kind=preview_kind,
        )

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)
        st = self.mcts_stats
        st['decisions'] += 1
        try:
            snap = snapshot_from_battle(battle)
            v2 = int(self.agent.cfg.version) >= 2
            root = self._root_joints(battle) if v2 else None
            beliefs = getattr(getattr(self, 'iap_battle', None), 'stat_beliefs', None)
            joint = self.agent.act(snap, root_joints=root, stat_beliefs=beliefs)
            tel = self.agent.last_telemetry
            if tel is not None and v2:
                st['search_sec'] += float(getattr(tel, 'elapsed_sec', 0.0) or 0.0)
                st['trivial'] += int(bool(getattr(tel, 'trivial', False)))
                st['calc_requests'] += int(getattr(tel, 'calc_requests', 0) or 0)
                st['calc_missing'] += int(getattr(tel, 'calc_missing', 0) or 0)
                st['calc_sec'] += float(getattr(tel, 'calc_sec', 0.0) or 0.0)
                st['foe_no_moves'] += int(getattr(tel, 'foe_no_moves', 0) or 0)
                if getattr(tel, 'calc_error', ''):
                    st['calc_errors'] += 1
                    _record_anomaly('calc_error', battle, str(tel.calc_error))
                if int(getattr(tel, 'calc_missing', 0) or 0) > 0:
                    _record_anomaly('calc_missing', battle, f'{tel.calc_missing}/{tel.calc_requests}')
                if int(getattr(tel, 'foe_no_moves', 0) or 0) > 0:
                    _record_anomaly('foe_no_moves', battle, str(tel.foe_no_moves))
            self.last_joint = int(joint)
            a1, a2 = decode_joint_action(joint)
            order, fb = joint_order_from_spec_actions(battle, (a1, a2))
            if fb:
                st['adapter_fallbacks'] += 1
                _record_anomaly('fallback', battle, f'{a1},{a2}')
            return order
        except Exception as exc:
            st['exceptions'] += 1
            _record_anomaly('exception', battle, repr(exc)[:200])
            if st['exceptions'] <= 3:
                _LOG.exception('mcts_puct_nash: excepción en choose_move → orden por defecto')
            return DefaultBattleOrder()


def make_validated_mcts_puct_nash_player_class():
    from showdown.inference_sim.validated_greedy_player import ValidatedBattleMixin

    class ValidatedMCTSPuctNashPlayer(ValidatedBattleMixin, MCTSPuctNashPlayer):
        """MCTS–PUCT/Nash + espejo IAP / sync / OTS.

        Sin inferencia de SP (22-11-2026): como G (diagnóstico de G §9.1), la inferencia no estrecha
        ningún stat en Reg M-C; los 6 stats del rival al reparto más frecuente del prior meta.
        """

        default_sp_info_mode = 'meta'

    return ValidatedMCTSPuctNashPlayer
