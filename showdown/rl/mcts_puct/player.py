"""poke-env MCTS+PUCT player + Validated factory."""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from showdown.rl.action_space import decode_joint_action
from showdown.rl.family_preview_mixin import FamilyPreviewMixin
from showdown.rl.mcts_puct.agent import MCTSPuctAgent, MCTSPuctConfig
from showdown.rl.pe_adapter import (
    joint_order_from_spec_actions,
    legal_joint_mask as pe_legal_joint_mask,
    snapshot_from_battle,
)

_LOG = logging.getLogger(__name__)


def _record_anomaly(kind: str, battle, detail: str = '') -> None:
    """Una línea por excepción/fallback en ``MCTS_STATS_FILE`` (si está definida).

    Los avisos de logging salen de los subprocesos de run_robust_e2e y pueden perderse;
    el archivo permite a la cola del servidor contarlos de forma fiable.
    """
    path = os.environ.get('MCTS_STATS_FILE', '').strip()
    if not path:
        return
    try:
        tag = str(getattr(battle, 'battle_tag', '?'))
        turn = int(getattr(battle, 'turn', 0) or 0)
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(f'{kind}\t{tag}\t{turn}\t{detail}\n')
    except Exception:
        pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


class MCTSPuctPlayer(FamilyPreviewMixin, Player):
    """Frozen ``team_preview_nash`` + MCTS+PUCT joint policy.

    Env: ``MCTS_SIMS`` (presupuesto), ``MCTS_SEED``, ``MCTS_PUCT_VERSION`` (3 por
    defecto = S3; 2 = S2 open-loop; 1 = búsqueda original). Contadores (``mcts_stats``): decisiones,
    excepciones (→ orden por defecto) y fallbacks del adaptador (→ orden al azar);
    antes eran silenciosos.
    """

    _combat_family_id = 'mcts_puct'

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
        sims = n_sims if n_sims is not None else _env_int('MCTS_SIMS', 200)
        self.agent = MCTSPuctAgent(
            MCTSPuctConfig(
                n_sims=int(sims),
                seed=_env_int('MCTS_SEED', seed),
                version=_env_int('MCTS_PUCT_VERSION', 3),
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
            preview_checkpoint=preview_checkpoint,
            preview_kind=preview_kind,
        )

    def _root_joints(self, battle):
        try:
            mask = pe_legal_joint_mask(battle)
            return [int(j) for j in np.flatnonzero(mask > 0.5)]
        except Exception:
            return None

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)
        st = self.mcts_stats
        st['decisions'] += 1
        try:
            snap = snapshot_from_battle(battle)
            root = self._root_joints(battle) if int(self.agent.cfg.version) >= 2 else None
            beliefs = getattr(getattr(self, 'iap_battle', None), 'stat_beliefs', None)
            joint = self.agent.act(snap, root_joints=root, stat_beliefs=beliefs)
            tel = self.agent.last_telemetry
            if tel is not None:
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
                    _record_anomaly('calc_missing', battle,
                                    f'{tel.calc_missing}/{tel.calc_requests}')
                if int(getattr(tel, 'foe_no_moves', 0) or 0) > 0:
                    _record_anomaly('foe_no_moves', battle, str(tel.foe_no_moves))
            self.last_joint = int(joint)
            a1, a2 = decode_joint_action(joint)
            order, fb = joint_order_from_spec_actions(battle, (a1, a2))
            if fb:
                st['adapter_fallbacks'] += 1
                _record_anomaly('fallback', battle, f'{a1},{a2}')
                if st['adapter_fallbacks'] <= 3:
                    _LOG.warning('mcts_puct: orden ilegal %s en %s → jugada al azar',
                                 (a1, a2), getattr(battle, 'battle_tag', '?'))
            return order
        except Exception as exc:
            st['exceptions'] += 1
            _record_anomaly('exception', battle, repr(exc)[:200])
            if st['exceptions'] <= 3:
                _LOG.exception('mcts_puct: excepción en choose_move → orden por defecto')
            return DefaultBattleOrder()


def make_validated_mcts_puct_player_class():
    from showdown.inference_sim.validated_greedy_player import ValidatedBattleMixin

    class ValidatedMCTSPuctPlayer(ValidatedBattleMixin, MCTSPuctPlayer):
        """MCTS+PUCT + IAP mirror sync / OTS."""

    return ValidatedMCTSPuctPlayer
