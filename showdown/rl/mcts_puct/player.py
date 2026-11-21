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
    """Preview greedy (tipado + BST) + MCTS+PUCT joint policy (spec v3.1).

    Preview (decisión del usuario, 20-11-2026): el mismo de solo/joint/expectimax.
    ``team_preview_nash`` nunca se entrenó (solo un checkpoint de smoke con etiquetas
    mock), así que la familia caía al orden fijo ``/team 1234`` (−4 pts medidos).
    ``MCTS_PREVIEW=nash`` vuelve al preview de familia (requiere checkpoint;
    ``MCTS_PREVIEW_CKPT`` elige cuál). Si el checkpoint no carga y se cae al orden fijo,
    se registra la anomalía ``preview_fallback`` (antes pasaba en silencio).

    Env: ``MCTS_SIMS`` (presupuesto), ``MCTS_SEED``, ``MCTS_PUCT_VERSION`` (5 por
    defecto = S3 fase 3c, confirmada; 6 = 3d; 4 = 3b; 3 = S3 fase 3a; 2 = S2 open-loop; 1 = búsqueda
    original). Contadores (``mcts_stats``): decisiones,
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
                version=_env_int('MCTS_PUCT_VERSION', 5),
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

    def teampreview(self, battle) -> str:
        if os.environ.get('MCTS_PREVIEW', 'greedy').strip().lower() == 'nash':
            out = FamilyPreviewMixin.teampreview(self, battle)
            fp = getattr(self, '_frozen_preview', None)
            if fp is not None and not fp.available:
                _record_anomaly('preview_fallback', battle, str(getattr(fp, '_load_error', '')))
            return out
        if os.environ.get('MCTS_PREVIEW', 'greedy').strip().lower() == 'search':
            return self._search_preview(battle)
        from showdown.battle_bots.greedy_solo.pokeenv_player import GreedyDamagePlayer
        return GreedyDamagePlayer.teampreview(self, battle)

    def _search_preview(self, battle) -> str:
        """Preview por búsqueda con S3 (``search_preview``); cae al greedy si falla."""
        from showdown.battle_bots.greedy_solo.pokeenv_player import GreedyDamagePlayer
        from showdown.rl.mcts_puct.search_preview import SearchPreviewConfig, choose_formation
        from showdown.rl.pe_adapter import mon_from_pe
        try:
            our_src = list(getattr(battle, 'teampreview_team', None) or list(battle.team.values()))
            foe_src = list(getattr(battle, 'teampreview_opponent_team', None)
                           or list(battle.opponent_team.values()))
            ours = [mon_from_pe(m, is_active=False, own=True) for m in our_src]
            foes = [mon_from_pe(m, is_active=False, own=False) for m in foe_src]
            ver = int(self.agent.cfg.version)
            cfg = SearchPreviewConfig(
                decision=os.environ.get('MCTS_PREVIEW_DECISION', 'mean').strip().lower(),
                phase=os.environ.get('MCTS_PREVIEW_PHASE', {6: '3d', 5: '3c', 4: '3b'}.get(ver, '3c')),
                seed=int(self.agent.cfg.seed),
            )
            beliefs = getattr(getattr(self, 'iap_battle', None), 'stat_beliefs', None)
            order, info = choose_formation(ours, foes, cfg, stat_beliefs=beliefs)
            self.mcts_stats['preview_sec'] = self.mcts_stats.get('preview_sec', 0.0) + info.elapsed_sec
            if info.fallback:
                _record_anomaly('preview_search_fallback', battle, info.fallback)
            if any(not m.moves for m in foes):
                _record_anomaly('preview_foe_no_moves', battle, str(sum(1 for m in foes if not m.moves)))
            selected = set(order)
            for i, mon in enumerate(getattr(battle, 'teampreview_team', None) or [], start=1):
                try:
                    mon._selected_in_teampreview = i in selected  # requisito de poke-env
                except Exception:
                    pass
            self.last_preview_order = ''.join(str(int(i)) for i in order)
            return f'/team {self.last_preview_order}'
        except Exception as exc:
            _record_anomaly('preview_search_fallback', battle, repr(exc)[:160])
            return GreedyDamagePlayer.teampreview(self, battle)

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
