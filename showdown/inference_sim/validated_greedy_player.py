"""Greedy poke-env players + espejo IAP Battle + contraste vs PS local.

ValidatedGreedyPlayer      → base greedy_solo
ValidatedGreedyJointPlayer → base greedy_joint

Ambos comparten ValidatedBattleMixin (sync + soundness OTS).
"""
from __future__ import annotations

import asyncio
from collections import Counter
from typing import List, Optional, Type

from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from showdown.battle_bots.expectimax_bayesian.pokeenv_player import ExpectimaxBayesianPlayer
from showdown.battle_bots.greedy_joint.pokeenv_player import GreedyJointPlayer
from showdown.battle_bots.greedy_solo.pokeenv_player import GreedyDamagePlayer
from showdown.inference_sim.battle_state_check import (
    StateCheckReport,
    StateMismatch,
    contrast_battle,
    record_contrasts,
)
from showdown.inference_sim.chat_quiet import chat_print
from showdown.inference_sim.feeder import bootstrap_battle, feed_line
from showdown.inference_sim.paths import BATTLE_FORMAT
from showdown.inference_sim.soundness import (
    SoundnessReport,
    check_soundness,
    record_timeline,
    snapshot_beliefs,
)
from showdown.inference_sim.team_factory import MonSet, TeamMeta, to_showdown_export
from showdown.stat_inference import reset_sp_limits

# Sync estable en |turn|; inferencia también en daño/switch
_SYNC_ACTIONS = frozenset({'turn'})
_INFER_ACTIONS = frozenset({
    'turn', '-damage', 'move', 'switch', 'drag', 'faint',
})


def _is_ps_battle_room(room: str) -> bool:
    """Solo salas battle-* (no game-bestof padre ni lobby)."""
    tag = (room or '').lstrip('>').strip()
    return tag.startswith('battle-')


PLAYER_A_CHOICES = (
    'greedy_solo', 'greedy_joint', 'expectimax_bayesian', 'ddqn', 'ppo',
    'mcts_puct', 'mcts_puct_nash',
)
PLAYER_B_CHOICES = (
    'random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian', 'ddqn', 'ppo',
    'mcts_puct', 'mcts_puct_nash',
)


class ValidatedBattleMixin:
    """
    Feed del protocolo a Battle IAP + contraste con poke-env.
    Opcionalmente valida soundness de Stat Points vs OTS (ground truth).
    """

    def __init__(
        self,
        *,
        user_team: TeamMeta,
        opp_team: TeamMeta,
        state_report: StateCheckReport,
        soundness_report: Optional[SoundnessReport] = None,
        account_configuration: Optional[AccountConfiguration] = None,
        battle_format: str = BATTLE_FORMAT,
        **kwargs,
    ):
        self.user_team = user_team
        self.opp_team = opp_team
        self.state_report = state_report
        self.soundness_report = soundness_report
        self.iap_battle = None
        self.protocol_log: List[str] = []
        self._bootstrapped = False
        self._our_role = 'p1'
        self._role_confirmed = False
        self._ready_to_check = False
        self._pe_battle: Optional[AbstractBattle] = None
        # Lineas del game actual (para replay si el rol p1/p2 se corrige tarde).
        self._game_lines: List[str] = []
        self._pending_lines: List[str] = []
        # Líneas ya aplicadas vía flush de pending; evita doble feed del re-envío
        # del cliente. Counter (no set): dos |-unboost| idénticos (2× Intimidate)
        # deben poder aplicarse tras el flush sin que el 2º live se pierda.
        self._recently_flushed: Counter = Counter()
        # Compact BO3: [(turns, won), ...] de cada game de la serie en curso
        self.series_games: List[tuple] = []
        self.series_rewards: List = []
        self.all_rewards: List = []
        self._reward_acc = None
        self._logged_game_tags: set = set()
        self._last_turn_log = -1
        kwargs.setdefault('server_configuration', LocalhostServerConfiguration)
        kwargs.setdefault('accept_open_team_sheet', True)
        kwargs.setdefault('max_concurrent_battles', 1)
        kwargs.setdefault('log_level', 30)
        super().__init__(
            account_configuration=account_configuration,
            battle_format=battle_format,
            team=to_showdown_export(user_team),
            **kwargs,
        )
        # Champions OTS público → expectimax (sin SP). No-op si el player no lo soporta.
        try:
            from showdown.battle_bots.expectimax_bayesian.ots import ots_sheet_from_team_meta
            if hasattr(self, 'set_opp_ots_sheet'):
                self.set_opp_ots_sheet(ots_sheet_from_team_meta(opp_team))
        except Exception:
            pass

    def reset_for_next_battle(self, user_team: TeamMeta, opp_team: TeamMeta) -> None:
        self.user_team = user_team
        self.opp_team = opp_team
        try:
            from showdown.battle_bots.expectimax_bayesian.ots import ots_sheet_from_team_meta
            if hasattr(self, 'set_opp_ots_sheet'):
                self.set_opp_ots_sheet(ots_sheet_from_team_meta(opp_team))
        except Exception:
            pass
        self.iap_battle = None
        self._bootstrapped = False
        self.protocol_log = []
        # No forzar p1: en BO3/challenges el rol se conserva; pisarlo aplica
        # |switch| al bando equivocado hasta el próximo |player|.
        self._ready_to_check = False
        self._pe_battle = None
        self._pe_battle_tag = ''
        self._game_lines = []
        self._pending_lines = []
        self._recently_flushed = Counter()
        self.series_games = []
        self.series_rewards = []
        self._reward_acc = None
        self._logged_game_tags = set()
        self._orphan_forfeit_tags = set()
        self._last_turn_log = -1
        self.update_team(to_showdown_export(user_team))
        # Crítico: drenar cola BO3 de poke-env (si un timeout anterior no vio |win|,
        # battle_against se cuelga en join() aunque el BO3 nuevo termine 3-0).
        try:
            from showdown.inference_sim.bo3_runtime import reset_bo3_runtime_state
            reset_bo3_runtime_state(self)
        except Exception:
            pass

    def choose_move(self, battle: AbstractBattle):
        """Si el BO3 ya está decidido, forfeit de games huérfanos (evita g4/g5)."""
        try:
            from showdown.inference_sim.bo3_runtime import bo3_series_decided
            from poke_env.player.battle_order import (
                DefaultBattleOrder,
                DoubleBattleOrder,
            )

            games = list(getattr(self, 'series_games', None) or [])
            if bo3_series_decided(games):
                tag = (getattr(battle, 'battle_tag', None) or '').lstrip('>')
                sent = getattr(self, '_orphan_forfeit_tags', None)
                if sent is None:
                    sent = set()
                    self._orphan_forfeit_tags = sent
                if tag and tag not in sent:
                    sent.add(tag)
                    try:
                        loop = asyncio.get_running_loop()
                        # PS exige /forfeit en la room, no en consola.
                        loop.create_task(
                            self.ps_client.send_message('/forfeit', room=tag)
                        )
                    except Exception:
                        pass
                return DoubleBattleOrder(DefaultBattleOrder(), DefaultBattleOrder())
        except Exception:
            pass
        return super().choose_move(battle)  # type: ignore[misc]

    def _preview_ordered_team(self) -> TeamMeta:
        """Reordena el TeamMeta según last_preview_order (leads = primeros 2)."""
        order = (getattr(self, 'last_preview_order', None) or '1234').strip()
        digits = [int(c) for c in order if c.isdigit()]
        if len(digits) < 2:
            return self.user_team
        mons: List[MonSet] = list(self.user_team.mons)
        picked: List[MonSet] = []
        for d in digits:
            idx = d - 1
            if 0 <= idx < len(mons):
                picked.append(mons[idx])
        for m in mons:
            if m not in picked:
                picked.append(m)
        return TeamMeta(
            name=self.user_team.name,
            source_url=getattr(self.user_team, 'source_url', '') or '',
            mons=picked,
        )

    async def _handle_battle_message(self, split_messages: List[List[str]]):
        room = split_messages[0][0] if split_messages else ''
        pending_sync = False
        pending_infer = False
        is_battle_room = _is_ps_battle_room(room)

        # Antes de alimentar: si poke-env ya cambió de battle-*, resetear espejo.
        if is_battle_room:
            pe_early = self._current_pe_battle(room)
            if pe_early is not None:
                pe_tag = (
                    getattr(pe_early, 'battle_tag', None) or room or ''
                ).lstrip('>').strip()
                if pe_tag and pe_tag != getattr(self, '_pe_battle_tag', ''):
                    prev = getattr(self, '_pe_battle_tag', '')
                    self._pe_battle_tag = pe_tag
                    if prev:
                        self.iap_battle = None
                        self._game_lines = []
                        self._pending_lines = []
                        self._recently_flushed = Counter()
                        self._ready_to_check = False
                        self._last_turn_log = -1

        for parts in split_messages[1:]:
            if not parts:
                continue
            line = '|'.join(parts)
            if not line.startswith('|'):
                line = '|' + line
            if len(self.protocol_log) < 400:
                self.protocol_log.append(line)
            elif len(self.protocol_log) == 400:
                self.protocol_log.append('# protocol_log truncated')

            if len(parts) > 2 and parts[1] == 'player':
                pid = parts[2].strip()
                nick = parts[3].strip() if len(parts) > 3 else ''
                if nick and nick == self.username and pid in ('p1', 'p2'):
                    old_role = self._our_role
                    self._our_role = pid
                    self._role_confirmed = True
                    if (
                        self.iap_battle is not None
                        and old_role != pid
                        and self._game_lines
                    ):
                        self._rebuild_iap_from_game_lines(room)
                    elif self.iap_battle is None and self._pending_lines:
                        # |start| llegó antes que |player|: crear espejo y vaciar buffer.
                        self._ensure_iap(room)
                    else:
                        self._sync_role_ids()

            action = parts[1].strip() if len(parts) > 1 else ''

            # Nueva partida del BO3: reiniciar espejo IAP en |start| (no en cada
            # |init| suelto: vaciar sin bootstrap deja el espejo a medias).
            if is_battle_room and action == 'start':
                self.iap_battle = None
                self._ready_to_check = False
                self._last_turn_log = -1
                self._game_lines = []
                self._pending_lines = []
                self._recently_flushed = Counter()
                self._pe_battle_tag = (room or '').lstrip('>').strip()
                self._ensure_iap(room)

            # Contador de flush solo cubre el re-envío inmediato post-bootstrap.
            # Si sobra (línea idéntica no reenviada), envenena turnos futuros
            # (mismo |-unboost|…|spa|1| cada turno → pe spa siempre -1 vs iap).
            if is_battle_room and action in ('turn', 'request') and self._recently_flushed:
                self._recently_flushed = Counter()

            # No alimentar IAP con mensajes de la sala padre game-bestof*.
            if not is_battle_room:
                continue

            if self.iap_battle is None:
                self._ensure_iap(room)
            if self.iap_battle is None:
                # Sin rol p1/p2 aún: bufferizar para no aplicar |switch| al bando malo.
                if action not in ('player', 'start'):
                    self._pending_lines.append(line)
                continue

            prev_iap_uid = ''
            if self.iap_battle is not None:
                prev_iap_uid = str(
                    getattr(getattr(self.iap_battle, 'user', None), 'id', '') or ''
                ).strip().lower()
            # Pending del bootstrap ya se aplicó en _flush_pending_lines; esas
            # líneas no vuelven por el socket (eran batches previos). No usar
            # Counter de skip: leftovers envenenaban |-unboost|/|-fieldstart|
            # idénticos en turnos posteriores.
            try:
                fed = feed_line(self.iap_battle, line)
                if not line.startswith('#'):
                    self._game_lines.append(line)
            except Exception as exc:
                self.state_report.feed_errors += 1
                # Conservar la línea para rebuild aunque el handler falle
                # (p.ej. |-fieldstart| Grassy Terrain).
                if not line.startswith('#'):
                    self._game_lines.append(line)
                self.state_report.add(
                    StateMismatch(
                        battle_tag=room,
                        turn=int(getattr(self.iap_battle, 'turn', 0) or 0),
                        field='feed',
                        expected='ok',
                        got=repr(exc),
                        detail=line[:160],
                    )
                )
                fed = None
            # |request| trae side.id: si el rol p1/p2 estaba mal (o el espejo
            # acababa de flippear bandos), reconstruir desde el log.
            if (fed == 'request' or action == 'request') and self.iap_battle is not None:
                req_role = str(getattr(self.iap_battle.user, 'id', '') or '').strip().lower()
                flipped = bool(getattr(self.iap_battle, '_request_side_flipped', False))
                if flipped:
                    try:
                        delattr(self.iap_battle, '_request_side_flipped')
                    except Exception:
                        self.iap_battle._request_side_flipped = False
                need_rebuild = False
                if req_role in ('p1', 'p2'):
                    if req_role != self._our_role:
                        need_rebuild = True
                    elif (
                        prev_iap_uid in ('p1', 'p2')
                        and prev_iap_uid != req_role
                    ):
                        need_rebuild = True
                    elif flipped:
                        need_rebuild = True
                    if need_rebuild:
                        self._our_role = req_role
                        self._role_confirmed = True
                        if self._game_lines:
                            self._rebuild_iap_from_game_lines(room)
            if action == 'switch' or fed == 'switch':
                self._ready_to_check = True
            key = fed or action
            if key in _SYNC_ACTIONS:
                pending_sync = True
                try:
                    turn_n = int(getattr(self.iap_battle, 'turn', 0) or 0)
                except Exception:
                    turn_n = 0
                if turn_n > 0 and turn_n != getattr(self, '_last_turn_log', -1):
                    self._last_turn_log = turn_n
                    g = len(getattr(self, 'series_games', []) or []) + 1
                    chat_print(f'  g{g}:t{turn_n}')
            if key in _INFER_ACTIONS:
                pending_infer = True

        try:
            await super()._handle_battle_message(split_messages)
        except KeyError as exc:
            if len(self.protocol_log) < 420:
                self.protocol_log.append(f'# poke-env KeyError skipped: {exc!r}')
            return
        except Exception as exc:
            name = type(exc).__name__
            msg = str(exc).lower()
            if name in ('ConnectionClosedOK', 'ConnectionClosedError', 'ConnectionClosed') or 'close' in msg:
                return
            if len(self.protocol_log) < 420:
                self.protocol_log.append(f'# poke-env handler skipped: {exc!r}')
            return

        if not is_battle_room:
            return

        pe = self._current_pe_battle(room)
        if pe is not None:
            self._pe_battle = pe
            pe_tag = (getattr(pe, 'battle_tag', None) or room or '').lstrip('>').strip()
            # Nueva hija battle-* del BO3: no reutilizar espejo del juego anterior
            # (síntoma: pe.turn=6 iap.turn=1 + fields/species fantasma).
            if pe_tag and pe_tag != getattr(self, '_pe_battle_tag', ''):
                prev = getattr(self, '_pe_battle_tag', '')
                self._pe_battle_tag = pe_tag
                self._reset_reward_acc(pe_tag)
                if prev and self.iap_battle is not None:
                    self.iap_battle = None
                    self._game_lines = []
                    self._pending_lines = []
                    self._recently_flushed = Counter()
                    self._ready_to_check = False
                    self._last_turn_log = -1
                    self._ensure_iap(room)
            pe_role = getattr(pe, 'player_role', None)
            if pe_role in ('p1', 'p2') and pe_role != self._our_role:
                self._our_role = pe_role
                self._role_confirmed = True
                if self._game_lines or self._pending_lines:
                    self._rebuild_iap_from_game_lines(room)
                else:
                    self._ensure_iap(room)
                    self._flush_pending_lines()

        self._sync_role_ids()

        # Ancla campo/boosts a PE (anti-péndulo sync E2E). Tras rebuild/feed.
        if self.iap_battle is not None and pe is not None:
            try:
                from showdown.inference_sim.battle_state_check import (
                    mirror_pe_field_and_boosts_into_iap,
                )
                mirror_pe_field_and_boosts_into_iap(self.iap_battle, pe)
            except Exception:
                pass

        if (
            self._ready_to_check
            and self.iap_battle is not None
            and pe is not None
            and not getattr(pe, 'finished', False)
        ):
            if pending_sync:
                self._contrast(room, pe)
            if pending_infer and self.soundness_report is not None:
                self._check_inference(room)

        if pe is not None:
            self._observe_reward(pe)
        self._note_finished_game(room, pe)

    def _reset_reward_acc(self, tag: str = '') -> None:
        from showdown.inference_sim.reward_telemetry import BattleRewardAccumulator

        self._reward_acc = BattleRewardAccumulator()
        self._reward_acc.reset(tag=tag)

    def _observe_reward(self, pe: AbstractBattle) -> None:
        try:
            if self._reward_acc is None:
                tag = str(getattr(pe, 'battle_tag', '') or '')
                self._reset_reward_acc(tag)
            assert self._reward_acc is not None
            self._reward_acc.observe(pe)
        except Exception:
            pass

    def _note_finished_game(self, room: str, pe: Optional[AbstractBattle]) -> None:
        """Registra gN:tT W/L al cerrar cada partida del BO3."""
        if pe is None or not getattr(pe, 'finished', False):
            return
        tag = (getattr(pe, 'battle_tag', None) or room or '').lstrip('>').strip()
        if not tag or tag in self._logged_game_tags:
            return
        # Solo batallas hijas; la sala padre es game-bestof*
        if not tag.startswith('battle-'):
            return
        # BO3 ya decidido: no registrar g4/g5 fantasma ni rewards extra.
        try:
            from showdown.inference_sim.bo3_runtime import bo3_series_decided
            if bo3_series_decided(list(getattr(self, 'series_games', None) or [])):
                return
        except Exception:
            pass
        self._logged_game_tags.add(tag)
        turns = int(getattr(pe, 'turn', 0) or 0)
        won = bool(getattr(pe, 'won', False))
        self.series_games.append((turns, won))
        try:
            if self._reward_acc is None:
                self._reset_reward_acc(tag)
            assert self._reward_acc is not None
            summary = self._reward_acc.finalize(pe)
            if not hasattr(self, 'series_rewards') or self.series_rewards is None:
                self.series_rewards = []
            self.series_rewards.append(summary)
            # Batch-wide list (survives reset_for_next_battle series clear)
            if not hasattr(self, 'all_rewards') or self.all_rewards is None:
                self.all_rewards = []
            self.all_rewards.append(summary)
        except Exception:
            pass
        try:
            flush = getattr(self, 'flush_battle_outcome', None)
            if callable(flush):
                flush(won)
        except Exception:
            pass
        self._reward_acc = None
        g = len(self.series_games)
        chat_print(f'  g{g}:t{turns}{"W" if won else "L"}')

    def series_games_compact(self) -> str:
        if not self.series_games:
            return '-'
        rewards = list(getattr(self, 'series_rewards', None) or [])
        parts = []
        for i, (turns, won) in enumerate(self.series_games, start=1):
            base = f'g{i}:t{turns}{"W" if won else "L"}'
            if i - 1 < len(rewards):
                r = rewards[i - 1]
                base += f' c2={r.c2_return:+.2f}'
            parts.append(base)
        return ' '.join(parts)

    def _current_pe_battle(self, room: str) -> Optional[AbstractBattle]:
        battles = getattr(self, 'battles', None) or {}
        if not battles:
            return None
        tag = (room or '').lstrip('>').strip()
        if tag and tag in battles:
            return battles[tag]
        for key, bat in battles.items():
            k = str(key).lstrip('>')
            if tag and (k == tag or tag.endswith(k) or k.endswith(tag)):
                return bat
        for bat in battles.values():
            if not getattr(bat, 'finished', False):
                return bat
        return None

    def _ensure_iap(self, room: str) -> None:
        if self.iap_battle is not None:
            return
        pe = self._current_pe_battle(room) or self._pe_battle
        pe_role = getattr(pe, 'player_role', None) if pe is not None else None
        if pe_role in ('p1', 'p2'):
            self._our_role = pe_role
            self._role_confirmed = True
        if not self._role_confirmed or self._our_role not in ('p1', 'p2'):
            return
        reset_sp_limits()
        ordered = self._preview_ordered_team()
        self.iap_battle = bootstrap_battle(ordered, self.opp_team)
        self._sync_role_ids()
        self._bootstrapped = True
        if len(self.protocol_log) < 420:
            self.protocol_log.append(
                f'# iap bootstrap room={room} role={self._our_role} '
                f'preview={getattr(self, "last_preview_order", "")}'
            )
        self._flush_pending_lines()

    def _flush_pending_lines(self) -> None:
        if self.iap_battle is None or not self._pending_lines:
            return
        pending = list(self._pending_lines)
        self._pending_lines = []
        self._recently_flushed = Counter()
        for line in pending:
            try:
                feed_line(self.iap_battle, line)
                if not line.startswith('#'):
                    self._game_lines.append(line)
            except Exception as exc:
                self.state_report.feed_errors += 1
                if len(self.protocol_log) < 420:
                    self.protocol_log.append(f'# pending feed error: {exc!r}')

    def _rebuild_iap_from_game_lines(self, room: str) -> None:
        """Reconstruye el espejo IAP con el rol correcto y re-aplica el game."""
        lines = [ln for ln in self._game_lines if not ln.startswith('#')]
        pending = list(self._pending_lines)
        self.iap_battle = None
        self._bootstrapped = False
        self._game_lines = []
        self._pending_lines = []
        self._recently_flushed = Counter()
        self._ready_to_check = False
        self._ensure_iap(room)
        if self.iap_battle is None:
            self._pending_lines = lines + pending
            return
        for line in lines + pending:
            try:
                fed = feed_line(self.iap_battle, line)
                self._game_lines.append(line)
                if fed == 'switch' or '|switch|' in line or line.startswith('|switch|'):
                    self._ready_to_check = True
            except Exception as exc:
                self.state_report.feed_errors += 1
                if len(self.protocol_log) < 420:
                    self.protocol_log.append(f'# replay feed error: {exc!r}')
        if len(self.protocol_log) < 420:
            self.protocol_log.append(
                f'# iap rebuild room={room} role={self._our_role} lines={len(self._game_lines)}'
            )

    def _sync_role_ids(self) -> None:
        if self.iap_battle is None:
            return
        self.iap_battle.user.id = self._our_role
        self.iap_battle.opponent.id = 'p2' if self._our_role == 'p1' else 'p1'

    def _contrast(self, room: str, pe_battle: AbstractBattle) -> None:
        if not isinstance(pe_battle, DoubleBattle):
            return
        try:
            pe_turn = int(getattr(pe_battle, 'turn', 0) or 0)
            iap_turn = int(getattr(self.iap_battle, 'turn', 0) or 0) if self.iap_battle else 0
            # Espejo de otra partida / rebuild a medias: rehacer desde el log.
            if (
                pe_turn >= 2
                and iap_turn >= 1
                and abs(pe_turn - iap_turn) > 1
                and self._game_lines
            ):
                self._rebuild_iap_from_game_lines(room)
                self._sync_role_ids()
            elif self._severe_species_desync(pe_battle) and self._game_lines:
                self._rebuild_iap_from_game_lines(room)
                self._sync_role_ids()
            try:
                from showdown.inference_sim.battle_state_check import (
                    mirror_pe_field_and_boosts_into_iap,
                )
                mirror_pe_field_and_boosts_into_iap(self.iap_battle, pe_battle)
            except Exception:
                pass
            found = contrast_battle(self.iap_battle, pe_battle, battle_tag=room)
            record_contrasts(self.state_report, found)
        except Exception as exc:
            self.state_report.feed_errors += 1
            if len(self.protocol_log) < 420:
                self.protocol_log.append(f'# contrast error: {exc!r}')

    def _severe_species_desync(self, pe_battle: AbstractBattle) -> bool:
        """True si our/foe vivos no coinciden en especies (no solo sets disjuntos)."""
        if self.iap_battle is None:
            return False
        try:
            from showdown.inference_sim.battle_state_check import (
                _actives_iap,
                _actives_pe,
                _canon,
                _species_iap,
                _species_pe,
            )

            def _alive_sets(pe_mons, iap_mons):
                pe_set = {
                    _canon(_species_pe(m))
                    for m in pe_mons
                    if m is not None and not getattr(m, 'fainted', False) and _species_pe(m)
                }
                iap_set = {
                    _canon(_species_iap(m))
                    for m in iap_mons
                    if m is not None
                    and not getattr(m, 'is_fainted', False)
                    and _species_iap(m)
                }
                return pe_set, iap_set

            pe_ours, pe_foes = _actives_pe(pe_battle)
            iap_ours, iap_foes = _actives_iap(self.iap_battle)
            for pe_set, iap_set in (
                _alive_sets(pe_ours, iap_ours),
                _alive_sets(pe_foes, iap_foes),
            ):
                if not pe_set or not iap_set:
                    continue
                # Cualquier especie de PE ausente en IAP (o al revés) = desync.
                if pe_set != iap_set:
                    return True
            return False
        except Exception:
            return False

    def _severe_our_species_desync(self, pe_battle: AbstractBattle) -> bool:
        """Compat: alias de desync our-side."""
        return self._severe_species_desync(pe_battle)

    def _check_inference(self, room: str) -> None:
        if self.soundness_report is None or self.iap_battle is None:
            return
        try:
            before = self.soundness_report.failure_count
            check_soundness(
                self.iap_battle,
                self.opp_team,
                battle_tag=room,
                recent_lines=self.protocol_log,
                report=self.soundness_report,
            )
            turn_n = int(getattr(self.iap_battle, 'turn', 0) or 0)
            snaps = snapshot_beliefs(
                self.iap_battle,
                self.opp_team,
                turn=turn_n,
                event='turn',
            )
            for sid, snap in snaps.items():
                hist = self.soundness_report.timelines.get(sid) or []
                if hist:
                    prev = hist[-1]
                    for stat, rng in (snap.get('ranges') or {}).items():
                        prev_n = (prev.get('ranges') or {}).get(stat, {}).get('n')
                        cur_n = rng.get('n')
                        if prev_n and cur_n and cur_n < prev_n:
                            self.soundness_report.narrowing_events += 1
            record_timeline(self.soundness_report, snaps)
            _ = before
        except Exception as exc:
            self.soundness_report.battle_errors += 1
            if len(self.protocol_log) < 420:
                self.protocol_log.append(f'# soundness error: {exc!r}')


class ValidatedGreedyPlayer(ValidatedBattleMixin, GreedyDamagePlayer):
    """Greedy-Solo + espejo IAP / sync / OTS."""


class ValidatedGreedyJointPlayer(ValidatedBattleMixin, GreedyJointPlayer):
    """Greedy-Joint + espejo IAP / sync / OTS."""


class ValidatedExpectimaxBayesianPlayer(ValidatedBattleMixin, ExpectimaxBayesianPlayer):
    """Expectimax-Bayesian + espejo IAP / sync / OTS."""


def _ddqn_player_classes():
    from showdown.rl.ddqn.player import DDQNPlayer, make_validated_ddqn_player_class
    return DDQNPlayer, make_validated_ddqn_player_class()


def _ppo_player_classes():
    from showdown.rl.ppo.player import PPOPlayer, make_validated_ppo_player_class
    return PPOPlayer, make_validated_ppo_player_class()


def _mcts_puct_player_classes():
    from showdown.rl.mcts_puct.player import (
        MCTSPuctPlayer,
        make_validated_mcts_puct_player_class,
    )
    return MCTSPuctPlayer, make_validated_mcts_puct_player_class()


def _mcts_puct_nash_player_classes():
    from showdown.rl.mcts_puct_nash.player import (
        MCTSPuctNashPlayer,
        make_validated_mcts_puct_nash_player_class,
    )
    return MCTSPuctNashPlayer, make_validated_mcts_puct_nash_player_class()


def validated_player_class(kind: str) -> Type[Player]:
    """Player A del robusto: siempre validated (sync + OTS)."""
    key = str(kind or 'greedy_solo').strip().lower().replace('-', '_')
    if key == 'greedy_solo':
        return ValidatedGreedyPlayer
    if key == 'greedy_joint':
        return ValidatedGreedyJointPlayer
    if key == 'expectimax_bayesian':
        return ValidatedExpectimaxBayesianPlayer
    if key == 'ddqn':
        _, ValidatedDDQNPlayer = _ddqn_player_classes()
        return ValidatedDDQNPlayer
    if key == 'ppo':
        _, ValidatedPPOPlayer = _ppo_player_classes()
        return ValidatedPPOPlayer
    if key == 'mcts_puct':
        _, ValidatedMCTSPuctPlayer = _mcts_puct_player_classes()
        return ValidatedMCTSPuctPlayer
    if key == 'mcts_puct_nash':
        _, ValidatedMCTSPuctNashPlayer = _mcts_puct_nash_player_classes()
        return ValidatedMCTSPuctNashPlayer
    raise ValueError(
        f'player-a inválido: {kind!r}. Opciones: {", ".join(PLAYER_A_CHOICES)}'
    )


def opponent_player_class(kind: str) -> Type[Player]:
    """Player B: sin wrapper validated (puede ser random o greedy)."""
    from poke_env.player import RandomPlayer

    key = str(kind or 'random').strip().lower().replace('-', '_')
    if key == 'random':
        return RandomPlayer
    if key == 'greedy_solo':
        return GreedyDamagePlayer
    if key == 'greedy_joint':
        return GreedyJointPlayer
    if key == 'expectimax_bayesian':
        return ExpectimaxBayesianPlayer
    if key == 'ddqn':
        DDQNPlayer, _ = _ddqn_player_classes()
        return DDQNPlayer
    if key == 'ppo':
        PPOPlayer, _ = _ppo_player_classes()
        return PPOPlayer
    if key == 'mcts_puct':
        MCTSPuctPlayer, _ = _mcts_puct_player_classes()
        return MCTSPuctPlayer
    if key == 'mcts_puct_nash':
        MCTSPuctNashPlayer, _ = _mcts_puct_nash_player_classes()
        return MCTSPuctNashPlayer
    raise ValueError(
        f'player-b inválido: {kind!r}. Opciones: {", ".join(PLAYER_B_CHOICES)}'
    )


def make_opponent_player(
    kind: str,
    *,
    team: TeamMeta,
    account_configuration: AccountConfiguration,
    battle_format: str,
    server_configuration=LocalhostServerConfiguration,
) -> Player:
    cls = opponent_player_class(kind)
    return cls(
        account_configuration=account_configuration,
        battle_format=battle_format,
        team=to_showdown_export(team),
        server_configuration=server_configuration,
        accept_open_team_sheet=True,
        max_concurrent_battles=1,
        log_level=30,
    )
