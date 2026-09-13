"""Greedy poke-env players + espejo IAP Battle + contraste vs PS local.

ValidatedGreedyPlayer      → base greedy_solo
ValidatedGreedyJointPlayer → base greedy_joint

Ambos comparten ValidatedBattleMixin (sync + soundness OTS).
"""
from __future__ import annotations

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


PLAYER_A_CHOICES = ('greedy_solo', 'greedy_joint', 'expectimax_bayesian')
PLAYER_B_CHOICES = ('random', 'greedy_solo', 'greedy_joint', 'expectimax_bayesian')


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
        # Compact BO3: [(turns, won), ...] de cada game de la serie en curso
        self.series_games: List[tuple] = []
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

    def reset_for_next_battle(self, user_team: TeamMeta, opp_team: TeamMeta) -> None:
        self.user_team = user_team
        self.opp_team = opp_team
        self.iap_battle = None
        self._bootstrapped = False
        self.protocol_log = []
        # No forzar p1: en BO3/challenges el rol se conserva; pisarlo aplica
        # |switch| al bando equivocado hasta el próximo |player|.
        self._ready_to_check = False
        self._pe_battle = None
        self.series_games = []
        self._logged_game_tags = set()
        self._last_turn_log = -1
        self.update_team(to_showdown_export(user_team))

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
                if nick and nick == self.username:
                    self._our_role = pid
                    self._role_confirmed = True
                    self._sync_role_ids()

            action = parts[1].strip() if len(parts) > 1 else ''

            # Nueva partida del BO3: reiniciar espejo IAP en |start| (no en cada
            # |init| suelto: vaciar sin bootstrap deja el espejo a medias).
            if is_battle_room and action == 'start':
                self.iap_battle = None
                self._ready_to_check = False
                self._last_turn_log = -1
                self._ensure_iap(room)

            # No alimentar IAP con mensajes de la sala padre game-bestof*.
            if not is_battle_room:
                continue

            if self.iap_battle is not None:
                try:
                    fed = feed_line(self.iap_battle, line)
                except Exception as exc:
                    self.state_report.feed_errors += 1
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
                        print(f'  g{g}:t{turn_n}', flush=True)
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

        self._sync_role_ids()

        pe = self._current_pe_battle(room)
        if pe is not None:
            self._pe_battle = pe

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

        self._note_finished_game(room, pe)

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
        self._logged_game_tags.add(tag)
        turns = int(getattr(pe, 'turn', 0) or 0)
        won = bool(getattr(pe, 'won', False))
        self.series_games.append((turns, won))
        g = len(self.series_games)
        print(f'  g{g}:t{turns}{"W" if won else "L"}', flush=True)

    def series_games_compact(self) -> str:
        if not self.series_games:
            return '-'
        parts = []
        for i, (turns, won) in enumerate(self.series_games, start=1):
            parts.append(f'g{i}:t{turns}{"W" if won else "L"}')
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
        reset_sp_limits()
        # Preferir el rol que poke-env ya conoce en esta sala.
        pe = self._current_pe_battle(room) or self._pe_battle
        pe_role = getattr(pe, 'player_role', None) if pe is not None else None
        if pe_role in ('p1', 'p2'):
            self._our_role = pe_role
            self._role_confirmed = True
        ordered = self._preview_ordered_team()
        self.iap_battle = bootstrap_battle(ordered, self.opp_team)
        self._sync_role_ids()
        self._bootstrapped = True
        if len(self.protocol_log) < 420:
            self.protocol_log.append(
                f'# iap bootstrap room={room} role={self._our_role} '
                f'preview={getattr(self, "last_preview_order", "")}'
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
            found = contrast_battle(self.iap_battle, pe_battle, battle_tag=room)
            record_contrasts(self.state_report, found)
        except Exception as exc:
            self.state_report.feed_errors += 1
            if len(self.protocol_log) < 420:
                self.protocol_log.append(f'# contrast error: {exc!r}')

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


def validated_player_class(kind: str) -> Type[Player]:
    """Player A del robusto: siempre validated (sync + OTS)."""
    key = str(kind or 'greedy_solo').strip().lower().replace('-', '_')
    if key == 'greedy_solo':
        return ValidatedGreedyPlayer
    if key == 'greedy_joint':
        return ValidatedGreedyJointPlayer
    if key == 'expectimax_bayesian':
        return ValidatedExpectimaxBayesianPlayer
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
    raise ValueError(
        f'player-b inválido: {kind!r}. Opciones: {", ".join(PLAYER_B_CHOICES)}'
    )


def make_opponent_player(
    kind: str,
    *,
    team: TeamMeta,
    account_configuration: AccountConfiguration,
    battle_format: str,
) -> Player:
    cls = opponent_player_class(kind)
    return cls(
        account_configuration=account_configuration,
        battle_format=battle_format,
        team=to_showdown_export(team),
        server_configuration=LocalhostServerConfiguration,
        accept_open_team_sheet=True,
        max_concurrent_battles=1,
        log_level=30,
    )
