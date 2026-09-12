"""GreedyDamagePlayer + espejo IAP Battle + contraste vs poke-env (PS local)."""
from __future__ import annotations

from typing import List, Optional

from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from showdown.battle_bots.heuristic.pokeenv_player import GreedyDamagePlayer
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


class ValidatedGreedyPlayer(GreedyDamagePlayer):
    """
    Heurístico greedy + feed del protocolo a Battle IAP + contraste con poke-env.
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
        self._our_role = 'p1'
        self._role_confirmed = False
        self._ready_to_check = False
        self._pe_battle = None
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
        # Completar con el resto para que reserved tenga el sheet completo
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

        for parts in split_messages[1:]:
            if not parts:
                continue
            line = '|'.join(parts)
            if not line.startswith('|'):
                line = '|' + line
            # Cap protocolo en memoria (créditos / RAM en BO3 largos)
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

            if len(parts) > 1 and parts[1] == 'start':
                # Cada game de un BO3 reinicia el espejo IAP
                self.iap_battle = None
                self._ready_to_check = False
                self._ensure_iap(room)

            action = parts[1].strip() if len(parts) > 1 else ''
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
                if key in _INFER_ACTIONS:
                    pending_infer = True

        try:
            await super()._handle_battle_message(split_messages)
        except KeyError as exc:
            if len(self.protocol_log) < 420:
                self.protocol_log.append(f'# poke-env KeyError skipped: {exc!r}')
            return
        except Exception as exc:
            # BO3: requests tras game-over / websocket ya cerrado
            name = type(exc).__name__
            msg = str(exc).lower()
            if name in ('ConnectionClosedOK', 'ConnectionClosedError', 'ConnectionClosed') or 'close' in msg:
                return
            if len(self.protocol_log) < 420:
                self.protocol_log.append(f'# poke-env handler skipped: {exc!r}')
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
            # Contar estrechamiento simple
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
            _ = before  # reserved for future delta metrics
        except Exception as exc:
            self.soundness_report.battle_errors += 1
            if len(self.protocol_log) < 420:
                self.protocol_log.append(f'# soundness error: {exc!r}')
