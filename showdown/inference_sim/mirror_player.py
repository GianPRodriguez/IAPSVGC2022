"""Player poke-env que espeja el protocolo al Battle IAPSVGC y valida soundness."""
from __future__ import annotations

from typing import List, Optional

from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from showdown.inference_sim.feeder import bootstrap_battle, feed_line
from showdown.inference_sim.paths import BATTLE_FORMAT
from showdown.inference_sim.soundness import (
    SoundnessReport,
    check_soundness,
    record_timeline,
    snapshot_beliefs,
)
from showdown.inference_sim.team_factory import TeamMeta, to_showdown_export
from showdown.stat_inference import reset_sp_limits


class InferenceMirrorPlayer(Player):
    """
    Juega al azar en doubles Champions, y en paralelo alimenta el módulo de
    inferencia del bot con el protocolo crudo.
    """

    def __init__(
        self,
        *,
        user_team: TeamMeta,
        opp_team_ref: TeamMeta,
        report: SoundnessReport,
        account_configuration: Optional[AccountConfiguration] = None,
        battle_format: str = BATTLE_FORMAT,
        **kwargs,
    ):
        self.user_team = user_team
        self.opp_team_ref = opp_team_ref  # ground truth del rival (el otro player)
        self.report = report
        self.iap_battle = None
        self.protocol_log: List[str] = []
        self._bootstrapped = False
        self._our_role = 'p1'
        export = to_showdown_export(user_team)
        super().__init__(
            account_configuration=account_configuration,
            battle_format=battle_format,
            team=export,
            server_configuration=LocalhostServerConfiguration,
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
            start_listening=True,
            **kwargs,
        )

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)
        return self.choose_random_doubles_move(battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        return '/team 12'

    async def _handle_battle_message(self, split_messages: List[List[str]]):
        # Captura líneas crudas antes/después del parse de poke-env
        room = split_messages[0][0] if split_messages else ''
        for parts in split_messages[1:]:
            if not parts:
                continue
            line = '|'.join(parts)
            if not line.startswith('|'):
                line = '|' + line
            self.protocol_log.append(line)

            # Detectar nuestro rol
            if len(parts) > 2 and parts[1] == 'player':
                # |player|p1|Nick|...
                pid = parts[2].strip()
                nick = parts[3].strip() if len(parts) > 3 else ''
                if nick and nick == self.username:
                    self._our_role = pid

            if len(parts) > 1 and parts[1] == 'start':
                self._ensure_iap(room)

            if self.iap_battle is not None:
                mapped = self._map_line_to_iap(line)
                try:
                    action = feed_line(self.iap_battle, mapped)
                except Exception as exc:
                    self.protocol_log.append(f'# feed error: {exc!r} :: {mapped}')
                    action = None
                if action in ('turn', '-damage', 'move', 'switch', 'drag', 'faint'):
                    try:
                        check_soundness(
                            self.iap_battle,
                            self.opp_team_ref,
                            battle_tag=room,
                            recent_lines=self.protocol_log,
                            report=self.report,
                        )
                        turn_n = int(getattr(self.iap_battle, 'turn', 0) or 0)
                        snaps = snapshot_beliefs(
                            self.iap_battle,
                            self.opp_team_ref,
                            turn=turn_n,
                            event=action,
                        )
                        record_timeline(self.report, snaps)
                    except Exception as exc:
                        self.protocol_log.append(f'# soundness error: {exc!r}')

        try:
            await super()._handle_battle_message(split_messages)
        except KeyError as exc:
            # poke-env dex gaps (Champions formes) no deben tumbar la batalla IAP
            self.protocol_log.append(f'# poke-env KeyError skipped: {exc!r}')
            return

        # Tras request/showteam de poke-env, sincronizar IDs si hace falta
        if self.iap_battle is not None and not self._bootstrapped:
            self._sync_role_ids()

    def _ensure_iap(self, room: str) -> None:
        if self.iap_battle is not None:
            return
        reset_sp_limits()
        self.iap_battle = bootstrap_battle(self.user_team, self.opp_team_ref)
        self._sync_role_ids()
        self._bootstrapped = True
        self.protocol_log.append(f'# iap bootstrap room={room}')

    def _sync_role_ids(self) -> None:
        if self.iap_battle is None:
            return
        # IAP siempre trata "user" como nuestro lado; ids deben coincidir con PS.
        self.iap_battle.user.id = self._our_role
        self.iap_battle.opponent.id = 'p2' if self._our_role == 'p1' else 'p1'

    def _map_line_to_iap(self, line: str) -> str:
        """Si somos p2, no reescribimos slots: user.id ya es p2."""
        return line

    def reset_for_next_battle(self, opp_team: TeamMeta) -> None:
        self.opp_team_ref = opp_team
        self.iap_battle = None
        self._bootstrapped = False
        self.protocol_log = []
        self._our_role = 'p1'
