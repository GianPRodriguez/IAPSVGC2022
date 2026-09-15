"""Live poke-env BattleBackend for VGCEnv (local PS, Spec v1.1)."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from showdown.rl.env import BattleBackend
from showdown.rl.pe_adapter import (
    joint_order_from_spec_actions,
    pe_autoregressive_masks,
    snapshot_from_battle,
)
from showdown.rl.snapshot import BattleSnapshot

logger = logging.getLogger(__name__)

TeamLike = Union[str, Any]  # showdown export str or TeamMeta


def _team_export(team: Optional[TeamLike]) -> Optional[str]:
    if team is None:
        return None
    if isinstance(team, str):
        return team
    from showdown.inference_sim.team_factory import to_showdown_export
    return to_showdown_export(team)


def _default_teams(seed: Optional[int] = None) -> Tuple[str, str]:
    from showdown.inference_sim.team_factory import active_teams, to_showdown_export

    pool = list(active_teams())
    if len(pool) < 2:
        raise RuntimeError('Need ≥2 teams in MetaVGC pool for live RL backend')
    rng = np.random.default_rng(seed)
    i, j = int(rng.integers(0, len(pool))), int(rng.integers(0, len(pool) - 1))
    if j >= i:
        j += 1
    return to_showdown_export(pool[i]), to_showdown_export(pool[j])


class PokeEnvBattleBackend(BattleBackend):
    """
    Sync Gym bridge: controlled `_EnvPlayer` vs `RandomPlayer` on local Showdown.

    Episode = one BO1 game (finished battle). Requires PS listening
    (`node pokemon-showdown start --no-security`).
    """

    def __init__(
        self,
        *,
        battle_format: Optional[str] = None,
        server_configuration=None,
        our_team: Optional[TeamLike] = None,
        opp_team: Optional[TeamLike] = None,
        challenge_timeout: float = 90.0,
        step_timeout: float = 180.0,
        start_ps: bool = False,
        log_level: int = 30,
        account_prefix: str = 'RlEnv',
    ):
        from poke_env.ps_client import LocalhostServerConfiguration

        from showdown.inference_sim.paths import BATTLE_FORMAT

        self.battle_format = battle_format or BATTLE_FORMAT
        self.server_configuration = server_configuration or LocalhostServerConfiguration
        self._our_team_arg = our_team
        self._opp_team_arg = opp_team
        self.challenge_timeout = float(challenge_timeout)
        self.step_timeout = float(step_timeout)
        self.start_ps = bool(start_ps)
        self.log_level = int(log_level)
        self.account_prefix = str(account_prefix)

        self._agent = None
        self._foe = None
        self._battle = None
        self._challenge_task = None
        self._uid = 0
        self._closed = False

    # --- BattleBackend -----------------------------------------------------

    def reset(self, *, seed: Optional[int] = None) -> BattleSnapshot:
        self._ensure_ps()
        self._abort_open_battle()
        self._ensure_players(seed=seed)

        assert self._agent is not None and self._foe is not None
        loop = self._agent.ps_client.loop
        self._challenge_task = asyncio.run_coroutine_threadsafe(
            self._agent.battle_against(self._foe, n_battles=1),
            loop,
        )
        try:
            self._battle = self._agent.battle_queue.get(timeout=self.challenge_timeout)
        except Exception as exc:
            self._challenge_task = None
            raise TimeoutError(
                f'RL env challenge timed out ({self.challenge_timeout}s): {exc!r}'
            ) from exc
        return snapshot_from_battle(self._battle)

    def step(
        self,
        action: Tuple[int, int],
    ) -> Tuple[BattleSnapshot, Dict[str, Any]]:
        if self._battle is None or self._agent is None:
            raise RuntimeError('PokeEnvBattleBackend.step called before reset')
        if self._battle.finished:
            return snapshot_from_battle(self._battle), {
                'engine': 'pokeenv',
                'used_fallback': False,
                'already_finished': True,
            }

        order, used_fallback = joint_order_from_spec_actions(self._battle, action)
        self._agent.order_queue.put(order)
        try:
            self._battle = self._agent.battle_queue.get(timeout=self.step_timeout)
        except Exception as exc:
            raise TimeoutError(
                f'RL env step timed out ({self.step_timeout}s): {exc!r}'
            ) from exc

        meta: Dict[str, Any] = {
            'engine': 'pokeenv',
            'used_fallback': bool(used_fallback),
            'order': str(order),
        }
        return snapshot_from_battle(self._battle), meta

    def action_masks(
        self,
        *,
        slot1_action_id: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._battle is None:
            return None
        return pe_autoregressive_masks(self._battle, slot1_action_id=slot1_action_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._abort_open_battle()
        for pl in (self._agent, self._foe):
            if pl is None:
                continue
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    pl.ps_client.stop_listening(),
                    pl.ps_client.loop,
                )
                fut.result(timeout=10)
            except Exception:
                pass
        self._agent = None
        self._foe = None
        self._battle = None

    # --- internals ---------------------------------------------------------

    def _ensure_ps(self) -> None:
        from showdown.inference_sim.ps_server import ensure_ps, is_ps_up, ps_start_hint

        if not ensure_ps(start=self.start_ps):
            raise RuntimeError(
                'Pokémon Showdown no responde en localhost. '
                f'Arranca: {ps_start_hint()}  o pasa start_ps=True'
            )
        if not is_ps_up():
            raise RuntimeError(f'PS still down after ensure; {ps_start_hint()}')

    def _ensure_players(self, *, seed: Optional[int]) -> None:
        from poke_env.environment.env import _EnvPlayer
        from poke_env.player import RandomPlayer
        from poke_env.ps_client import AccountConfiguration

        our = _team_export(self._our_team_arg)
        opp = _team_export(self._opp_team_arg)
        if our is None or opp is None:
            d0, d1 = _default_teams(seed=seed)
            our = our or d0
            opp = opp or d1

        self._uid += 1
        stamp = int(time.time()) % 100000
        name_a = f'{self.account_prefix}A{stamp}{self._uid}'
        name_b = f'{self.account_prefix}B{stamp}{self._uid}'

        if self._agent is None:
            self._agent = _EnvPlayer(
                account_configuration=AccountConfiguration(name_a, None),
                battle_format=self.battle_format,
                team=our,
                server_configuration=self.server_configuration,
                accept_open_team_sheet=True,
                max_concurrent_battles=1,
                log_level=self.log_level,
                choose_on_teampreview=False,
            )
            self._foe = RandomPlayer(
                account_configuration=AccountConfiguration(name_b, None),
                battle_format=self.battle_format,
                team=opp,
                server_configuration=self.server_configuration,
                accept_open_team_sheet=True,
                max_concurrent_battles=1,
                log_level=self.log_level,
            )
        else:
            assert self._foe is not None
            self._agent.update_team(our)
            self._foe.update_team(opp)
            # Drain queues
            while not self._agent.battle_queue.empty():
                try:
                    self._agent.battle_queue.get(timeout=0.1)
                except Exception:
                    break
            while not self._agent.order_queue.empty():
                try:
                    self._agent.order_queue.get(timeout=0.1)
                except Exception:
                    break
            self._agent.battle = None

    def _abort_open_battle(self) -> None:
        from poke_env.player.battle_order import ForfeitBattleOrder

        if self._agent is None or self._battle is None:
            self._challenge_task = None
            return
        if not getattr(self._battle, 'finished', True):
            try:
                if not self._agent.battle_queue.empty():
                    self._agent.battle_queue.get(timeout=1.0)
                self._agent.order_queue.put(ForfeitBattleOrder())
                self._agent.battle_queue.get(timeout=15.0)
            except Exception as exc:
                logger.debug('forfeit during abort: %r', exc)
        if self._challenge_task is not None:
            try:
                self._challenge_task.result(timeout=30)
            except Exception:
                pass
            self._challenge_task = None
        self._battle = None
