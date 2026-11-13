"""Guarda robusta contra /choose|/team a salas BO3 ya cerradas.

Problema: choose_move async (joint/expectimax) puede terminar DESPUÉS de
/leave → popup PS "not in that room".

Capas:
  1. Marcar room en _left_rooms al terminar la batalla.
  2. Revalidar finished/_left_rooms ANTES y DESPUÉS de await choose.
  3. Wrapper de ps_client.send_message que traga choose/team a rooms stale.
  4. Filtro de log: popup "not in that room" → DEBUG (no WARNING).
"""
from __future__ import annotations

import logging
import random
from typing import Any, Awaitable, Optional, Set

from poke_env.battle import AbstractBattle


def _norm_room(tag: Any) -> str:
    return str(tag or '').lstrip('>')


class _StaleRoomPopupFilter(logging.Filter):
    """Silencia popups cosméticos de cleanup (sala / challenge ya resuelto)."""

    _DROP_SNIPPETS = (
        'not in that room',
        'were not in that room',
        'You are not challenging',
        'is not challenging you',
        'Maybe they accepted/rejected before you cancelled',
        'Maybe they cancelled before you rejected',
        'User not found',
        'is offline',
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        # Shutdown BO3: cancel de _handle_message (ruido, no fallo).
        if 'CancelledError intercepted' in msg:
            return False
        if 'Popup message received' not in msg:
            return True
        for snip in self._DROP_SNIPPETS:
            if snip in msg:
                return False
        return True


class StaleRoomGuardMixin:
    """Mixin para poke-env Player: no enviar órdenes a rooms ya left/finished."""

    _left_rooms: Set[str]
    _stale_send_guard_installed: bool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_left_rooms') or self._left_rooms is None:
            self._left_rooms = set()
        self._stale_send_guard_installed = False
        self._install_stale_room_guards()

    def _mark_room_left(self, battle_or_tag: Any) -> None:
        tag = (
            getattr(battle_or_tag, 'battle_tag', None)
            if not isinstance(battle_or_tag, str)
            else battle_or_tag
        )
        key = _norm_room(tag)
        if key:
            self._left_rooms.add(key)

    def _room_is_stale(self, battle: AbstractBattle) -> bool:
        if getattr(battle, 'finished', False):
            return True
        tag = getattr(battle, 'battle_tag', None)
        if not tag:
            return False
        key = _norm_room(tag)
        if key in self._left_rooms:
            return True
        battles = getattr(self, 'battles', None) or {}
        b = battles.get(tag)
        if b is None:
            for k, v in battles.items():
                if _norm_room(k) == key:
                    b = v
                    break
        if b is not None and getattr(b, 'finished', False):
            self._mark_room_left(key)
            return True
        return False

    def _battle_finished_callback(self, battle: AbstractBattle):
        self._mark_room_left(battle)
        return super()._battle_finished_callback(battle)  # type: ignore[misc]

    async def _handle_battle_request(
        self,
        battle: AbstractBattle,
        maybe_default_order: bool = False,
    ):
        """Copia el flujo poke-env con guarda post-await antes de send_message."""
        if self._room_is_stale(battle):
            return
        if getattr(battle, '_wait', False):
            waiting = getattr(self, '_waiting', None)
            if waiting is not None:
                waiting.set()
            return

        message: Optional[str] = None
        try:
            default_chance = float(getattr(self, 'DEFAULT_CHOICE_CHANCE', 0.0) or 0.0)
            if maybe_default_order and random.random() < default_chance:
                message = self.choose_default_move().message  # type: ignore[attr-defined]
            elif battle.teampreview:
                m = self.teampreview(battle)  # type: ignore[attr-defined]
                if isinstance(m, Awaitable):
                    m = await m
                message = m
            else:
                if maybe_default_order:
                    trying = getattr(self, '_trying_again', None)
                    if trying is not None:
                        trying.set()
                choice = self.choose_move(battle)  # type: ignore[attr-defined]
                if isinstance(choice, Awaitable):
                    choice = await choice
                message = choice.message if choice is not None else None
        except Exception:
            return

        # Crítico: revalidar tras el await (calc lento / BO3 leave).
        if not message or self._room_is_stale(battle):
            return
        try:
            await self.ps_client.send_message(message, battle.battle_tag)  # type: ignore[attr-defined]
        except Exception:
            return

    def _install_stale_room_guards(self) -> None:
        client = getattr(self, 'ps_client', None)
        if client is None:
            return
        self._install_popup_filter(client)
        if getattr(client, '_stale_room_send_guard', False):
            return
        orig = client.send_message
        player = self

        async def _guarded_send(
            message: str,
            room: str = '',
            message_2: Optional[str] = None,
        ):
            msg = str(message or '')
            room_key = _norm_room(room)
            if room_key and msg.startswith((
                '/choose', '/team', '/confirmready', '/timer',
            )):
                if room_key in player._left_rooms:
                    return
                battles = getattr(player, 'battles', None) or {}
                battle = battles.get(room) or battles.get(f'>{room_key}')
                if battle is None:
                    for k, b in battles.items():
                        if _norm_room(k) == room_key:
                            battle = b
                            break
                if battle is not None and getattr(battle, 'finished', False):
                    player._mark_room_left(room_key)
                    return
                # Tras /leave el mapa puede quedarse vacío o sin ese tag:
                # no reenviar órdenes de batalla a rooms fantasma.
                if battle is None:
                    player._mark_room_left(room_key)
                    return
            return await orig(message, room, message_2)

        client.send_message = _guarded_send  # type: ignore[method-assign]
        client._stale_room_send_guard = True

    @staticmethod
    def _install_popup_filter(client: Any) -> None:
        logger = getattr(client, 'logger', None)
        if logger is None:
            return
        name = '_stale_room_popup_filter'
        if any(getattr(f, 'name', '') == name for f in logger.filters):
            return
        filt = _StaleRoomPopupFilter()
        filt.name = name  # type: ignore[attr-defined]
        logger.addFilter(filt)
