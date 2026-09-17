"""Parches defensivos poke-env para OTS/BO3 (idempotentes).

Causa histórica: |showteam| hacía IndexError si teampreview no matcheaba
(p.ej. |poke| acumulados sin vaciar en clearpoke). Eso mataba el handler,
la serie no terminaba y el harness esperaba todo el --timeout (antes 600s).

Además: match suave preview↔packed (Indeedee-F / Mega / …) y demote del
WARNING `showteam: no match for preview` a debug.

Importante: NUNCA tragar excepciones en salas >game* (BO3 padre): si se pierde
el |win|, `_battle_count_queue` no hace task_done y battle_against cuelga.
"""
from __future__ import annotations

import logging

_APPLIED = False
# Unbound real: nunca cerrar sobre un wrapper previo (reentrada → RecursionError).
_LOGGER_WARNING = logging.Logger.warning
_LOGGER_DEBUG = logging.Logger.debug


def _soft_identifies_extra(preview_mon, nickname: str) -> bool:
    try:
        from poke_env.data.normalize import to_id_str
    except Exception:
        return False
    nid = to_id_str(nickname or '')
    if not nid:
        return False
    ps = to_id_str(getattr(preview_mon, 'species', '') or '')
    pb = to_id_str(getattr(preview_mon, 'base_species', '') or '')
    if nid in (ps, pb):
        return True
    if pb and (nid.startswith(pb) or pb.startswith(nid)):
        return True
    if ps and (nid.startswith(ps) or ps.startswith(nid)):
        return True
    for suf in ('mega', 'megax', 'megay', 'gmax', 'totem'):
        if len(nid) > len(suf) and nid.endswith(suf) and nid[: -len(suf)] in (ps, pb):
            return True
        if len(ps) > len(suf) and ps.endswith(suf) and ps[: -len(suf)] == nid:
            return True
    return False


def apply_pokeenv_showteam_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return
    try:
        from poke_env.battle.pokemon import Pokemon
        from poke_env.player import player as pl
    except Exception:
        return

    _orig_identifies = Pokemon.identifies_as

    def _identifies_as(self, ident: str) -> bool:  # type: ignore[no-untyped-def]
        try:
            if _orig_identifies(self, ident):
                return True
        except Exception:
            pass
        return _soft_identifies_extra(self, ident)

    Pokemon.identifies_as = _identifies_as  # type: ignore[method-assign]

    _orig_handle = pl.Player._handle_battle_message

    async def _handle_battle_message(self, split_messages):  # type: ignore[no-untyped-def]
        room = ''
        try:
            if split_messages and split_messages[0]:
                room = str(split_messages[0][0] or '')
        except Exception:
            room = ''
        # Salas padre BO3: no swallow — |win|/|init| deben actualizar la cola.
        if room.lstrip('>').startswith('game'):
            return await _orig_handle(self, split_messages)

        # Demote showteam miss. Contador anti-reentrada: mensajes anidados no
        # deben capturar _warning_filtered como "original" (RecursionError).
        depth = int(getattr(self, '_iap_showteam_warn_depth', 0) or 0)
        log = self.logger

        def _warning_filtered(msg, *args, **kwargs):  # type: ignore[no-untyped-def]
            text = str(msg)
            if 'showteam: no match for preview' in text:
                _LOGGER_DEBUG(log, msg, *args, **kwargs)
                return
            _LOGGER_WARNING(log, msg, *args, **kwargs)

        if depth == 0:
            self._iap_showteam_warn_prev = log.warning
            log.warning = _warning_filtered  # type: ignore[method-assign]
        self._iap_showteam_warn_depth = depth + 1
        try:
            return await _orig_handle(self, split_messages)
        except IndexError as exc:
            _LOGGER_WARNING(log, 'poke-env showteam/battle IndexError skipped: %r', exc)
            return None
        except ValueError as exc:
            msg = str(exc).lower()
            if 'unpack' in msg or 'values' in msg:
                _LOGGER_WARNING(log, 'poke-env packed-team ValueError skipped: %r', exc)
                return None
            raise
        finally:
            self._iap_showteam_warn_depth = max(
                0, int(getattr(self, '_iap_showteam_warn_depth', 1) or 1) - 1,
            )
            if self._iap_showteam_warn_depth == 0:
                prev = getattr(self, '_iap_showteam_warn_prev', None)
                if prev is not None:
                    log.warning = prev  # type: ignore[method-assign]

    pl.Player._handle_battle_message = _handle_battle_message  # type: ignore[method-assign]
    _APPLIED = True


apply_pokeenv_showteam_patches()
