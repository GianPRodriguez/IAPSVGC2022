"""Parches defensivos poke-env para OTS/BO3 (idempotentes).

Causa histórica: |showteam| hacía IndexError si teampreview no matcheaba
(p.ej. |poke| acumulados sin vaciar en clearpoke). Eso mataba el handler,
la serie no terminaba y el harness esperaba todo el --timeout (antes 600s).

El fix principal está en el repo poke-env local (clearpoke + match seguro).
Este módulo añade una red de seguridad al importar el harness.

Importante: NUNCA tragar excepciones en salas >game* (BO3 padre): si se pierde
el |win|, `_battle_count_queue` no hace task_done y battle_against cuelga.
"""
from __future__ import annotations

_APPLIED = False


def apply_pokeenv_showteam_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return
    try:
        from poke_env.player import player as pl
    except Exception:
        return

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
        try:
            return await _orig_handle(self, split_messages)
        except IndexError as exc:
            self.logger.warning('poke-env showteam/battle IndexError skipped: %r', exc)
            return None
        except ValueError as exc:
            msg = str(exc).lower()
            if 'unpack' in msg or 'values' in msg:
                self.logger.warning('poke-env packed-team ValueError skipped: %r', exc)
                return None
            raise

    pl.Player._handle_battle_message = _handle_battle_message  # type: ignore[method-assign]
    _APPLIED = True


apply_pokeenv_showteam_patches()
