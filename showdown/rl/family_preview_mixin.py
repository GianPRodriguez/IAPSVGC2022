"""Mixin for poke-env combat players: frozen family Team Preview."""
from __future__ import annotations

from typing import Any, Optional

from showdown.rl.frozen_preview import family_preview_for_combat, load_frozen_preview


class FamilyPreviewMixin:
    """Requires subclass to set ``_combat_family_id`` (e.g. ``'ddqn'``)."""

    _combat_family_id: str = ''
    _frozen_preview: Any = None

    def init_family_preview(
        self,
        *,
        preview_checkpoint: Optional[str] = None,
        preview_kind: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if preview_kind:
            self._frozen_preview = load_frozen_preview(
                kind=preview_kind,
                checkpoint=preview_checkpoint,
                device=device,
            )
        else:
            self._frozen_preview = family_preview_for_combat(
                self._combat_family_id,
                checkpoint=preview_checkpoint,
                device=device,
            )

    def teampreview(self, battle) -> str:
        if self._frozen_preview is None:
            self.init_family_preview()
        return self._frozen_preview.teampreview(battle)
