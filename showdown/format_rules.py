"""Format constraints for the thesis stack (VGC Gen 9 Regulation M-C).

Regulation M-C (Champions / Open Team Sheets) uses **Mega Evolution**
(including new Megas in the regulation) and does **not** use Terastallization.
"""
from __future__ import annotations

# Kill-switch: keep ActionSpec id 22 / code paths for protocol parity, but
# never enable Tera under this project's target format.
TERASTALLIZATION_ENABLED: bool = False


def allow_tera(raw_can_tera: bool = False) -> bool:
    """True only if format enables Tera *and* the engine offers it this turn."""
    return bool(TERASTALLIZATION_ENABLED) and bool(raw_can_tera)
