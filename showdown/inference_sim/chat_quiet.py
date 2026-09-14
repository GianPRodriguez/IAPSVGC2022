"""Control de prints de protocolo / sim (menos chat, más throughput).

Por defecto silencioso en corridas batch. Opt-in verbose:
  ROBUST_E2E_VERBOSE=1  o  --verbose en run_robust_e2e
  PS_CHAT_QUIET=0       fuerza chat aunque el batch sea quiet
"""
from __future__ import annotations

import os
from typing import Any

_force_quiet: bool | None = None


def set_chat_quiet(quiet: bool) -> None:
    global _force_quiet
    _force_quiet = bool(quiet)


def is_chat_quiet() -> bool:
    if _force_quiet is not None:
        return _force_quiet
    raw = os.environ.get('PS_CHAT_QUIET', '').strip().lower()
    if raw in ('0', 'false', 'no', 'off'):
        return False
    if raw in ('1', 'true', 'yes', 'on'):
        return True
    # Default: quiet (sim farm / robust sin --verbose)
    verbose = os.environ.get('ROBUST_E2E_VERBOSE', '').strip().lower()
    if verbose in ('1', 'true', 'yes', 'on'):
        return False
    return True


def chat_print(*args: Any, **kwargs: Any) -> None:
    if is_chat_quiet():
        return
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)
