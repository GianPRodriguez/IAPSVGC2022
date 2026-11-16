"""Guardia de decisiones de los jugadores RL: contar y registrar fallos en vez de tragarlos.

Antes, ``DDQNPlayer.choose_move`` capturaba cualquier excepción y devolvía la orden por
defecto de Showdown sin avisar: un error de observación o de máscara se veía como un agente
que juega mal. Ahora:

- se cuentan las decisiones, las excepciones (por tipo) y las órdenes que el adaptador tuvo
  que reemplazar porque la acción de la red no era ejecutable (``order_fallbacks``);
- las primeras ``PS_RL_GUARD_LOG_N`` excepciones (5 por defecto) se registran con traceback;
- con ``PS_RL_STRICT=1`` la excepción se propaga (para tests y validación en vivo).
"""
from __future__ import annotations

import logging
import os
import traceback
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict

log = logging.getLogger('showdown.rl.decision_guard')


def strict_mode() -> bool:
    return os.environ.get('PS_RL_STRICT', '').strip().lower() in ('1', 'true', 'yes', 'on')


@dataclass
class DecisionStats:
    agent: str = 'rl'
    decisions: int = 0
    exceptions: int = 0
    order_fallbacks: int = 0
    by_error: Counter = field(default_factory=Counter)
    last_error: str = ''

    def record_ok(self, *, order_fallback: bool = False) -> None:
        self.decisions += 1
        if order_fallback:
            self.order_fallbacks += 1

    def record_exception(self, exc: BaseException) -> None:
        """Cuenta la excepción, la registra (las primeras N) y la propaga en modo estricto."""
        self.decisions += 1
        self.exceptions += 1
        name = type(exc).__name__
        self.by_error[name] += 1
        self.last_error = f'{name}: {exc}'
        try:
            limit = int(os.environ.get('PS_RL_GUARD_LOG_N', '5'))
        except ValueError:
            limit = 5
        if self.exceptions <= limit:
            log.warning('%s: excepción en choose_move (%d/%d decisiones): %s\n%s',
                        self.agent, self.exceptions, self.decisions, self.last_error,
                        ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        if strict_mode():
            raise exc

    def as_dict(self) -> Dict[str, Any]:
        n = self.decisions or 1
        return {
            'agent': self.agent,
            'decisions': self.decisions,
            'exceptions': self.exceptions,
            'exception_rate': self.exceptions / n,
            'order_fallbacks': self.order_fallbacks,
            'order_fallback_rate': self.order_fallbacks / n,
            'by_error': dict(self.by_error),
            'last_error': self.last_error,
        }
