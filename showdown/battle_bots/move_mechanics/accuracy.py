"""Accuracy de moves para scoring (no árbol hit/miss completo).

Regla: score_esperado ≈ daño * p_hit; KO_BONUS también escalado por p_hit.
Preferir 100% cuando el KO no es matemático seguro (acc < ~99%).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from showdown.battle_bots.move_mechanics.charge import norm_id


def _moves_json() -> Dict[str, Any]:
    try:
        from showdown.battle_modifier import _moves_json as _load
        return _load() or {}
    except Exception:
        return {}


# Overrides cuando moves.json no tiene el move (Champions / gen nueva).
_ACC_OVERRIDES: Dict[str, float] = {
    'direclaw': 1.0,
}


@lru_cache(maxsize=512)
def move_accuracy(move_name: Optional[str]) -> float:
    """Probabilidad de acierto en [0, 1]. True/ausente → 1.0."""
    mid = norm_id(move_name)
    if not mid:
        return 1.0
    if mid in _ACC_OVERRIDES:
        return float(_ACC_OVERRIDES[mid])
    entry = _moves_json().get(mid) or {}
    raw = entry.get('accuracy', True)
    if raw is True or raw is None:
        return 1.0
    if raw is False:
        return 0.0
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))


def accuracy_scale_damage(damage_pct: float, move_name: Optional[str]) -> float:
    """E[daño%] = daño% * accuracy."""
    return float(damage_pct) * move_accuracy(move_name)


def accuracy_scale_ko_bonus(ko_bonus: float, move_name: Optional[str]) -> float:
    """E[KO_BONUS] = KO_BONUS * accuracy (no tratar 80% como KO garantizado)."""
    return float(ko_bonus) * move_accuracy(move_name)


def accuracy_tiebreak(move_name: Optional[str]) -> float:
    """Pequeño empate a favor de más accuracy (≪ 1% HP)."""
    return 0.05 * move_accuracy(move_name)
