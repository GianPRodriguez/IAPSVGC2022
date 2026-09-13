"""Orden de turno 1-ply: Spe efectiva + prioridad de movimiento."""
from __future__ import annotations

from typing import Optional

from showdown.battle_bots.greedy_solo.greedy import CalcPoke, _stats_or_default
from showdown.stat_inference import effective_speed, move_priority


def poke_effective_spe(
    poke: CalcPoke,
    *,
    tailwind: bool = False,
) -> int:
    stats = _stats_or_default(poke)
    spe = int(stats.get('spe') or 100)
    boost = int((poke.boosts or {}).get('spe') or 0)
    status = str(poke.status or '').lower()
    paralyzed = status.startswith('par')
    return int(effective_speed(
        spe,
        boost=boost,
        item=poke.item,
        paralyzed=paralyzed,
        tailwind=tailwind,
    ))


def move_prio(
    name: Optional[str],
    *,
    ability: Optional[str] = None,
    terrain: Optional[str] = None,
) -> int:
    """Prioridad numérica; desconocido → 0 (no descartamos el duelo Spe)."""
    raw = move_priority(name, ability, terrain)
    return int(raw) if raw is not None else 0


def acts_before(
    spe_a: int,
    prio_a: int,
    spe_b: int,
    prio_b: int,
    *,
    trick_room: bool = False,
) -> bool:
    """True si A actúa estrictamente antes que B (empate Spe → False)."""
    if int(prio_a) != int(prio_b):
        return int(prio_a) > int(prio_b)
    if trick_room:
        return int(spe_a) < int(spe_b)
    return int(spe_a) > int(spe_b)


def race_scale(p_ko_first: float, *, weight: float = 0.85) -> float:
    """
    Si actúas primero y puedes KO, la acción del rival se escala por esto.
    Continuo: (1 - weight * P(KO)).
    """
    pk = max(0.0, min(1.0, float(p_ko_first)))
    return max(0.0, 1.0 - float(weight) * pk)
