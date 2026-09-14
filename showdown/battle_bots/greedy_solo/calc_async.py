"""Cliente asyncio hacia el microservicio Node @smogon/calc (TCP/stdio)."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple

from showdown.damage_calc_bridge import damage_ranges_batch


_WARMED = False


def warm_damage_calc() -> bool:
    """Precalienta transporte TCP/stdio (una vez por proceso)."""
    global _WARMED
    if _WARMED:
        return True
    try:
        from showdown.damage_calc_bridge import DamageCalcBridge
        ok = DamageCalcBridge.instance()._ensure_transport()
        _WARMED = bool(ok)
        return _WARMED
    except Exception:
        return False


async def calc_damage_batch(
    requests: Sequence[Dict[str, Any]],
    *,
    chunk_size: Optional[int] = None,
    deadline_monotonic: Optional[float] = None,
) -> List[Optional[Tuple[int, int]]]:
    """
    Evalúa un lote de cálculos de daño sin bloquear el event loop.

    Los misses a Node van en chunks (default DAMAGE_CALC_BATCH_CHUNK=32).
    Si deadline_monotonic (perf_counter) se excede entre chunks, el resto queda None.
    """
    if not requests:
        return []
    if not _WARMED:
        await asyncio.to_thread(warm_damage_calc)

    def _run():
        return damage_ranges_batch(
            list(requests),
            chunk_size=chunk_size,
            deadline_monotonic=deadline_monotonic,
        )

    return await asyncio.to_thread(_run)


async def calc_damage_one(request: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    results = await calc_damage_batch([request])
    return results[0] if results else None


def expected_damage(rng: Optional[Tuple[int, int]]) -> float:
    if not rng:
        return 0.0
    lo, hi = rng
    return 0.5 * (float(lo) + float(hi))


def damage_pct(rng: Optional[Tuple[int, int]], max_hp: int) -> float:
    if not rng or max_hp <= 0:
        return 0.0
    return 100.0 * expected_damage(rng) / float(max_hp)


def guaranteed_ko(rng: Optional[Tuple[int, int]], cur_hp: int) -> bool:
    if not rng or cur_hp <= 0:
        return False
    return int(rng[0]) >= int(cur_hp)
