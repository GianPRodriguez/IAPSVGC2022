"""Post-hit recharge moves (Hyper Beam, Giga Impact, …).

Tras el hit, el usuario queda en `mustrecharge` y no actúa el turno siguiente.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from showdown.battle_bots.move_mechanics.charge import norm_id

# Moves con flags.recharge en dex PS.
RECHARGE_MOVES: frozenset[str] = frozenset({
    'blastburn',
    'eternabeam',
    'frenzyplant',
    'gigaimpact',
    'hydrocannon',
    'hyperbeam',
    'meteorassault',
    'prismaticlaser',
    'roaroftime',
    'rockwrecker',
})

# Penalización 1-ply por perder el próximo turno (doubles: alto).
RECHARGE_LOCK_PENALTY = 28.0


@dataclass(frozen=True)
class RechargeResolution:
    move_id: str
    is_recharge_move: bool
    """True si este move, al ejecutarse, impone mustrecharge el siguiente turno."""
    must_recharge_next: bool
    """True si el Pokémon ya está obligado a recargar este turno (no puede actuar)."""
    currently_recharging: bool

    @property
    def can_act(self) -> bool:
        return not self.currently_recharging


def is_recharge_move(move_name: Optional[str]) -> bool:
    return norm_id(move_name) in RECHARGE_MOVES


def resolve_recharge(
    move_name: Optional[str] = None,
    *,
    currently_recharging: bool = False,
) -> RechargeResolution:
    """
    - currently_recharging: volatile mustrecharge / poke-env.must_recharge.
    - Si el move tiene flags.recharge y se ejecuta ahora → must_recharge_next.
    """
    mid = norm_id(move_name)
    if currently_recharging:
        return RechargeResolution(
            move_id=mid,
            is_recharge_move=mid in RECHARGE_MOVES,
            must_recharge_next=False,
            currently_recharging=True,
        )
    if mid in RECHARGE_MOVES:
        return RechargeResolution(
            move_id=mid,
            is_recharge_move=True,
            must_recharge_next=True,
            currently_recharging=False,
        )
    return RechargeResolution(
        move_id=mid,
        is_recharge_move=False,
        must_recharge_next=False,
        currently_recharging=False,
    )


def recharge_lock_penalty(res: RechargeResolution, *, ko: bool = False) -> float:
    """Coste de imponer mustrecharge el turno siguiente (menor si KO garantizado)."""
    if not res.must_recharge_next:
        return 0.0
    return RECHARGE_LOCK_PENALTY * (0.4 if ko else 1.0)


def is_must_recharge_from_volatiles(
    volatiles: Optional[Mapping] = None,
) -> bool:
    if not volatiles:
        return False
    for key, val in volatiles.items():
        if not val:
            continue
        if norm_id(key) in ('mustrecharge', 'must_recharge'):
            return True
    return False
