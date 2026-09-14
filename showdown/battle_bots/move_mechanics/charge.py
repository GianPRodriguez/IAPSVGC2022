"""Two-turn charge moves (Electro Shot, Solar Beam, Fly, …).

Fuente de verdad compartida para greedy / joint / expectimax y agentes futuros.
Aproxima el `onTryMove` de Pokémon Showdown: flag `charge` + excepciones
(weather / Power Herb / ya en twoturnmove).

Power Herb solo se consume si el weather NO dispara el move en 1 turno
(mismo orden que PS: weather early-return antes del evento ChargeMove).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple


def norm_id(raw: Optional[str]) -> str:
    return (
        str(raw or '')
        .strip()
        .lower()
        .replace(' ', '')
        .replace('-', '')
        .replace('_', '')
        .replace("'", '')
        .replace('.', '')
    )


# Moves con flags.charge en dex PS (gen actual / Champions).
CHARGE_MOVES: frozenset[str] = frozenset({
    'bounce',
    'dig',
    'dive',
    'electroshot',
    'fly',
    'freezeshock',
    'geomancy',
    'iceburn',
    'meteorbeam',
    'phantomforce',
    'razorwind',
    'shadowforce',
    'skullbash',
    'skyattack',
    'skydrop',
    'solarbeam',
    'solarblade',
})

# Semi-invulnerables durante el turno de carga.
INVULNERABLE_ON_CHARGE: frozenset[str] = frozenset({
    'bounce',
    'dig',
    'dive',
    'fly',
    'phantomforce',
    'shadowforce',
    'skydrop',
})

# Boosts que ocurren en el turno de preparación (también si el weather dispara ya).
CHARGE_BOOSTS: Dict[str, Dict[str, int]] = {
    'electroshot': {'spa': 1},
    'meteorbeam': {'spa': 1},
    'skullbash': {'def': 1},
    'geomancy': {'spa': 2, 'spd': 2, 'spe': 2},
}

_RAIN = frozenset({'raindance', 'primordialsea'})
_SUN = frozenset({'sunnyday', 'desolateland'})

# Utilidad 1-ply por punto de boost en turno de carga (sin daño).
CHARGE_BOOST_UNIT = 6.0
# Coste de gastar Power Herb para saltar la carga (ítem se pierde).
POWER_HERB_COST = 18.0


@dataclass(frozen=True)
class ChargeResolution:
    """Resolución del timing de un move con charge este turno."""

    move_id: str
    is_charge_move: bool
    fires_this_turn: bool
    is_charging: bool
    invulnerable: bool
    locked_next: bool
    charge_boosts: Dict[str, int]
    # True solo si Power Herb es lo que permite disparar (weather no bastaba).
    consumes_power_herb: bool = False

    @property
    def deals_damage_this_turn(self) -> bool:
        """True solo si el hit de daño (o efecto de turno 2) ocurre ahora."""
        return bool(self.is_charge_move and self.fires_this_turn) or (
            not self.is_charge_move
        )


def is_charge_move(move_name: Optional[str]) -> bool:
    return norm_id(move_name) in CHARGE_MOVES


def _norm_weather(weather: Optional[str]) -> str:
    return norm_id(weather)


def has_power_herb(item: Optional[str]) -> bool:
    return norm_id(item) == 'powerherb'


def _instant_by_weather(move_id: str, weather: Optional[str]) -> bool:
    w = _norm_weather(weather)
    if move_id == 'electroshot' and w in _RAIN:
        return True
    if move_id in ('solarbeam', 'solarblade') and w in _SUN:
        return True
    return False


def resolve_charge(
    move_name: Optional[str],
    *,
    weather: Optional[str] = None,
    item: Optional[str] = None,
    already_charging: bool = False,
) -> ChargeResolution:
    """
    ¿Este move pega / aplica efecto este turno?

    Orden PS:
      1) already_charging (volatile del move) → hit
      2) weather instant (rain/sun) → hit, NO consume Power Herb
      3) Power Herb (evento ChargeMove) → hit + consume ítem
      4) else → turno de carga
    """
    mid = norm_id(move_name)
    empty = ChargeResolution(
        move_id=mid,
        is_charge_move=False,
        fires_this_turn=True,
        is_charging=False,
        invulnerable=False,
        locked_next=False,
        charge_boosts={},
        consumes_power_herb=False,
    )
    if mid not in CHARGE_MOVES:
        return empty

    boosts = dict(CHARGE_BOOSTS.get(mid) or {})
    invuln = mid in INVULNERABLE_ON_CHARGE

    if already_charging:
        return ChargeResolution(
            move_id=mid,
            is_charge_move=True,
            fires_this_turn=True,
            is_charging=False,
            invulnerable=False,
            locked_next=False,
            charge_boosts={},
            consumes_power_herb=False,
        )

    if _instant_by_weather(mid, weather):
        return ChargeResolution(
            move_id=mid,
            is_charge_move=True,
            fires_this_turn=True,
            is_charging=False,
            invulnerable=False,
            locked_next=False,
            charge_boosts=boosts,
            consumes_power_herb=False,
        )

    if has_power_herb(item):
        return ChargeResolution(
            move_id=mid,
            is_charge_move=True,
            fires_this_turn=True,
            is_charging=False,
            invulnerable=False,
            locked_next=False,
            charge_boosts=boosts,
            consumes_power_herb=True,
        )

    return ChargeResolution(
        move_id=mid,
        is_charge_move=True,
        fires_this_turn=False,
        is_charging=True,
        invulnerable=invuln,
        locked_next=True,
        charge_boosts=boosts,
        consumes_power_herb=False,
    )


def charge_boost_score(res: ChargeResolution) -> float:
    """Heurística 1-ply por boosts del turno de carga."""
    if not res.charge_boosts:
        return 0.0
    return CHARGE_BOOST_UNIT * float(sum(abs(int(v)) for v in res.charge_boosts.values()))


def power_herb_cost(res: ChargeResolution) -> float:
    """Coste 1-ply por consumir Power Herb este turno."""
    return POWER_HERB_COST if res.consumes_power_herb else 0.0


def item_after_charge(
    item: Optional[str],
    res: ChargeResolution,
) -> Optional[str]:
    """Ítem residual tras resolver el move (None si se gastó Power Herb)."""
    if res.consumes_power_herb:
        return None
    return item


def gate_damage_range(
    rng: Optional[Tuple[int, int]],
    move_name: Optional[str],
    *,
    weather: Optional[str] = None,
    item: Optional[str] = None,
    already_charging: bool = False,
    must_recharge: bool = False,
) -> Optional[Tuple[int, int]]:
    """Anula el rango si está recargando o en turno de carga."""
    if must_recharge:
        return None
    res = resolve_charge(
        move_name,
        weather=weather,
        item=item,
        already_charging=already_charging,
    )
    if res.is_charge_move and not res.fires_this_turn:
        return None
    return rng


def is_two_turn_charging_from_volatiles(
    volatiles: Optional[Mapping] = None,
) -> bool:
    """True si Battle/IAP marca twoturnmove o el id del move en volatiles."""
    if not volatiles:
        return False
    for key, val in volatiles.items():
        if not val:
            continue
        kid = norm_id(key)
        if kid in ('twoturnmove', 'twoturn') or kid in CHARGE_MOVES:
            return True
    return False
