"""Proyección 1-ply de habilidades al entrar (switch-in → ΔV).

Generaliza Intimidate / clima / Screen Cleaner / self-boosts sin hardcodear
especies: solo ability del entrante + estado de foes/campo.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from showdown.battle_bots.move_mechanics.charge import norm_id

# −1 Atk a cada foe afectado ≈ alivio de amenaza física.
INTIMIDATE_PER_FOE = 32.0
# Self +1 Atk/Def/SpA al entrar.
SELF_BOOST_VALUE = 22.0
# Fijar clima/terreno cuando el campo aún no lo tiene.
WEATHER_SET_VALUE = 38.0
TERRAIN_SET_VALUE = 34.0
# Limpiar Reflect/Light Screen rivales.
SCREEN_CLEANER_VALUE = 40.0
# Costar: copiar boosts del compañero (por stage positivo).
COSTAR_PER_STAGE = 10.0
COSTAR_CAP = 45.0
# Download: +1 Atk o SpA según Def/SpD rival.
DOWNLOAD_VALUE = 20.0
# Supersweet Syrup: −1 evasion (menor).
EVASION_DROP_VALUE = 8.0

# Gen8+: bloquean Intimidate (y Adrenaline Orb no aplica aquí).
_INTIMIDATE_IMMUNE_ABI = frozenset({
    'clearbody',
    'whitesmoke',
    'fullmetalbody',
    'hypercutter',
    'innerfocus',
    'owntempo',
    'oblivious',
    'scrappy',
})

# Les sube stats ante Intimidate: no contar como alivio.
_INTIMIDATE_REVERSAL_ABI = frozenset({
    'defiant',
    'competitive',
    'guarddog',
})

_WEATHER_SETTERS: Dict[str, str] = {
    'drizzle': 'raindance',
    'drought': 'sunnyday',
    'sandstream': 'sandstorm',
    'snowwarning': 'snow',
    'orichalcumpulse': 'sunnyday',
    'hadronengine': 'electricterrain',  # también terrain; ver abajo
}

_TERRAIN_SETTERS: Dict[str, str] = {
    'electricsurge': 'electricterrain',
    'psychicsurge': 'psychicterrain',
    'grassysurge': 'grassyterrain',
    'mistysurge': 'mistyterrain',
    'hadronengine': 'electricterrain',
}

_SELF_ENTRY_BOOST: Dict[str, str] = {
    'intrepidsword': 'atk',
    'dauntlessshield': 'def',
}


def _norm_weather(raw: Optional[str]) -> str:
    w = norm_id(raw)
    aliases = {
        'rain': 'raindance',
        'sun': 'sunnyday',
        'sunny': 'sunnyday',
        'desolateland': 'sunnyday',
        'primordialsea': 'raindance',
        'hail': 'snow',
        'snowscape': 'snow',
    }
    return aliases.get(w, w)


def _norm_terrain(raw: Optional[str]) -> str:
    return norm_id(raw)


def _boost(poke: Any, stat: str) -> int:
    b = getattr(poke, 'boosts', None) or {}
    try:
        return int(b.get(stat, 0) or 0)
    except (TypeError, ValueError):
        return 0


def intimidate_can_affect(foe: Any) -> bool:
    """False si Substitute, −6 Atk, inmune, o Defiant/Competitive/Guard Dog."""
    if foe is None:
        return False
    if bool(getattr(foe, 'has_substitute', False)):
        return False
    if _boost(foe, 'atk') <= -6:
        return False
    abi = norm_id(getattr(foe, 'ability', None))
    if abi in _INTIMIDATE_IMMUNE_ABI or abi in _INTIMIDATE_REVERSAL_ABI:
        return False
    return True


def _intimidate_value(foes: Sequence[Any]) -> float:
    n = sum(1 for f in foes if intimidate_can_affect(f))
    return float(n) * INTIMIDATE_PER_FOE


def _self_boost_value(entrant: Any, stat: str) -> float:
    if _boost(entrant, stat) >= 6:
        return 0.0
    return SELF_BOOST_VALUE


def _costar_value(ally: Optional[Any]) -> float:
    if ally is None:
        return 0.0
    stages = 0
    for st in ('atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'):
        v = _boost(ally, st)
        if v > 0:
            stages += v
    if stages <= 0:
        return 0.0
    return min(COSTAR_CAP, float(stages) * COSTAR_PER_STAGE)


def _download_value(foes: Sequence[Any]) -> float:
    if not foes:
        return DOWNLOAD_VALUE * 0.5
    # Proxy: si hay foes, Download casi siempre da +1 a algo útil.
    return DOWNLOAD_VALUE


def _weather_set_value(ability: str, current_weather: Optional[str]) -> float:
    want = _WEATHER_SETTERS.get(ability)
    if not want:
        return 0.0
    # Hadron Engine es terrain, no weather (salvo clave dual).
    if ability == 'hadronengine':
        return 0.0
    cur = _norm_weather(current_weather)
    if cur and cur == want:
        return 0.0
    # Climas permanentes rivales no se pisan (aprox.).
    if cur in ('desolateland', 'primordialsea', 'deltastream'):
        return 0.0
    return WEATHER_SET_VALUE


def _terrain_set_value(ability: str, current_terrain: Optional[str]) -> float:
    want = _TERRAIN_SETTERS.get(ability)
    if not want:
        return 0.0
    cur = _norm_terrain(current_terrain)
    if cur and cur == want:
        return 0.0
    return TERRAIN_SET_VALUE


def _screen_cleaner_value(
    *,
    opp_reflect: bool = False,
    opp_light_screen: bool = False,
    opp_aurora_veil: bool = False,
) -> float:
    n = int(bool(opp_reflect)) + int(bool(opp_light_screen)) + int(bool(opp_aurora_veil))
    if n <= 0:
        return 0.0
    return SCREEN_CLEANER_VALUE * (0.65 + 0.35 * n)


def entry_ability_switch_value(
    entrant: Any,
    foes: Sequence[Any],
    *,
    ally: Optional[Any] = None,
    weather: Optional[str] = None,
    terrain: Optional[str] = None,
    opp_reflect: bool = False,
    opp_light_screen: bool = False,
    opp_aurora_veil: bool = False,
    weight: float = 1.0,
) -> float:
    """
    ΔV al sacar `entrant` este turno por su ability de entrada.

    foes: activos rivales (CalcPoke u objetos con ability/boosts/sub).
    """
    if entrant is None:
        return 0.0
    abi = norm_id(getattr(entrant, 'ability', None))
    if not abi:
        return 0.0

    total = 0.0
    if abi == 'intimidate':
        total += _intimidate_value(foes)
    elif abi == 'supersweetsyrup':
        total += EVASION_DROP_VALUE * float(len([f for f in foes if f is not None]))
    elif abi in _SELF_ENTRY_BOOST:
        total += _self_boost_value(entrant, _SELF_ENTRY_BOOST[abi])
    elif abi == 'download':
        total += _download_value(foes)
    elif abi == 'costar':
        total += _costar_value(ally)
    elif abi == 'screencleaner':
        total += _screen_cleaner_value(
            opp_reflect=opp_reflect,
            opp_light_screen=opp_light_screen,
            opp_aurora_veil=opp_aurora_veil,
        )

    total += _weather_set_value(abi, weather)
    total += _terrain_set_value(abi, terrain)

    return float(weight) * total


def entry_ability_breakdown(
    entrant: Any,
    foes: Sequence[Any],
    **kwargs: Any,
) -> Dict[str, float]:
    """Debug/tests: componentes no nulos."""
    abi = norm_id(getattr(entrant, 'ability', None) if entrant else None)
    out: Dict[str, float] = {}
    if not abi:
        return out
    if abi == 'intimidate':
        out['intimidate'] = _intimidate_value(foes)
    elif abi == 'supersweetsyrup':
        out['supersweetsyrup'] = EVASION_DROP_VALUE * float(
            len([f for f in foes if f is not None])
        )
    elif abi in _SELF_ENTRY_BOOST:
        out[abi] = _self_boost_value(entrant, _SELF_ENTRY_BOOST[abi])
    elif abi == 'download':
        out['download'] = _download_value(foes)
    elif abi == 'costar':
        out['costar'] = _costar_value(kwargs.get('ally'))
    elif abi == 'screencleaner':
        out['screencleaner'] = _screen_cleaner_value(
            opp_reflect=bool(kwargs.get('opp_reflect', False)),
            opp_light_screen=bool(kwargs.get('opp_light_screen', False)),
            opp_aurora_veil=bool(kwargs.get('opp_aurora_veil', False)),
        )
    w = _weather_set_value(abi, kwargs.get('weather'))
    if w:
        out['weather'] = w
    t = _terrain_set_value(abi, kwargs.get('terrain'))
    if t:
        out['terrain'] = t
    return out
