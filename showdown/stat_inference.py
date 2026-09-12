"""
Estimacion de los Stat Points (SP) del oponente a partir de lo observado en combate.

Contexto Champions:
  - Los IV son siempre 31 y el nivel es 50, asi que la unica incognita por stat
    son los Stat Points: 0..32 por stat, 66 en total.
  - La naturaleza del rival SI se conoce (llega en el |showteam| / OTS), igual que
    su objeto y habilidad. Eso deja el problema en un espacio discreto y pequeno.

Modelo:
  Para cada Pokemon rival se guarda, por stat, el conjunto de SP todavia
  compatibles con todo lo observado. El conjunto arranca completo y SOLO se
  estrecha, nunca se ensancha, de modo que la convergencia es monotona.

Regla de oro: ante cualquier duda NO se estrecha. Una poda incorrecta es
irreversible y envenena el resto de la partida, mientras que una observacion
descartada solo cuesta un turno.

Fuentes de evidencia:
  1. Orden de accion (velocidad / Spe).
  2. Dano observado via damage-calc Champions (Atk/SpA y Def/SpD).
  3. Con dano en % HP del rival, tambien HP (par Def|SpD + HP).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import constants
from showdown.damage_calc_bridge import damage_ranges_batch

# --------------------------------------------------------------------------- #
# Reglas de Champions
# --------------------------------------------------------------------------- #

SP_MIN = 0
SP_MAX = 32          # Champions: maximo de Stat Points en un solo stat
SP_BUDGET = 66       # Champions: maximo repartido entre los seis stats
IV = 31
LEVEL = 50

# Tope absoluto por si el servidor no aplica las reglas de Champions y acepta
# repartos al estilo EV clasico (252 EV -> 63 puntos dentro de la formula).
SP_HARD_MAX = 63
SP_HARD_BUDGET = 6 * SP_HARD_MAX

# Limites vigentes. Arrancan en los de Champions y se ensanchan solo si la
# evidencia los contradice (ver calibrate_from_known_team).
_limits = {'max': SP_MAX, 'budget': SP_BUDGET}


def sp_limits() -> Tuple[int, int]:
    return _limits['max'], _limits['budget']


def configure_sp_limits(max_per_stat: int, budget: int) -> None:
    _limits['max'] = max(SP_MIN, min(int(max_per_stat), SP_HARD_MAX))
    _limits['budget'] = max(0, min(int(budget), SP_HARD_BUDGET))


def reset_sp_limits() -> None:
    configure_sp_limits(SP_MAX, SP_BUDGET)

STAT_KEYS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')

# stat interno -> nombre usado por constants.NATURE_EFFECTS
_NATURE_STAT_NAMES = {
    'atk': 'attack',
    'def': 'defense',
    'spa': 'special-attack',
    'spd': 'special-defense',
    'spe': 'speed',
}

# stat interno -> atributo de base de especie en la clase Pokemon
BASE_ATTRS = {
    'hp': 'HPB_battle_base',
    'atk': 'ATK_battle_base',
    'def': 'DEF_battle_base',
    'spa': 'SPA_battle_base',
    'spd': 'SPD_battle_base',
    'spe': 'VEL_battle_base',
}


def nature_multiplier(nature: Optional[str], stat: str) -> float:
    """Devuelve 1.1 / 0.9 / 1.0 segun la naturaleza afecte al stat."""
    if stat == 'hp':
        return 1.0
    effects = constants.NATURE_EFFECTS.get((nature or '').strip().lower())
    if not effects:
        return 1.0
    target = _NATURE_STAT_NAMES.get(stat)
    if effects.get('plus') == target:
        return 1.1
    if effects.get('minus') == target:
        return 0.9
    return 1.0


def champions_stat(base: int, sp: int, nature: Optional[str], stat: str) -> int:
    """
    Stat final a nivel 50 en Champions.

        no-HP: floor((floor((2*base + 31 + SP) * 50/100) + 5) * naturaleza)
        HP:    floor((2*base + 31 + SP) * 50/100) + nivel + 10

    Verificado contra los stats exactos que Showdown manda en |request|.
    """
    inner = (2 * base + IV + sp) * LEVEL // 100
    if stat == 'hp':
        return inner + LEVEL + 10
    return int((inner + 5) * nature_multiplier(nature, stat))


# --------------------------------------------------------------------------- #
# Creencia por Pokemon
# --------------------------------------------------------------------------- #

class StatBelief:
    """Conjuntos de Stat Points todavia posibles para un Pokemon rival."""

    def __init__(self, species: str, base_stats: Dict[str, int], nature: Optional[str]):
        self.species = species
        self.nature = nature
        self.base_stats = dict(base_stats)
        # Forma cuyas bases están en base_stats (mega, blade, etc.). Los SP
        # son del Pokemon; al megaevolucionar solo cambian las bases.
        self.current_forme: str = species
        max_sp, _ = sp_limits()
        self.candidates: Dict[str, Set[int]] = {
            stat: set(range(SP_MIN, max_sp + 1)) for stat in STAT_KEYS
        }
        self.observations = 0
        self.skipped = 0

    # -- consultas -------------------------------------------------------- #

    def stat_values_with_bases(self, stat: str, bases: Dict[str, int]) -> Set[int]:
        """Valores finales con un set de bases concreto (p.ej. forma base vs mega)."""
        base = bases.get(stat)
        if not base or base <= 0:
            return set()
        return {
            champions_stat(int(base), sp, self.nature, stat)
            for sp in self.candidates[stat]
        }

    def stat_values(self, stat: str) -> Set[int]:
        """Valores finales de stat compatibles con los SP que quedan."""
        return self.stat_values_with_bases(stat, self.base_stats)

    def stat_range_with_bases(
        self, stat: str, bases: Dict[str, int]
    ) -> Optional[Tuple[int, int]]:
        values = self.stat_values_with_bases(stat, bases)
        if not values:
            return None
        return min(values), max(values)

    def stat_range(self, stat: str) -> Optional[Tuple[int, int]]:
        return self.stat_range_with_bases(stat, self.base_stats)

    def is_resolved(self, stat: str) -> bool:
        """True cuando ya solo queda un valor final posible."""
        return len(self.stat_values(stat)) == 1

    def point_estimate_stats(self) -> Dict[str, int]:
        """
        Stats finales puntuales para decisión (greedy): SP mediano del
        conjunto candidato actual → champions_stat.
        """
        out: Dict[str, int] = {}
        for stat in STAT_KEYS:
            sps = sorted(self.candidates.get(stat) or [])
            base = int(self.base_stats.get(stat) or 0)
            if not sps or base <= 0:
                continue
            mid_sp = sps[len(sps) // 2]
            out[stat] = champions_stat(base, mid_sp, self.nature, stat)
        return out

    def sp_range(self, stat: str) -> Optional[Tuple[int, int]]:
        sps = self.candidates.get(stat)
        if not sps:
            return None
        return min(sps), max(sps)

    # -- estrechamiento --------------------------------------------------- #

    def narrow(self, stat: str, allowed: Iterable[int]) -> bool:
        """
        Intersecta los candidatos de un stat. Devuelve True si hubo cambio.
        Si la interseccion quedara vacia se descarta la observacion completa:
        significa que alguna premisa del contexto era falsa y es preferible
        conservar lo que ya se sabia.
        """
        allowed = set(allowed)
        current = self.candidates[stat]
        new = current & allowed
        if not new:
            self.skipped += 1
            return False
        if new == current:
            return False
        self.candidates[stat] = new
        self._prune_budget()
        return True

    def _prune_budget(self) -> None:
        """
        Aplica el tope global de Stat Points: un candidato es viable solo si,
        sumado al minimo obligatorio de los demas stats, no se pasa del
        presupuesto.

        Solo se confia en el minimo de un stat cuando ya esta casi resuelto
        (<=3 candidatos). Si Spe/Atk siguen abiertos, usar su minimo como suelo
        duro envenena SpD/HP (p.ej. Spe mal acotado deja SpD=0 como unico
        valor presupuestariamente viable).
        """
        _, budget = sp_limits()
        for _ in range(len(STAT_KEYS)):
            changed = False
            floors = {}
            for stat in STAT_KEYS:
                cands = self.candidates[stat]
                floors[stat] = min(cands) if len(cands) <= 3 else 0
            for stat in STAT_KEYS:
                others = sum(v for s, v in floors.items() if s != stat)
                viable = {sp for sp in self.candidates[stat] if sp + others <= budget}
                if viable and viable != self.candidates[stat]:
                    self.candidates[stat] = viable
                    changed = True
            if not changed:
                return

    # -- reporte ---------------------------------------------------------- #

    def summary_line_with_bases(self, bases: Dict[str, int]) -> str:
        """Una linea compacta con el rango estimado de cada stat (bases dadas)."""
        parts = []
        for stat, label in (
            ('hp', 'HP'),
            ('atk', 'Atk'),
            ('def', 'Def'),
            ('spa', 'SpA'),
            ('spd', 'SpD'),
            ('spe', 'Spe'),
        ):
            rng = self.stat_range_with_bases(stat, bases)
            if rng is None:
                parts.append(f'{label}=?')
            elif rng[0] == rng[1]:
                parts.append(f'{label}={rng[0]}!')
            else:
                parts.append(f'{label}={rng[0]}-{rng[1]}')
        return ' '.join(parts)

    def summary_line(self) -> str:
        """Rango final con las bases actuales (mega/blade si aplica)."""
        return self.summary_line_with_bases(self.base_stats)

    def sp_summary_line(self) -> str:
        """Rangos de Stat Points candidatos (invariantes a la forma)."""
        parts = []
        for stat, label in (
            ('hp', 'HP'),
            ('atk', 'Atk'),
            ('def', 'Def'),
            ('spa', 'SpA'),
            ('spd', 'SpD'),
            ('spe', 'Spe'),
        ):
            rng = self.sp_range(stat)
            if rng is None:
                parts.append(f'{label}=?')
            elif rng[0] == rng[1]:
                parts.append(f'{label}={rng[0]}!')
            else:
                parts.append(f'{label}={rng[0]}-{rng[1]}')
        return ' '.join(parts)

    def convergence_line(self) -> str:
        """Cuantos valores finales quedan por stat (menos es mejor, 1 = resuelto)."""
        parts = []
        for stat, label in (
            ('hp', 'HP'),
            ('atk', 'Atk'),
            ('def', 'Def'),
            ('spa', 'SpA'),
            ('spd', 'SpD'),
            ('spe', 'Spe'),
        ):
            parts.append(f'{label}={len(self.stat_values(stat))}')
        return f'obs={self.observations} ' + ' '.join(parts)

    def is_narrow(self, stat: str, *, max_cands: int = 8) -> bool:
        """True si el SP de este stat ya está lo bastante estrecho para confiar."""
        cands = self.candidates.get(stat) or set()
        if not cands:
            return False
        if self.is_resolved(stat):
            return True
        return len(cands) <= max_cands


# --------------------------------------------------------------------------- #
# Velocidad efectiva
# --------------------------------------------------------------------------- #

_SPEED_HALVING_ITEMS = {'ironball', 'machobrace', 'poweranklet', 'powerband',
                        'powerbelt', 'powerbracer', 'powerlens', 'powerweight'}

# Habilidades cuya influencia en velocidad depende de condiciones que no
# modelamos con certeza. Si alguna esta en juego, la observacion se descarta.
_UNCERTAIN_SPEED_ABILITIES = {
    'swiftswim', 'chlorophyll', 'sandrush', 'slushrush', 'quickfeet',
    'unburden', 'slowstart', 'protosynthesis', 'quarkdrive', 'surgesurfer',
    'steamengine',
}


def effective_speed(stat_speed: int, *, boost: int = 0, item: Optional[str] = None,
                    paralyzed: bool = False, tailwind: bool = False) -> int:
    """
    Velocidad efectiva con los modificadores que si podemos afirmar con certeza:
    etapas de boost, Choice Scarf, objetos que la reducen a la mitad, Tailwind y
    paralisis (Gen 7+ la reduce al 50%).
    """
    speed = int(stat_speed * constants.BOOST_MULTIPLIERS.get(boost, 1.0))

    item_id = (item or '').strip().lower().replace(' ', '')
    if item_id == 'choicescarf':
        speed = speed * 3 // 2
    elif item_id in _SPEED_HALVING_ITEMS:
        speed //= 2

    if tailwind:
        speed *= 2
    if paralyzed:
        speed //= 2

    return speed


class SpeedContext:
    """Modificadores de velocidad de un Pokemon en el momento de actuar."""

    def __init__(self, *, boost: int = 0, item: Optional[str] = None,
                 ability: Optional[str] = None, paralyzed: bool = False,
                 tailwind: bool = False):
        self.boost = boost
        self.item = item
        self.ability = (ability or '').strip().lower().replace(' ', '')
        self.paralyzed = paralyzed
        self.tailwind = tailwind

    @property
    def is_uncertain(self) -> bool:
        """True si algun modificador no se puede afirmar con seguridad."""
        return self.ability in _UNCERTAIN_SPEED_ABILITIES

    def speed_of(self, stat_speed: int) -> int:
        return effective_speed(
            stat_speed,
            boost=self.boost,
            item=self.item,
            paralyzed=self.paralyzed,
            tailwind=self.tailwind,
        )


# --------------------------------------------------------------------------- #
# Metadata de movimientos (prioridad y categoria)
# --------------------------------------------------------------------------- #

_MOVE_META_PATH = (
    Path(__file__).resolve().parent.parent / 'API' / 'JSONs' / 'champions_move_meta.json'
)
_MOVE_META: Optional[Dict[str, Dict[str, object]]] = None


def _move_meta() -> Dict[str, Dict[str, object]]:
    global _MOVE_META
    if _MOVE_META is None:
        try:
            with _MOVE_META_PATH.open(encoding='utf-8') as handle:
                _MOVE_META = json.load(handle)
        except Exception as error:  # datos ausentes: se degrada sin romper
            print(f'[STAT-INF] No se pudo leer champions_move_meta.json: {error}')
            _MOVE_META = {}
    return _MOVE_META


def move_id(name: Optional[str]) -> str:
    return ''.join(ch for ch in (name or '').lower() if ch.isalnum())


def move_priority(
    name: Optional[str],
    ability: Optional[str] = None,
    terrain: Optional[str] = None,
) -> Optional[int]:
    """
    Prioridad del movimiento, incluyendo:
      - +1 de Prankster sobre movimientos de estado
      - +1 de Grassy Glide bajo Grassy Terrain

    Devuelve None si el movimiento es desconocido, para que quien llame
    pueda descartar la observacion en lugar de asumir prioridad 0.
    """
    mid = move_id(name)
    meta = _move_meta().get(mid)
    if meta is None:
        return None
    # No usar `or 0`: prioridades negativas (Trick Room = -7) son validas.
    raw = meta.get('p')
    priority = int(raw) if raw is not None else 0
    abi = (ability or '').strip().lower().replace(' ', '')
    if abi == 'prankster' and meta.get('c') == 't':
        priority += 1
    terr = (terrain or '').strip().lower().replace(' ', '')
    if mid == 'grassyglide' and terr == 'grassyterrain':
        priority += 1
    return priority


def move_category(name: Optional[str]) -> Optional[str]:
    """
    Categoria del movimiento: 'p' fisico, 's' especial, 't' estado.
    None si el movimiento no esta en el meta (no asumir).
    """
    meta = _move_meta().get(move_id(name))
    if meta is None:
        return None
    cat = meta.get('c')
    return cat if cat in ('p', 's', 't') else None


# --------------------------------------------------------------------------- #
# Contexto de dano
# --------------------------------------------------------------------------- #

# Habilidades cuyo efecto de daño no modelamos con certeza (aunque el calc las
# soporte en parte). El resto se pasa al calc vía ability / side flags.
_UNCERTAIN_DAMAGE_ABILITIES = {
    'parentalbond',   # dos hits; un solo |-damage| no basta
    'analytic',       # depende de ir último
    'rivalry',        # depende del género
    'wonderguard',    # inmunidad total; si hay daño, algo raro
}


class DamageContext:
    """Modificadores afirmables en el momento del golpe."""

    def __init__(
        self,
        *,
        atk_boost: int = 0,
        def_boost: int = 0,
        spa_boost: int = 0,
        spd_boost: int = 0,
        attacker_item: Optional[str] = None,
        defender_item: Optional[str] = None,
        attacker_ability: Optional[str] = None,
        defender_ability: Optional[str] = None,
        attacker_status: Optional[str] = None,
        defender_status: Optional[str] = None,
        weather: Optional[str] = None,
        terrain: Optional[str] = None,
        wonder_room: bool = False,
        magic_room: bool = False,
        gravity: bool = False,
        reflect: bool = False,
        light_screen: bool = False,
        aurora_veil: bool = False,
        helping_hand: bool = False,
        friend_guard: bool = False,
        charge: bool = False,
        is_spread: bool = False,
        is_crit: bool = False,
        defender_hp_fraction: Optional[float] = None,
        attacker_ability_on: bool = False,
        defender_ability_on: bool = False,
        attacker_types: Optional[List[str]] = None,
        defender_types: Optional[List[str]] = None,
        attacker_tera: Optional[str] = None,
        defender_tera: Optional[str] = None,
        is_battery: bool = False,
        is_power_spot: bool = False,
        is_steely_spirit: bool = False,
        is_flower_gift: bool = False,
        is_beads_of_ruin: bool = False,
        is_sword_of_ruin: bool = False,
        is_tablets_of_ruin: bool = False,
        is_vessel_of_ruin: bool = False,
        attacker_allies_fainted: int = 0,
    ):
        self.atk_boost = atk_boost
        self.def_boost = def_boost
        self.spa_boost = spa_boost
        self.spd_boost = spd_boost
        self.attacker_item = attacker_item
        self.defender_item = defender_item
        self.attacker_ability = (attacker_ability or '').strip().lower().replace(' ', '')
        self.defender_ability = (defender_ability or '').strip().lower().replace(' ', '')
        self.attacker_status = attacker_status
        self.defender_status = defender_status
        self.weather = weather
        self.terrain = terrain
        self.wonder_room = wonder_room
        self.magic_room = magic_room
        self.gravity = gravity
        self.reflect = reflect
        self.light_screen = light_screen
        self.aurora_veil = aurora_veil
        self.helping_hand = helping_hand
        self.friend_guard = friend_guard
        self.charge = charge
        self.is_spread = is_spread
        self.is_crit = is_crit
        # 1.0 = full HP (Multiscale / Shadow Shield). None = asumir full.
        self.defender_hp_fraction = defender_hp_fraction
        self.attacker_ability_on = bool(attacker_ability_on)
        self.defender_ability_on = bool(defender_ability_on)
        self.attacker_types = list(attacker_types) if attacker_types else None
        self.defender_types = list(defender_types) if defender_types else None
        self.attacker_tera = attacker_tera
        self.defender_tera = defender_tera
        self.is_battery = bool(is_battery)
        self.is_power_spot = bool(is_power_spot)
        self.is_steely_spirit = bool(is_steely_spirit)
        self.is_flower_gift = bool(is_flower_gift)
        self.is_beads_of_ruin = bool(is_beads_of_ruin)
        self.is_sword_of_ruin = bool(is_sword_of_ruin)
        self.is_tablets_of_ruin = bool(is_tablets_of_ruin)
        self.is_vessel_of_ruin = bool(is_vessel_of_ruin)
        self.attacker_allies_fainted = max(0, int(attacker_allies_fainted or 0))

    @property
    def is_uncertain(self) -> bool:
        return (
            self.is_crit
            or self.attacker_ability in _UNCERTAIN_DAMAGE_ABILITIES
            or self.defender_ability in _UNCERTAIN_DAMAGE_ABILITIES
        )


def _display_item(item: Optional[str]) -> Optional[str]:
    if not item:
        return None
    raw = str(item).strip()
    if not raw or raw == '-':
        return None
    # El worker acepta ids compactos; dejamos el string tal cual.
    return raw


def _display_ability(ability: Optional[str]) -> Optional[str]:
    if not ability:
        return None
    raw = str(ability).strip()
    return raw or None


def _weather_for_calc(weather: Optional[str]) -> Optional[str]:
    if not weather or weather in ('none', '-', 'null'):
        return None
    key = weather.strip().lower().replace(' ', '')
    mapping = {
        'sunnyday': 'Sun',
        'desolateland': 'Harsh Sunshine',
        'raindance': 'Rain',
        'primordialsea': 'Heavy Rain',
        'sandstorm': 'Sand',
        'snowscape': 'Snow',
        'hail': 'Hail',
    }
    return mapping.get(key)


def _terrain_for_calc(terrain: Optional[str]) -> Optional[str]:
    if not terrain:
        return None
    key = terrain.strip().lower().replace(' ', '')
    mapping = {
        'electricterrain': 'Electric',
        'grassyterrain': 'Grassy',
        'mistyterrain': 'Misty',
        'psychicterrain': 'Psychic',
    }
    return mapping.get(key)


def _species_types(species: Optional[str]) -> List[str]:
    """Tipos para el calc cuando la especie no está en el dex Champions."""
    if not species:
        return []
    try:
        from API.PokemonData import getPkmnInfo
        info = getPkmnInfo(species)
    except Exception:
        return []
    if not info:
        return []
    out = []
    for tip in (info[0], info[1]):
        if tip:
            out.append(str(tip))
    return out


def _poke_payload(
    species: str,
    nature: Optional[str],
    stats: Dict[str, int],
    *,
    item: Optional[str] = None,
    ability: Optional[str] = None,
    status: Optional[str] = None,
    boosts: Optional[Dict[str, int]] = None,
    cur_hp: Optional[int] = None,
    ability_on: bool = False,
    types: Optional[List[str]] = None,
    tera_type: Optional[str] = None,
    allies_fainted: int = 0,
) -> Dict:
    max_hp = int(stats.get('hp', 1) or 1)
    payload = {
        'species': species,
        'nature': (nature or 'Serious').capitalize() if nature else 'Serious',
        'item': _display_item(item),
        'ability': _display_ability(ability),
        'status': status or '',
        'boosts': {
            'atk': int((boosts or {}).get('atk', 0) or 0),
            'def': int((boosts or {}).get('def', 0) or 0),
            'spa': int((boosts or {}).get('spa', 0) or 0),
            'spd': int((boosts or {}).get('spd', 0) or 0),
            'spe': int((boosts or {}).get('spe', 0) or 0),
        },
        'stats': {
            'hp': max_hp,
            'atk': int(stats.get('atk', 1) or 1),
            'def': int(stats.get('def', 1) or 1),
            'spa': int(stats.get('spa', 1) or 1),
            'spd': int(stats.get('spd', 1) or 1),
            'spe': int(stats.get('spe', 1) or 1),
        },
    }
    live = [t for t in (types or []) if t]
    if live:
        payload['types'] = live
    else:
        dex_types = _species_types(species)
        if dex_types:
            payload['types'] = dex_types
    if cur_hp is not None and cur_hp > 0:
        payload['curHP'] = min(int(cur_hp), max_hp)
    if ability_on:
        payload['abilityOn'] = True
    if tera_type:
        payload['teraType'] = tera_type
    if allies_fainted and int(allies_fainted) > 0:
        payload['alliesFainted'] = int(allies_fainted)
    return payload


def _field_payload(ctx: DamageContext, category: str) -> Dict:
    """
    Side flags del calc Champions.

    Aurora Veil NO se emula como Reflect+Light Screen: el calc tiene
    isAuroraVeil (rebaja físico y especial) y, si está activo, ignora R/LS.
    """
    attacker_side: Dict[str, bool] = {}
    defender_side: Dict[str, bool] = {}

    if ctx.helping_hand:
        attacker_side['isHelpingHand'] = True
    if ctx.charge:
        attacker_side['isCharge'] = True
    if ctx.is_battery:
        attacker_side['isBattery'] = True
    if ctx.is_power_spot:
        attacker_side['isPowerSpot'] = True
    if ctx.is_steely_spirit:
        attacker_side['isSteelySpirit'] = True
    if ctx.is_flower_gift:
        attacker_side['isFlowerGift'] = True

    # Pantallas: Aurora Veil tiene prioridad semántica propia.
    if ctx.aurora_veil:
        defender_side['isAuroraVeil'] = True
    else:
        if category == 'p' and ctx.reflect:
            defender_side['isReflect'] = True
        if category == 's' and ctx.light_screen:
            defender_side['isLightScreen'] = True

    if ctx.friend_guard:
        defender_side['isFriendGuard'] = True

    return {
        'gameType': 'Doubles',
        'weather': _weather_for_calc(ctx.weather),
        'terrain': _terrain_for_calc(ctx.terrain),
        'isWonderRoom': bool(ctx.wonder_room),
        'isMagicRoom': bool(ctx.magic_room),
        'isGravity': bool(ctx.gravity),
        'isSpread': bool(ctx.is_spread),
        'isBeadsOfRuin': bool(ctx.is_beads_of_ruin),
        'isSwordOfRuin': bool(ctx.is_sword_of_ruin),
        'isTabletsOfRuin': bool(ctx.is_tablets_of_ruin),
        'isVesselOfRuin': bool(ctx.is_vessel_of_ruin),
        'attackerSide': attacker_side,
        'defenderSide': defender_side,
    }


# Desync tipico calc Champions vs PS: pantallas+terreno, redondeo de % HP,
# shells de especies ausentes en el dex. Preferimos no podar el verdadero.
_ABS_DAMAGE_TOLERANCE = 8
_PERCENT_DAMAGE_TOLERANCE = 6
# Margen en Spe efectiva: Tailwind 80→160 vs Spe 163, empates, mega mid-turn.
_SPEED_COMPARE_MARGIN = 6


def _rolls_match_absolute(
    min_roll: int, max_roll: int, observed: int, *, is_ko: bool = False
) -> bool:
    """
    Cota unilateral (soundness): el SP es viable si el calc PUEDE alcanzar el
    dano observado. No excluimos Atk/SpA altos (ni Def/SpD bajos) porque el calc
    sobreestime el golpe — esa fue la fuente de falsos negativos e2e.
    """
    if observed <= 0:
        return False
    tol = _ABS_DAMAGE_TOLERANCE
    # Capaz de hacer al menos el dano visto (KO/overkill incluido).
    return max_roll >= max(1, observed - tol)


def _rolls_match_percent(
    min_roll: int, max_roll: int, max_hp: int, delta_pct: int, *, is_ko: bool = False
) -> bool:
    """
    Igual que absolute pero en franja de % HP del rival: solo cota inferior de
    dano (max_roll debe alcanzar el delta observado).
    """
    if max_hp <= 0 or delta_pct <= 0:
        return False
    tol = _PERCENT_DAMAGE_TOLERANCE
    low = max(1, (delta_pct - tol) * max_hp // 100)
    return max_roll >= low


# --------------------------------------------------------------------------- #
# Gestor de creencias del equipo rival
# --------------------------------------------------------------------------- #

def solve_sp(base: int, nature: Optional[str], stat: str, observed: int) -> Set[int]:
    """Stat Points que reproducen exactamente un stat ya conocido."""
    if not base or base <= 0 or observed <= 0:
        return set()
    return {
        sp for sp in range(SP_MIN, SP_HARD_MAX + 1)
        if champions_stat(base, sp, nature, stat) == observed
    }


def calibrate_from_known_team(pokemons: Iterable) -> None:
    """
    Ajusta los topes de Stat Points usando el propio equipo, del que Showdown
    manda los stats exactos en el |request|.

    Si el reparto propio necesita mas puntos de los que permite Champions es que
    el servidor no esta aplicando esas reglas, y mantener el tope bajo haria
    descartar el valor verdadero del rival. En ese caso se ensancha: preferimos
    converger mas despacio antes que converger a un numero equivocado.

    Ojo: en el lado propio los atributos *_battle_base NO guardan la base de
    especie (solo el rival los usa asi), de modo que las bases se leen siempre
    del pokedex y los stats reales de los *_battle_actual.
    """
    needed_max = 0
    needed_budget = 0

    for pkmn in pokemons or []:
        if pkmn is None:
            continue
        nature = getattr(pkmn, 'nature', None)
        name = getattr(pkmn, 'name', '')
        per_pokemon = 0
        for stat, actual_attr in ACTUAL_ATTRS.items():
            try:
                observed = int(getattr(pkmn, actual_attr, -1) or -1)
            except (TypeError, ValueError):
                continue
            base = _base_from_pokedex(name, stat)
            candidates = solve_sp(base, nature, stat, observed)
            if not candidates:
                continue
            # El minimo del par de SP equivalentes basta para acotar el reparto.
            low = min(candidates)
            needed_max = max(needed_max, low)
            per_pokemon += low
        needed_budget = max(needed_budget, per_pokemon)

    max_sp, budget = sp_limits()
    if needed_max > max_sp or needed_budget > budget:
        configure_sp_limits(max(max_sp, needed_max), max(budget, needed_budget))
        new_max, new_budget = sp_limits()
        print(
            f'[STAT-INF] El equipo propio necesita hasta {needed_max} SP en un stat '
            f'y {needed_budget} en total: topes ensanchados a {new_max}/{new_budget} '
            f'(Champions seria {SP_MAX}/{SP_BUDGET}).'
        )


ACTUAL_ATTRS = {
    'hp': 'HPB_battle_base',
    'atk': 'ATK_battle_actual',
    'def': 'DEF_battle_actual',
    'spa': 'SPA_battle_actual',
    'spd': 'SPD_battle_actual',
    'spe': 'VEL_battle_actual',
}

_POKEDEX_STAT_FIELDS = {
    'hp': 'hp',
    'atk': 'ataque',
    'def': 'defensa',
    'spa': 'ataque_especial',
    'spd': 'defensa_especial',
    'spe': 'velocidad',
}


def _base_from_pokedex(name: str, stat: str) -> int:
    """Base de especie desde el JSON de Champions (el lado propio no la guarda)."""
    from API.PokemonData import getPkmnInfo

    info = getPkmnInfo(name)
    if not info:
        return 0
    order = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
    try:
        return int(info[2 + order.index(stat)] or 0)
    except (IndexError, TypeError, ValueError):
        return 0


def dex_base_stats(name: str) -> Dict[str, int]:
    """Bases de especie (hp/atk/def/spa/spd/spe) desde el dex Champions."""
    out: Dict[str, int] = {}
    for stat in STAT_KEYS:
        value = _base_from_pokedex(name, stat)
        if value > 0:
            out[stat] = value
    return out


_FORME_SUFFIXES = ('-mega-x', '-mega-y', '-mega', '-blade', '-shield', '-gmax')


def normalize_species(name: Optional[str]) -> str:
    """
    Clave de especie para agrupar creencias. Las formas que comparten Stat Points
    (Mega, Blade/Shield) colapsan en la especie base; las que son Pokemon
    distintos (Zoroark-Hisui, Indeedee-F) se mantienen separadas.
    """
    key = (name or '').strip().lower().replace(' ', '-')
    for suffix in _FORME_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


class TeamStatBeliefs:
    """
    Creencias de todo el equipo rival, indexadas por especie base.

    Los Stat Points pertenecen al Pokemon, no a la forma, asi que Mega
    Evoluciones y cambios de forma comparten el mismo conjunto de candidatos y
    solo cambian las bases con las que se evaluan. Eso hace que megaevolucionar
    aporte restricciones extra sobre lo ya aprendido.
    """

    def __init__(self):
        self.beliefs: Dict[str, StatBelief] = {}

    # -- ciclo de vida ---------------------------------------------------- #

    def toClean(self) -> None:
        """Olvida todo (se llama al terminar el BO3, no entre juegos)."""
        self.beliefs.clear()

    def forget(self, species: str) -> None:
        """
        Descarta lo aprendido de una especie. Se usa cuando se revela un Illusion:
        las observaciones atribuidas al disfraz eran de Zoroark, no de la especie
        real, asi que lo aprendido de ella no es fiable.
        """
        key = normalize_species(species)
        if key in self.beliefs:
            del self.beliefs[key]
            print(f'[STAT-INF] Creencias de {key} descartadas (Illusion)')

    # -- acceso ----------------------------------------------------------- #

    def belief_for(self, pkmn) -> Optional[StatBelief]:
        """Devuelve (creando si hace falta) la creencia del Pokemon rival dado."""
        if pkmn is None:
            return None
        species = normalize_species(getattr(pkmn, 'name', ''))
        if not species:
            return None

        base_stats = {}
        for stat, attr in BASE_ATTRS.items():
            try:
                value = int(getattr(pkmn, attr, -1) or -1)
            except (TypeError, ValueError):
                value = -1
            if value > 0:
                base_stats[stat] = value

        # Rival: HPB_battle_base guarda el denominador del protocolo (casi siempre
        # 100 = HP%), NO la base de especie. Sin esto, la creencia de HP usa
        # base=100 y excluye el SP verdadero.
        if getattr(pkmn, 'is_opponent', False):
            try:
                from API.PokemonData import getPkmnInfo
                info = getPkmnInfo(getattr(pkmn, 'name', '') or species)
                species_hp = int(info[2] or 0) if info else 0
            except Exception:
                species_hp = 0
            if species_hp > 0:
                base_stats['hp'] = species_hp

        if 'spe' not in base_stats:
            return None  # sin bases fiables no se puede inferir nada

        forme = (getattr(pkmn, 'name', '') or species).strip().lower().replace(' ', '-')
        belief = self.beliefs.get(species)
        if belief is None:
            belief = StatBelief(species, base_stats, getattr(pkmn, 'nature', None))
            belief.current_forme = forme
            self.beliefs[species] = belief
        else:
            # Cambio de forma (Mega, Blade/Shield): mismas SP, bases distintas.
            belief.base_stats.update(base_stats)
            belief.current_forme = forme
        return belief

    # -- evidencia de velocidad ------------------------------------------- #

    def observe_speed_order(self, opponent_pkmn, opponent_ctx: SpeedContext,
                            user_speed: int, user_ctx: SpeedContext,
                            opponent_first: bool, trick_room: bool = False) -> bool:
        """
        Registra que el rival actuo antes o despues que un Pokemon propio, con
        movimientos de la misma prioridad.

        Si el rival actuo primero su velocidad efectiva es >= la propia (>= y no >
        porque los empates de velocidad se resuelven al azar). Con Trick Room la
        comparacion se invierte. Se descarta la observacion si algun modificador
        del contexto no es afirmable con certeza.

        Devuelve True si el conjunto de candidatos se estrecho.
        """
        if opponent_ctx.is_uncertain or user_ctx.is_uncertain or user_speed <= 0:
            return False

        belief = self.belief_for(opponent_pkmn)
        if belief is None:
            return False

        base_spe = belief.base_stats.get('spe')
        if not base_spe:
            return False

        # Quien actua primero tiene velocidad >= la del otro (>= y no >, porque un
        # empate exacto se resuelve al azar y el que gano pudo ir igualado).
        # Ambas direcciones son inclusivas, asi que no son complementarias entre si.
        allowed = set()
        margin = _SPEED_COMPARE_MARGIN
        for sp in belief.candidates['spe']:
            stat = champions_stat(base_spe, sp, belief.nature, 'spe')
            speed = opponent_ctx.speed_of(stat)
            # Cotas inclusivas + margen: ante desync menor no expulsar el Spe real.
            if trick_room:
                ok = (
                    speed <= user_speed + margin
                    if opponent_first
                    else speed >= user_speed - margin
                )
            else:
                ok = (
                    speed >= user_speed - margin
                    if opponent_first
                    else speed <= user_speed + margin
                )
            if ok:
                allowed.add(sp)

        belief.observations += 1
        changed = belief.narrow('spe', allowed)
        if changed:
            rng = belief.stat_range('spe')
            print(
                f'[STAT-INF] {belief.species} Spe -> {rng[0]}-{rng[1]} '
                f'({len(belief.candidates["spe"])} SP posibles)'
            )
        return changed

    # -- evidencia de dano ------------------------------------------------ #

    def observe_damage(
        self,
        *,
        opponent_pkmn,
        move_name: str,
        category: str,
        opponent_is_attacker: bool,
        observed_damage: int,
        damage_is_percent: bool,
        known_stats: Dict[str, int],
        known_species: str,
        known_nature: Optional[str],
        ctx: DamageContext,
        known_hp: Optional[int] = None,
        is_ko: bool = False,
    ) -> bool:
        """
        Estrecha Atk/SpA (si el rival ataca) o Def/SpD (si el rival recibe) segun
        el dano observado y los rolls de damage-calc Champions.

        Con dano en % HP del rival (protocolo Showdown), tambien estrecha HP:
        el mismo golpe debe reproducirse para el par (Def|SpD, HP).

        known_stats: stats finales del Pokemon propio involucrado (atk/def/spa/spd/spe/hp).
        observed_damage: puntos de HP (absolutos) o delta de porcentaje (si damage_is_percent).
        is_ko: el golpe dejo al defensor a 0 HP; el dano observado es cota inferior.
        """
        if category not in ('p', 's') or observed_damage <= 0:
            return False
        if ctx.is_uncertain:
            return False

        belief = self.belief_for(opponent_pkmn)
        if belief is None:
            return False

        offen_stat = 'atk' if category == 'p' else 'spa'
        def_stat = 'def' if category == 'p' else 'spd'
        # Wonder Room intercambia Def/SpD efectivos en el calc via field flag;
        # el stat que estrechos sigue siendo el del move (fisico->def).
        unknown_stat = offen_stat if opponent_is_attacker else def_stat

        if unknown_stat not in belief.base_stats:
            return False

        # Cuando el rival defiende, sus boosts defensivos vienen del ctx;
        # cuando ataca, los ofensivos. El lado propio usa known + boosts del ctx
        # en el otro rol.
        if opponent_is_attacker:
            opp_boosts = {
                'atk': ctx.atk_boost,
                'def': 0,
                'spa': ctx.spa_boost,
                'spd': 0,
                'spe': 0,
            }
            known_boosts = {
                'atk': 0,
                'def': ctx.def_boost,
                'spa': 0,
                'spd': ctx.spd_boost,
                'spe': 0,
            }
            atk_item, def_item = ctx.attacker_item, ctx.defender_item
            atk_abi, def_abi = ctx.attacker_ability, ctx.defender_ability
            atk_status, def_status = ctx.attacker_status, ctx.defender_status
        else:
            opp_boosts = {
                'atk': 0,
                'def': ctx.def_boost,
                'spa': 0,
                'spd': ctx.spd_boost,
                'spe': 0,
            }
            known_boosts = {
                'atk': ctx.atk_boost,
                'def': 0,
                'spa': ctx.spa_boost,
                'spd': 0,
                'spe': 0,
            }
            atk_item, def_item = ctx.attacker_item, ctx.defender_item
            atk_abi, def_abi = ctx.attacker_ability, ctx.defender_ability
            atk_status, def_status = ctx.attacker_status, ctx.defender_status

        # Stats "conocidas" del rival en los ejes que no estamos estimando:
        # usamos el punto medio del rango actual (no poda: solo para rellenar el calc).
        def _mid_stat(stat: str) -> int:
            rng = belief.stat_range(stat)
            if rng:
                return (rng[0] + rng[1]) // 2
            base = belief.base_stats.get(stat, 1)
            return champions_stat(base, 0, belief.nature, stat)

        def _opp_stats_for(sp_unknown: int, hp_sp: Optional[int] = None) -> Dict[str, int]:
            stats = {s: _mid_stat(s) for s in ('atk', 'def', 'spa', 'spd', 'spe')}
            base_u = belief.base_stats[unknown_stat]
            stats[unknown_stat] = champions_stat(base_u, sp_unknown, belief.nature, unknown_stat)
            if 'hp' in belief.base_stats:
                hp_candidates = belief.candidates.get('hp') or {0}
                use_hp = hp_sp if hp_sp is not None else min(hp_candidates)
                stats['hp'] = champions_stat(
                    belief.base_stats['hp'], use_hp, belief.nature, 'hp'
                )
            else:
                stats['hp'] = known_hp or 100
            return stats

        known_payload_stats = {
            'hp': int(known_stats.get('hp') or known_hp or 100),
            'atk': int(known_stats.get('atk') or 1),
            'def': int(known_stats.get('def') or 1),
            'spa': int(known_stats.get('spa') or 1),
            'spd': int(known_stats.get('spd') or 1),
            'spe': int(known_stats.get('spe') or 1),
        }

        field = _field_payload(ctx, category)
        move_label = move_name  # el worker resuelve por nombre

        # Inferir HP solo cuando el rival es el defensor y el protocolo da % HP.
        infer_hp = (
            damage_is_percent
            and not opponent_is_attacker
            and 'hp' in belief.base_stats
        )
        hp_sp_list: list
        if infer_hp:
            # Un representante SP por HP final unico (barato y completo).
            by_final_hp: Dict[int, int] = {}
            for hp_sp in sorted(belief.candidates['hp']):
                fh = champions_stat(
                    belief.base_stats['hp'], hp_sp, belief.nature, 'hp'
                )
                by_final_hp.setdefault(fh, hp_sp)
            hp_sp_list = list(by_final_hp.values())
        elif damage_is_percent and 'hp' in belief.base_stats:
            hp_sps = sorted(belief.candidates['hp'])
            mid = hp_sps[len(hp_sps) // 2]
            hp_sp_list = sorted({hp_sps[0], mid, hp_sps[-1]})
        else:
            hp_sp_list = [None]

        # Coalesce: (final_unknown, final_hp) -> SP unknown + SP hp que mapean ahi.
        groups: Dict[Tuple[int, int], Dict[str, Set[int]]] = {}
        group_payloads: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for sp in list(belief.candidates[unknown_stat]):
            for hp_sp in hp_sp_list:
                opp_stats = _opp_stats_for(sp, hp_sp)
                key = (opp_stats[unknown_stat], opp_stats['hp'])
                bucket = groups.setdefault(key, {'unknown': set(), 'hp': set()})
                bucket['unknown'].add(sp)
                if hp_sp is not None:
                    bucket['hp'].add(hp_sp)
                if key in group_payloads:
                    continue
                if opponent_is_attacker:
                    atk_cur = None
                    def_frac = ctx.defender_hp_fraction
                    def_cur = (
                        max(1, int(round(known_payload_stats['hp'] * def_frac)))
                        if def_frac is not None else None
                    )
                    atk_species = getattr(belief, 'current_forme', None) or belief.species
                    attacker = _poke_payload(
                        atk_species,
                        belief.nature,
                        opp_stats,
                        item=atk_item,
                        ability=atk_abi,
                        status=atk_status,
                        boosts=opp_boosts,
                        cur_hp=atk_cur,
                        ability_on=ctx.attacker_ability_on,
                        types=ctx.attacker_types,
                        tera_type=ctx.attacker_tera,
                        allies_fainted=ctx.attacker_allies_fainted,
                    )
                    defender = _poke_payload(
                        known_species,
                        known_nature,
                        known_payload_stats,
                        item=def_item,
                        ability=def_abi,
                        status=def_status,
                        boosts=known_boosts,
                        cur_hp=def_cur,
                        ability_on=ctx.defender_ability_on,
                        types=ctx.defender_types,
                        tera_type=ctx.defender_tera,
                    )
                else:
                    def_frac = ctx.defender_hp_fraction
                    def_cur = (
                        max(1, int(round(opp_stats['hp'] * def_frac)))
                        if def_frac is not None else None
                    )
                    attacker = _poke_payload(
                        known_species,
                        known_nature,
                        known_payload_stats,
                        item=atk_item,
                        ability=atk_abi,
                        status=atk_status,
                        boosts=known_boosts,
                        ability_on=ctx.attacker_ability_on,
                        types=ctx.attacker_types,
                        tera_type=ctx.attacker_tera,
                        allies_fainted=ctx.attacker_allies_fainted,
                    )
                    defender = _poke_payload(
                        getattr(belief, 'current_forme', None) or belief.species,
                        belief.nature,
                        opp_stats,
                        item=def_item,
                        ability=def_abi,
                        status=def_status,
                        boosts=opp_boosts,
                        cur_hp=def_cur,
                        ability_on=ctx.defender_ability_on,
                        types=ctx.defender_types,
                        tera_type=ctx.defender_tera,
                    )
                group_payloads[key] = {
                    'attacker': attacker,
                    'defender': defender,
                    'move': move_label,
                    'field': field,
                    'isCrit': False,
                    '_hp': opp_stats['hp'],
                }

        if not group_payloads:
            return False

        keys_ordered = list(group_payloads.keys())
        batch_reqs = [{k: v for k, v in group_payloads[key].items() if not k.startswith('_')}
                      for key in keys_ordered]
        rolls_list = damage_ranges_batch(batch_reqs)

        # Si algun calc falla (especie/item ausente en dex), no podar: un None
        # tratado como "no encaja" expulsaria SP validos.
        if any(rolls is None for rolls in rolls_list):
            return False

        allowed_u: Set[int] = set()
        allowed_hp: Set[int] = set()
        any_calc = False
        for key, rolls in zip(keys_ordered, rolls_list):
            if rolls is None:
                continue
            any_calc = True
            min_roll, max_roll = rolls
            hp_val = group_payloads[key]['_hp']
            if damage_is_percent:
                ok = _rolls_match_percent(
                    min_roll, max_roll, hp_val, observed_damage, is_ko=is_ko
                )
            else:
                ok = _rolls_match_absolute(
                    min_roll, max_roll, observed_damage, is_ko=is_ko
                )
            if ok:
                allowed_u.update(groups[key]['unknown'])
                allowed_hp.update(groups[key]['hp'])

        if not any_calc:
            return False

        if not allowed_u:
            # Ningun SP reproduce el dano: premisa dudosa (sash, desync, etc.).
            belief.skipped += 1
            return False

        belief.observations += 1
        changed = belief.narrow(unknown_stat, allowed_u)
        if infer_hp and allowed_hp:
            changed = belief.narrow('hp', allowed_hp) or changed
        if changed:
            rng = belief.stat_range(unknown_stat)
            label = {'atk': 'Atk', 'def': 'Def', 'spa': 'SpA', 'spd': 'SpD'}[unknown_stat]
            extra = ''
            if infer_hp:
                hp_rng = belief.stat_range('hp')
                if hp_rng:
                    extra = (
                        f' | HP {hp_rng[0]}-{hp_rng[1]} '
                        f'({len(belief.candidates["hp"])} SP)'
                    )
            print(
                f'[STAT-INF] {belief.species} {label} -> {rng[0]}-{rng[1]} '
                f'({len(belief.candidates[unknown_stat])} SP posibles){extra}'
            )
        return changed
