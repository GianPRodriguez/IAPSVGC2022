import json
import math
import os
import constants
import re

from copy import deepcopy
from collections import defaultdict
from typing import Optional
from teams.team_converter import normalize_name
from showdown.battle import Pokemon
from showdown.battle import Battle
from showdown.mega_abilities import MEGA_FORME_ABILITIES
from showdown.stat_inference import SpeedContext, move_priority, move_category, DamageContext



def _parse_hp_and_status_from_switch(hp_status_str: str):
    """
    Extrae (hpa, hpb, status_dict) desde la cadena de condición de un |switch| o |drag|.
    Ejemplos de entrada:
      '190/202'       -> (190, 202, {})
      '28/198 par'    -> (28, 198, {'par': True})
      '62/100 brn'    -> (62, 100, {'brn': True})
      '0 fnt'         -> (0, 0, {})
      '0/100'         -> (0, 0, {})
    Retorna hpa=-1 si la condición está ausente/ilegible (no forzar KO).
    """
    hp_status_str = (hp_status_str or '').strip()
    if not hp_status_str:
        return -1, -1, {}

    # Debilitado: '0 fnt', '0/100', o exactamente '0'
    if (
        constants.FNT in hp_status_str
        or hp_status_str == '0'
        or hp_status_str.startswith('0/')
    ):
        return 0, 0, {}

    parts = hp_status_str.split('/')
    if len(parts) >= 2:
        # Extraemos HPA actual
        hpa_digits = re.findall(r'\d+', parts[0])
        hpa_val = int(hpa_digits[0]) if hpa_digits else 0

        # Extraemos HPB base y posible estado
        rest_parts = parts[1].strip().split()
        hpb_digits = re.findall(r'\d+', rest_parts[0]) if rest_parts else []
        hpb_val = int(hpb_digits[0]) if hpb_digits else 0

        # Si hay texto adicional (ej: 'par', 'tox', 'brn'), lo extraemos como estado no volátil
        non_volatile_dict = defaultdict(lambda: 0)
        if len(rest_parts) > 1:
            status_tag = rest_parts[1].lower().strip()
            if status_tag not in ("fnt", ""):
                non_volatile_dict[status_tag] = True

        return hpa_val, hpb_val, non_volatile_dict

    return -1, -1, {}


def _apply_switch_condition(pkmn, hpa_val: int, hpb_val: int, new_non_volatile) -> None:
    """Aplica HP/estado del |switch|/|drag| y sincroniza is_fainted."""
    if hpb_val > 0:
        pkmn.HPB_battle_base = hpb_val
    if hpa_val > 0:
        pkmn.HPA_battle_actual = hpa_val
        pkmn.is_fainted = False
    elif hpa_val == 0:
        pkmn.HPA_battle_actual = 0
        pkmn.is_fainted = True
    # hpa_val < 0 → condición desconocida: no tocar faint/HP
    if new_non_volatile:
        pkmn.non_volatile_statuses = new_non_volatile


# Estados mayores mutuamente excluyentes en Showdown / Pokémon.
_MAJOR_NON_VOLATILE = frozenset({'brn', 'par', 'slp', 'frz', 'psn', 'tox'})

def _canonical_species_name(name: str) -> str:
    """Normaliza nombres variantes para que Zoroark y Zoroark-Hisui se consideren la misma especie."""
    if not name:
        return ''

    normalized = normalize_name(str(name)).lower().replace(' ', '-').replace('.', '')
    if normalized.startswith('zoroark'):
        return 'zoroark'
    # Mr. Mime / Mime Jr. / Mr. Rime
    compact = normalized.replace('-', '')
    if compact in ('mrmime', 'mimejr', 'mrrime'):
        return compact
    return normalized


_FORME_SUFFIXES = (
    '-mega-x',
    '-mega-y',
    '-mega',
    '-gmax',
    '-eternamax',
    '-blade',
    '-shield',
    '-busted',
    '-disguised',
)


def _base_species_name(name: str) -> str:
    """Especie base sin formas (mega, gmax, aegislash-blade, etc.) para deducir Illusion."""
    normalized = _canonical_species_name(name)
    for suffix in _FORME_SUFFIXES:
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _party_species_key(name: str) -> str:
    """Clave de party para |switch|/|drag|: Mega/Blade colapsan; Floette-Eternal ≡ Floette-Mega."""
    base = _base_species_name(name)
    compact = base.replace('-', '').replace('_', '')
    if compact.startswith('floette'):
        return 'floette'
    return base


def _same_party_species(left: str, right: str) -> bool:
    a, b = _party_species_key(left), _party_species_key(right)
    return bool(a and b and a == b)


def _apply_switch_identity(pkmn, namePkmn: str) -> None:
    """Aplica el nombre/forme del log al mon que entra (Mega vs base, Eternal, etc.)."""
    if pkmn is None or not namePkmn:
        return
    pkmn.name = namePkmn
    key = _canonical_species_name(namePkmn)
    if 'mega' in key.replace('-', ''):
        pkmn.is_mega_evo = True
        mega_ability = _mega_forme_ability(namePkmn)
        if mega_ability:
            pkmn.abilityPkmn = mega_ability
        try:
            from API.PokemonData import getPkmnInfo
            info = getPkmnInfo(namePkmn)
            if info:
                pkmn.ATK_battle_base = int(info[3] or 0)
                pkmn.DEF_battle_base = int(info[4] or 0)
                pkmn.SPA_battle_base = int(info[5] or 0)
                pkmn.SPD_battle_base = int(info[6] or 0)
                pkmn.VEL_battle_base = int(info[7] or 0)
        except Exception:
            pass
    else:
        pkmn.is_mega_evo = False


def _active_status_keys(status_dict) -> set:
    status_dict = status_dict or {}
    return {k for k, v in status_dict.items() if v}


def _mega_forme_ability(forme_name: str) -> str:
    key = _canonical_species_name(forme_name)
    hit = MEGA_FORME_ABILITIES.get(key, '')
    if hit:
        return hit
    # Showdown: floette-eternal-mega → Champions floette-mega
    try:
        from API.PokemonData import champions_species_key
        resolved = champions_species_key(key)
        if resolved and resolved != key:
            return MEGA_FORME_ABILITIES.get(resolved, '')
    except Exception:
        pass
    return ''


def _count_side_fainted(side) -> int:
    """Cuántos del party (activos+reserva) están fainted — Last Respects / Supreme Overlord."""
    n = 0
    for p in _iter_side_pokemon(side, include_opponents=False):
        if getattr(p, 'is_fainted', False) or int(getattr(p, 'HPA_battle_actual', 1) or 0) <= 0:
            n += 1
    return n


def _iter_side_pokemon(side, include_opponents: bool = False):
    candidates = [
        getattr(side, 'active1', None),
        getattr(side, 'active2', None),
        *getattr(side, 'reserved', []),
    ]
    if include_opponents:
        candidates.extend(getattr(side, 'opponents', []))
    return [p for p in candidates if p is not None]


def _pokemon_state_differs(pkmn, hpa_val: int, status_dict) -> bool:
    """True si HP% o estados no volátiles discrepan del fingerprint entrante."""
    incoming_status = _active_status_keys(status_dict)
    known_status = _active_status_keys(getattr(pkmn, 'non_volatile_statuses', {}))
    status_differs = incoming_status != known_status
    hp_differs = (
        hpa_val > 0
        and getattr(pkmn, 'HPA_battle_actual', None) is not None
        and getattr(pkmn, 'HPA_battle_actual', None) != hpa_val
        and not getattr(pkmn, 'is_fainted', False)
    )
    return bool(status_differs or hp_differs)


def _species_exists_on_side(side, species_name: str, exclude=None, ignore_active_slot=None, include_opponents: bool = False) -> bool:
    """Devuelve True si el Pokémon ya está presente en el lado actual.

    La lista de preview ('opponents') NO debe bloquear la reserva del Pokémon que acaba de salir,
    porque ahí se listan todos los miembros del equipo aunque todavía no hayan sido revelados.
    """
    if not species_name:
        return False

    species = _canonical_species_name(species_name)
    candidates = [
        getattr(side, 'active1', None),
        getattr(side, 'active2', None),
        *getattr(side, 'reserved', []),
    ]
    if include_opponents:
        candidates.extend(getattr(side, 'opponents', []))

    for candidate in candidates:
        if candidate is None or candidate is exclude:
            continue
        if ignore_active_slot is not None and candidate is ignore_active_slot:
            continue
        if _canonical_species_name(getattr(candidate, 'name', '')) == species:
            return True

    return False


def _remove_exact_reserve_entry(side, pkmn) -> None:
    """Elimina solo la referencia exacta de una entrada en reserva, sin borrar otras especies confirmadas."""
    if pkmn is None:
        return

    side.reserved = [
        p for p in getattr(side, 'reserved', [])
        if p is not pkmn
    ]


def _keep_highest_hp_reserve_for_species(side, species_name: str) -> None:
    """Conserva una sola reserva de la especie indicada, priorizando su mayor HP."""
    species = _canonical_species_name(species_name)
    matching = [
        p for p in getattr(side, 'reserved', [])
        if p is not None and _canonical_species_name(getattr(p, 'name', '')) == species
    ]
    if len(matching) < 2:
        return

    survivor = max(matching, key=lambda p: getattr(p, 'HPA_battle_actual', 0))
    side.reserved = [
        p for p in side.reserved
        if p not in matching or p is survivor
    ]


def _status_dict_matches(left, right) -> bool:
    """Compara estados no volátiles activos (par, tox, brn, etc.)."""
    return _active_status_keys(left) == _active_status_keys(right)


def _find_illusion_zoroark_reserve_index(side, incoming_name: str, hpa_val: int, new_non_volatile) -> int:
    """
    Detecta si un |switch|/|drag| con apariencia de otra especie es Illusion.
    Criterios (como un jugador humano):
      - Fingerprint completo HP% + estado = Zoroark conocido en reserva.
      - Ya existe la especie real con HP% y/o estado discordantes, y el entrante
        encaja con el Zoroark (por estado, por HP%, o por ambos).
    """
    incoming_species = _base_species_name(incoming_name)
    if not incoming_species or incoming_species == 'zoroark':
        return -1

    zoroark_entries = [
        (i, p) for i, p in enumerate(getattr(side, 'reserved', []))
        if p is not None and _is_zoroark(p) and not getattr(p, 'is_fainted', False)
    ]
    if not zoroark_entries:
        return -1

    real_copies = [
        p for p in _iter_side_pokemon(side)
        if not _is_zoroark(p)
        and _base_species_name(getattr(p, 'name', '')) == incoming_species
        and not getattr(p, 'is_fainted', False)
    ]
    incoming_status = _active_status_keys(new_non_volatile)

    for idx, zoroark in zoroark_entries:
        z_hp = getattr(zoroark, 'HPA_battle_actual', None)
        z_status = _active_status_keys(getattr(zoroark, 'non_volatile_statuses', {}))
        hp_match = hpa_val > 0 and z_hp == hpa_val
        status_match = incoming_status == z_status
        contradicts_real = any(
            _pokemon_state_differs(p, hpa_val, new_non_volatile) for p in real_copies
        )

        # Caso fuerte: el entrante es idéntico al Zoroark ya fichado (HP + estado).
        # Incluye Blastoise-Mega a 62/par cuando el Blastoise real está a 73 limpio.
        if hp_match and status_match:
            if not real_copies or contradicts_real or bool(z_status) or (z_hp is not None and z_hp < 100):
                return idx

        # Deducción por estado: Zoroark quemado/paralizado/dormido y la copia real no.
        if real_copies and status_match and bool(z_status) and contradicts_real:
            return idx

        # Deducción por HP%: mismo % que Zoroark y distinto al de la especie real.
        if real_copies and hp_match and contradicts_real:
            return idx

    return -1


def _resolve_duplicate_illusion_reserves(side) -> None:
    """
    Elimina disfraces residuales en reserva cuando ya está el Zoroark real.
    Ej.: dragonite 62/par + zoroark 62/par → se elimina el dragonite espejo.
    También por solo HP% si hay otra copia más sana de esa especie.
    """
    reserved = getattr(side, 'reserved', [])
    zoroarks = [
        p for p in reserved
        if p is not None and _is_zoroark(p) and not getattr(p, 'is_fainted', False)
    ]
    if not zoroarks:
        return

    to_remove = []
    for pkmn in reserved:
        if pkmn is None or _is_zoroark(pkmn):
            continue
        species = _base_species_name(getattr(pkmn, 'name', ''))
        healthier_same = any(
            p is not pkmn
            and p is not None
            and not _is_zoroark(p)
            and _base_species_name(getattr(p, 'name', '')) == species
            and (
                getattr(p, 'HPA_battle_actual', 0) > getattr(pkmn, 'HPA_battle_actual', 0)
                or (
                    not _active_status_keys(getattr(p, 'non_volatile_statuses', {}))
                    and _active_status_keys(getattr(pkmn, 'non_volatile_statuses', {}))
                )
            )
            for p in reserved
        )
        for zoroark in zoroarks:
            same_hp = (
                getattr(pkmn, 'HPA_battle_actual', None)
                == getattr(zoroark, 'HPA_battle_actual', None)
            )
            same_status = _status_dict_matches(
                getattr(pkmn, 'non_volatile_statuses', {}),
                getattr(zoroark, 'non_volatile_statuses', {}),
            )
            z_has_status = bool(_active_status_keys(getattr(zoroark, 'non_volatile_statuses', {})))
            if same_hp and same_status and (z_has_status or healthier_same):
                to_remove.append(pkmn)
                break

    if to_remove:
        side.reserved = [p for p in reserved if p not in to_remove]


def _find_party_pokemon_for_heal(side, species_name: str, prefer_fainted: bool = False):
    """Busca un Pokémon del bando por especie (activos + reserva), útil para Revival Blessing."""
    species = _canonical_species_name(species_name)
    if not species:
        return None

    candidates = [
        p for p in (
            getattr(side, 'active1', None),
            getattr(side, 'active2', None),
            *getattr(side, 'reserved', []),
        )
        if p is not None and _canonical_species_name(getattr(p, 'name', '')) == species
    ]
    if not candidates:
        return None

    if prefer_fainted:
        fainted = [p for p in candidates if getattr(p, 'is_fainted', False) or getattr(p, 'HPA_battle_actual', 0) <= 0]
        if fainted:
            return fainted[0]

    return candidates[0]


def _clear_reserve_effects(pkmn: Pokemon) -> Pokemon:
    """Limpia efectos temporales al guardar un Pokémon en la reserva."""
    if pkmn is None:
        return None

    pkmn.volatile_statuses = defaultdict(lambda: 0)
    pkmn.boosts = defaultdict(lambda: 0)
    # Yawn / Perish Song se limpian al salir del campo.
    pkmn.is_drowsy = False
    pkmn.perish_count = 0
    _clear_all_field_temp_flags(pkmn)
    pkmn.ability_on = False
    pkmn.just_switched = False
    # Illusion: el disfraz solo aplica mientras está en campo
    pkmn.illusion_disguise = None
    pkmn.illusion_protocol_revealed = False
    # Guard Split termina al salir (como en PS)
    pkmn.guard_split_active = False
    if getattr(pkmn, 'is_opponent', False):
        pkmn.DEF_battle_actual = -1
        pkmn.SPD_battle_actual = -1
    return pkmn


# --------------------------------------------------------------------------- #
# Flags efímeros de turno (Protect, Helping Hand, Destiny Bond, Charge, …)
# --------------------------------------------------------------------------- #

_PROTECT_SINGLETURN = frozenset(getattr(
    constants, 'PROTECT_SINGLETURN_IDS',
    {'protect', 'detect', 'spikyshield', 'banefulbunker', 'kingsshield',
     'obstruct', 'silktrap', 'burningbulwark', 'maxguard'},
))
_HELPING_HAND_IDS = frozenset({'helpinghand'})
_DESTINY_BOND_IDS = frozenset({'destinybond'})
_CHARGE_IDS = frozenset({'charge'})
_REDIRECT_SINGLETURN = frozenset({'followme', 'ragepowder', 'spotlight'})
_EPHEMERAL_TURN_VOLATILES = (
    _PROTECT_SINGLETURN | _HELPING_HAND_IDS | _REDIRECT_SINGLETURN
    | frozenset({'roost', 'focuspunch', 'beakblast', 'shelltrap'})
)

# Moves Eléctricos (Champions/gen9) para consumir Charge. Preferible a moves.json
# (no está en el repo); si aparece type en un JSON futuro, _move_is_electric lo usa.
_ELECTRIC_MOVE_IDS = frozenset({
    'thunder', 'thunderwave', 'thunderbolt', 'thunderpunch', 'thundershock',
    'zapcannon', 'spark', 'volttackle', 'shockwave', 'chargebeam', 'discharge',
    'thunderfang', 'electroball', 'voltswitch', 'boltstrike', 'electroweb',
    'fusionbolt', 'wildcharge', 'paraboliccharge', 'nuzzle', 'plasmafists',
    'zingzap', 'aurawheel', 'boltbeak', 'overdrive', 'risingvoltage',
    'thundercage', 'doubleshock', 'electrodrift', 'electroshot', 'supercellslam',
    'thunderclap', 'wildboltstorm',
})

_MOVES_JSON_CACHE: Optional[dict] = None


def _moves_json() -> dict:
    global _MOVES_JSON_CACHE
    if _MOVES_JSON_CACHE is None:
        try:
            path = os.path.join('API', 'JSONs', 'moves.json')
            with open(path, encoding='utf-8') as handle:
                _MOVES_JSON_CACHE = json.load(handle)
        except Exception:
            _MOVES_JSON_CACHE = {}
    return _MOVES_JSON_CACHE


def _move_type_name(move_name: Optional[str]) -> Optional[str]:
    mid = normalize_name(move_name or '')
    if not mid:
        return None
    entry = _moves_json().get(mid) or {}
    t = entry.get('type')
    return str(t) if t else None


def _move_is_electric(move_name: Optional[str]) -> bool:
    t = _move_type_name(move_name)
    if t:
        return t.lower() == 'electric'
    return normalize_name(move_name or '') in _ELECTRIC_MOVE_IDS


# Targets PS que golpean a más de un activo (daño / Inferencia isSpread).
_SPREAD_MOVE_TARGETS = frozenset({
    'alladjacent',
    'alladjacentfoes',
})


def _move_is_spread(move_name: Optional[str], tags: str = '') -> bool:
    """True si el |move| es spread ([spread] en protocolo o target allAdjacent*)."""
    if '[spread]' in (tags or '').lower():
        return True
    mid = normalize_name(move_name or '')
    if not mid:
        return False
    entry = _moves_json().get(mid) or {}
    target = str(entry.get('target') or '').replace(' ', '').lower()
    return target in _SPREAD_MOVE_TARGETS


def _clear_ephemeral_turn_flags(pkmn: Pokemon) -> None:
    """Protect / Helping Hand / roost-like: mueren al cerrar el turno."""
    if pkmn is None:
        return
    had_roost = False
    vs = getattr(pkmn, 'volatile_statuses', None)
    if isinstance(vs, (dict, defaultdict)):
        had_roost = bool(vs.get('roost'))
    pkmn.is_protected = False
    pkmn.is_helping_hand = False
    pkmn.is_redirecting = False
    if isinstance(vs, (dict, defaultdict)):
        for key in list(vs.keys()):
            if key in _EPHEMERAL_TURN_VOLATILES:
                vs.pop(key, None)
    if had_roost:
        try:
            from API.PokemonData import getPkmnInfo
            info = getPkmnInfo(pkmn.name)
            if info:
                t1, t2, *_rest = info
                pkmn.type1, pkmn.type2 = t1, t2
        except Exception:
            pass


def _clear_all_field_temp_flags(pkmn: Pokemon, *, entering: bool = False) -> None:
    """Limpia Protect/HH/Destiny/Charge al entrar, salir o faint."""
    if pkmn is None:
        return
    _clear_ephemeral_turn_flags(pkmn)
    pkmn.is_destiny_bonded = False
    pkmn.is_charged = False
    pkmn.must_recharge = False
    if isinstance(getattr(pkmn, 'volatile_statuses', None), (dict, defaultdict)):
        pkmn.volatile_statuses.pop('destinybond', None)
        pkmn.volatile_statuses.pop('charge', None)
        pkmn.volatile_statuses.pop('mustrecharge', None)
    if entering:
        pkmn.ability_on = False
        pkmn.just_switched = True


def _clear_all_ephemeral_turn_flags(battle: Battle) -> None:
    for side in (battle.user, battle.opponent):
        for attr in ('active1', 'active2'):
            _clear_ephemeral_turn_flags(getattr(side, attr, None))


def _set_protected(pkmn: Pokemon, value: bool, volatile_id: str = 'protect') -> None:
    if pkmn is None:
        return
    pkmn.is_protected = bool(value)
    if not isinstance(getattr(pkmn, 'volatile_statuses', None), (dict, defaultdict)):
        pkmn.volatile_statuses = defaultdict(lambda: 0)
    if value:
        pkmn.volatile_statuses[volatile_id] = True
    else:
        for key in list(_PROTECT_SINGLETURN):
            pkmn.volatile_statuses.pop(key, None)


def _set_helping_hand(pkmn: Pokemon, value: bool) -> None:
    if pkmn is None:
        return
    pkmn.is_helping_hand = bool(value)
    if not isinstance(getattr(pkmn, 'volatile_statuses', None), (dict, defaultdict)):
        pkmn.volatile_statuses = defaultdict(lambda: 0)
    if value:
        pkmn.volatile_statuses['helpinghand'] = True
    else:
        pkmn.volatile_statuses.pop('helpinghand', None)


def _set_destiny_bonded(pkmn: Pokemon, value: bool) -> None:
    if pkmn is None:
        return
    pkmn.is_destiny_bonded = bool(value)
    if not isinstance(getattr(pkmn, 'volatile_statuses', None), (dict, defaultdict)):
        pkmn.volatile_statuses = defaultdict(lambda: 0)
    if value:
        pkmn.volatile_statuses['destinybond'] = True
    else:
        pkmn.volatile_statuses.pop('destinybond', None)


def _set_must_recharge(pkmn: Pokemon, value: bool) -> None:
    if pkmn is None:
        return
    pkmn.must_recharge = bool(value)
    if not isinstance(getattr(pkmn, 'volatile_statuses', None), (dict, defaultdict)):
        pkmn.volatile_statuses = defaultdict(lambda: 0)
    if value:
        pkmn.volatile_statuses['mustrecharge'] = True
    else:
        pkmn.volatile_statuses.pop('mustrecharge', None)


def _set_charged(pkmn: Pokemon, value: bool) -> None:
    if pkmn is None:
        return
    pkmn.is_charged = bool(value)
    if not isinstance(getattr(pkmn, 'volatile_statuses', None), (dict, defaultdict)):
        pkmn.volatile_statuses = defaultdict(lambda: 0)
    if value:
        pkmn.volatile_statuses['charge'] = True
    else:
        pkmn.volatile_statuses.pop('charge', None)


def _ally_has_friend_guard(side, defender) -> bool:
    """Friend Guard protege al aliado (no a sí mismo)."""
    for attr in ('active1', 'active2'):
        ally = getattr(side, attr, None)
        if ally is None or ally is defender:
            continue
        if getattr(ally, 'is_fainted', False) or int(getattr(ally, 'HPA_battle_actual', 0) or 0) <= 0:
            continue
        abi = normalize_name(getattr(ally, 'abilityPkmn', '') or '')
        if abi == 'friendguard':
            return True
    return False


def _ally_has_ability(side, self_mon, ability_ids) -> bool:
    want = {normalize_name(a) for a in ability_ids}
    for attr in ('active1', 'active2'):
        ally = getattr(side, attr, None)
        if ally is None or ally is self_mon:
            continue
        if getattr(ally, 'is_fainted', False) or int(getattr(ally, 'HPA_battle_actual', 0) or 0) <= 0:
            continue
        if normalize_name(getattr(ally, 'abilityPkmn', '') or '') in want:
            return True
    return False


def _active_has_ability(battle: Battle, ability_ids) -> bool:
    want = {normalize_name(a) for a in ability_ids}
    for side in (battle.user, battle.opponent):
        for attr in ('active1', 'active2'):
            mon = getattr(side, attr, None)
            if mon is None:
                continue
            if getattr(mon, 'is_fainted', False) or int(getattr(mon, 'HPA_battle_actual', 0) or 0) <= 0:
                continue
            if normalize_name(getattr(mon, 'abilityPkmn', '') or '') in want:
                return True
    return False


def _live_types(pkmn) -> Optional[list]:
    if pkmn is None:
        return None
    t1 = getattr(pkmn, 'type1', None)
    t2 = getattr(pkmn, 'type2', None)
    out = [str(t) for t in (t1, t2) if t]
    return out or None


def _apply_typechange(pkmn, raw_types: str) -> None:
    if pkmn is None or not raw_types:
        return
    parts = [p for p in re.split(r'[/,\s]+', str(raw_types).strip()) if p]
    if not parts:
        return
    pkmn.type1 = parts[0]
    pkmn.type2 = parts[1] if len(parts) > 1 else None


def _strip_flying_for_roost(pkmn) -> None:
    if pkmn is None:
        return
    for attr in ('type1', 'type2'):
        if str(getattr(pkmn, attr, '') or '').lower() == 'flying':
            setattr(pkmn, attr, None)
    if pkmn.type1 is None and pkmn.type2 is not None:
        pkmn.type1, pkmn.type2 = pkmn.type2, None
    if pkmn.type1 is None and pkmn.type2 is None:
        pkmn.type1 = 'Normal'


def _ability_on_for_calc(pkmn, *, side=None, foe=None) -> bool:
    """Valor abilityOn que espera @smogon/calc para esta habilidad."""
    if pkmn is None:
        return False
    abi = normalize_name(getattr(pkmn, 'abilityPkmn', '') or '')
    if abi in ('flashfire', 'slowstart', 'electromorphosis'):
        return bool(getattr(pkmn, 'ability_on', False))
    if abi in ('plus', 'minus') and side is not None:
        return _ally_has_ability(side, pkmn, ('plus', 'minus'))
    if abi == 'stakeout' and foe is not None:
        return bool(getattr(foe, 'just_switched', False))
    return bool(getattr(pkmn, 'ability_on', False))


def _hp_fraction_before_hit(pkmn) -> Optional[float]:
    """Fracción de HP del defensor antes del golpe (Multiscale)."""
    try:
        cur = float(getattr(pkmn, 'HPA_battle_actual', 0) or 0)
        mx = float(getattr(pkmn, 'HPB_battle_base', 0) or 0)
    except (TypeError, ValueError):
        return None
    if mx <= 0 or cur <= 0:
        return None
    # Rival suele reportar xx/100 → cur ya es %.
    if getattr(pkmn, 'is_opponent', False) and mx == 100:
        return max(0.0, min(1.0, cur / 100.0))
    return max(0.0, min(1.0, cur / mx))


def _set_redirecting(pkmn: Pokemon, value: bool, volatile_id: str = 'followme') -> None:
    if pkmn is None:
        return
    pkmn.is_redirecting = bool(value)
    if not isinstance(getattr(pkmn, 'volatile_statuses', None), (dict, defaultdict)):
        pkmn.volatile_statuses = defaultdict(lambda: 0)
    if value:
        # Solo un redirector efectivo por bando: limpia aliado.
        pkmn.volatile_statuses[volatile_id] = True
    else:
        for key in list(_REDIRECT_SINGLETURN):
            pkmn.volatile_statuses.pop(key, None)


def singleturn(battle: Battle, split_msg: list) -> Battle:
    # |-singleturn|p1a: Primarina|Protect
    # |-singleturn|p1a: Manectric|Helping Hand
    # |-singleturn|p2a: Talonflame|Roost
    # |-singleturn|p1b: Indeedee|Follow Me
    if len(split_msg) < 4:
        return battle

    raw = split_msg[3].split(':')[-1].strip()
    effect_id = normalize_name(raw)
    pkmn = _resolve_active_from_ident(battle, split_msg[2].strip())
    if pkmn is None:
        return battle

    if effect_id in _PROTECT_SINGLETURN:
        _set_protected(pkmn, True, effect_id)
        return battle
    if effect_id in _HELPING_HAND_IDS:
        _set_helping_hand(pkmn, True)
        return battle
    if effect_id in _CHARGE_IDS:
        _set_charged(pkmn, True)
        return battle
    if effect_id in _REDIRECT_SINGLETURN:
        # Nuevo Follow Me / Rage Powder: este mon redirige; limpia el otro activo.
        side = battle.opponent if getattr(pkmn, 'is_opponent', False) else battle.user
        for attr in ('active1', 'active2'):
            other = getattr(side, attr, None)
            if other is not None and other is not pkmn:
                _set_redirecting(other, False)
        _set_redirecting(pkmn, True, effect_id)
        return battle

    # Otros singleturn (Roost, Focus Punch…): solo registro en volatiles.
    if not isinstance(getattr(pkmn, 'volatile_statuses', None), (dict, defaultdict)):
        pkmn.volatile_statuses = defaultdict(lambda: 0)
    pkmn.volatile_statuses[effect_id] = True
    if effect_id == 'roost':
        _strip_flying_for_roost(pkmn)
    return battle


def singlemove(battle: Battle, split_msg: list) -> Battle:
    # |-singlemove|p1a: Gengar|Destiny Bond
    # |-singlemove|p2a: Shedinja|Grudge
    if len(split_msg) < 4:
        return battle

    raw = split_msg[3].split(':')[-1].strip()
    effect_id = normalize_name(raw)
    pkmn = _resolve_active_from_ident(battle, split_msg[2].strip())
    if pkmn is None:
        return battle

    if effect_id in _DESTINY_BOND_IDS:
        _set_destiny_bonded(pkmn, True)
        return battle

    if not isinstance(getattr(pkmn, 'volatile_statuses', None), (dict, defaultdict)):
        pkmn.volatile_statuses = defaultdict(lambda: 0)
    pkmn.volatile_statuses[effect_id] = True
    return battle


def _is_yawn_volatile_name(name: str) -> bool:
    """True para 'yawn' / 'moveyawn' (Showdown manda move: Yawn → normalize = yawn)."""
    key = normalize_name(name) if name else ''
    return key in ('yawn', 'moveyawn', constants.DROWSY)


def _set_drowsy(pkmn: Pokemon, value: bool) -> None:
    if pkmn is None:
        return
    pkmn.is_drowsy = bool(value)
    if not value and hasattr(pkmn, 'volatile_statuses') and isinstance(pkmn.volatile_statuses, dict):
        for key in ('yawn', 'moveyawn', constants.DROWSY):
            pkmn.volatile_statuses.pop(key, None)


def _parse_perish_count(volatile_status: str):
    """
    Showdown: |-start|pXa: Mon|perish3|[silent]
               |-start|pXa: Mon|perish2
               |-start|pXa: Mon|perish1
               |-start|pXa: Mon|perish0   (luego faint)
    Devuelve 0..3 o None si no es un mensaje perish.
    """
    if not volatile_status:
        return None
    match = re.fullmatch(r'perish([0-3])', str(volatile_status).lower().strip())
    if not match:
        return None
    return int(match.group(1))


def _set_perish_count(pkmn: Pokemon, count: int) -> None:
    if pkmn is None:
        return
    if not isinstance(getattr(pkmn, 'volatile_statuses', None), (dict, defaultdict)):
        pkmn.volatile_statuses = defaultdict(lambda: 0)

    # Limpia claves perish* previas para no acumular perish3+perish2.
    for key in list(pkmn.volatile_statuses.keys()):
        if str(key).startswith('perish'):
            del pkmn.volatile_statuses[key]

    if count and count > 0:
        pkmn.perish_count = count
        pkmn.volatile_statuses[f'perish{count}'] = True
        pkmn.volatile_statuses['perishsong'] = True
    else:
        # perish0 o limpieza: ya no hay contador activo útil para decisión.
        pkmn.perish_count = 0


def _clear_perish(pkmn: Pokemon) -> None:
    _set_perish_count(pkmn, 0)


def _is_zoroark(pkmn: Pokemon) -> bool:
    return bool(
        pkmn
        and (
            getattr(pkmn, 'is_zoroark', False)
            or _canonical_species_name(getattr(pkmn, 'name', '')).startswith('zoroark')
        )
    )


def _infer_zoroark_from_disguise(pkmn_leaving: Pokemon, side, opponents_list):
    """Devuelve la ficha de Zoroark si el disfraz contradice la reserva/activo conocido."""
    if pkmn_leaving is None or _is_zoroark(pkmn_leaving):
        return None

    disguised_species = _base_species_name(getattr(pkmn_leaving, 'name', ''))
    if not disguised_species:
        return None

    zoroark_template = next((
        p for p in opponents_list if _is_zoroark(p)
    ), None)
    if zoroark_template is None:
        # También aceptar Zoroark ya revelado en el propio bando.
        zoroark_template = next((
            p for p in _iter_side_pokemon(side) if _is_zoroark(p)
        ), None)
    if zoroark_template is None:
        return None

    leaving_hp = getattr(pkmn_leaving, 'HPA_battle_actual', None)
    leaving_status = getattr(pkmn_leaving, 'non_volatile_statuses', {})

    other_copies = [
        p for p in _iter_side_pokemon(side)
        if p is not pkmn_leaving
        and not _is_zoroark(p)
        and _base_species_name(getattr(p, 'name', '')) == disguised_species
        and not getattr(p, 'is_fainted', False)
    ]
    # Discordancia de estado o de % HP frente a otra copia de la misma especie.
    if any(_pokemon_state_differs(p, leaving_hp or 0, leaving_status) for p in other_copies):
        return zoroark_template

    return None


def _revert_special_pokemon(pkmn_leaving: Pokemon, side) -> Pokemon:
    """
    Revierte formas especiales al salir a reserva:
      - Ditto transformado → Ditto base
      - Illusion / Zoroark disfrazado → ficha Zoroark (si se puede deducir)
      - Aegislash-Blade → Aegislash (escudo)
    Conserva HPA_battle_actual y non_volatile_statuses.
    """
    if pkmn_leaving is None:
        return None

    opponents_list = getattr(side, 'opponents', getattr(side, 'not_confirmed', []))

    # =========================================================================
    # 1. REVERSIÓN DE DITTO
    # =========================================================================
    if getattr(pkmn_leaving, 'is_ditto', False):
        # Buscamos la ficha original de Ditto en la lista de opponents para recuperar sus datos reales
        ditto_template = None
        for p in opponents_list:
            if p and getattr(p, 'name', '').lower() == 'ditto':
                ditto_template = p
                break
        
        if ditto_template:
            reverted = deepcopy(ditto_template)
            reverted.HPA_battle_actual = pkmn_leaving.HPA_battle_actual
            reverted.HPB_battle_base = pkmn_leaving.HPB_battle_base
            reverted.non_volatile_statuses = deepcopy(pkmn_leaving.non_volatile_statuses)
            reverted.is_ditto = True
            reverted.is_zoroark = False
            if getattr(side, 'isUser', False):
                for stat_name in ('ATK', 'DEF', 'SPA', 'SPD', 'VEL'):
                    setattr(reverted, f'{stat_name}_battle_base', -1)
                    setattr(reverted, f'{stat_name}_battle_actual', getattr(pkmn_leaving, f'{stat_name}_battle_actual'))
            else:
                for stat_name in ('ATK', 'DEF', 'SPA', 'SPD', 'VEL'):
                    setattr(reverted, f'{stat_name}_battle_base', getattr(ditto_template, f'{stat_name}_battle_base'))
                    setattr(reverted, f'{stat_name}_battle_actual', -1)
            return _clear_reserve_effects(reverted)
        else:
            # Fallback manual por seguridad si no se encontrara el template
            pkmn_leaving.name = 'ditto'
            pkmn_leaving.is_ditto = True
            pkmn_leaving.is_zoroark = False
            if getattr(side, 'isUser', False):
                pkmn_leaving.ATK_battle_base = -1
                pkmn_leaving.DEF_battle_base = -1
                pkmn_leaving.SPA_battle_base = -1
                pkmn_leaving.SPD_battle_base = -1
                pkmn_leaving.VEL_battle_base = -1
            else:
                pkmn_leaving.ATK_battle_base = 48
                pkmn_leaving.DEF_battle_base = 48
                pkmn_leaving.SPA_battle_base = 48
                pkmn_leaving.SPD_battle_base = 48
                pkmn_leaving.VEL_battle_base = 48
                pkmn_leaving.ATK_battle_actual = -1
                pkmn_leaving.DEF_battle_actual = -1
                pkmn_leaving.SPA_battle_actual = -1
                pkmn_leaving.SPD_battle_actual = -1
                pkmn_leaving.VEL_battle_actual = -1
            pkmn_leaving.moves = ['transform', None, None, None]
            pkmn_leaving.movesPP = [16, -1, -1, -1]
            pkmn_leaving.abilityPkmn = 'imposter'
            pkmn_leaving.type1 = 'Normal'
            pkmn_leaving.type2 = None
            return _clear_reserve_effects(pkmn_leaving)

    # =========================================================================
    # 2. RESTAURACIÓN / INFERENCIA DE ZOROARK
    # =========================================================================
    inferred_zoroark = _infer_zoroark_from_disguise(pkmn_leaving, side, opponents_list)
    if _is_zoroark(pkmn_leaving) or inferred_zoroark:
        if inferred_zoroark:
            reverted = deepcopy(inferred_zoroark)
            reverted.name = inferred_zoroark.name
            reverted.is_zoroark = True
            reverted.HPA_battle_actual = pkmn_leaving.HPA_battle_actual
            reverted.HPB_battle_base = pkmn_leaving.HPB_battle_base
            reverted.non_volatile_statuses = deepcopy(pkmn_leaving.non_volatile_statuses)
            pkmn_leaving = reverted
        return _clear_reserve_effects(pkmn_leaving)

    # =========================================================================
    # 3. REVERSIÓN DE AEGISLASH (De Blade Form a Shield Form)
    # =========================================================================
    if getattr(pkmn_leaving, 'name', '').lower() == 'aegislash-blade':
        # Buscamos la ficha original de Aegislash en la lista de opponents
        aegislash_template = None
        for p in opponents_list:
            if p and getattr(p, 'name', '').lower() == 'aegislash':
                aegislash_template = p
                break
        
        if aegislash_template:
            reverted = deepcopy(aegislash_template)
            reverted.HPA_battle_actual = pkmn_leaving.HPA_battle_actual
            reverted.HPB_battle_base = pkmn_leaving.HPB_battle_base
            reverted.non_volatile_statuses = deepcopy(pkmn_leaving.non_volatile_statuses)
            return _clear_reserve_effects(reverted)
        else:
            # Fallback manual con los stats oficiales de Aegislash Shield Form (Gen 8/9 VGC)
            pkmn_leaving.name = 'aegislash'
            pkmn_leaving.ATK_battle_base = 50
            pkmn_leaving.DEF_battle_base = 140
            pkmn_leaving.SPA_battle_base = 50
            pkmn_leaving.SPD_battle_base = 140
            pkmn_leaving.VEL_battle_base = 60
            return _clear_reserve_effects(pkmn_leaving)

    return _clear_reserve_effects(pkmn_leaving)

############################################################################################################################################################
### OK - Devuelve el valor booleando indicativo si la actualización del entorno afecta al oponente o al usuario                                          ###
############################################################################################################################################################
def is_opponent(battle: Battle,  split_msg: list) -> bool:
    # split_msg = ["","switch","p1a: Incineroar","Incineroar, L50, M","190/202"]
    # split_msg[2] = "p1a: Incineroar"  |  sidestart: "p1: Nick"
    ident = (split_msg[2] if len(split_msg) > 2 else '').strip()
    slot = ident.split(':')[0].strip().lower()
    uid = (getattr(getattr(battle, 'user', None), 'id', None) or 'p1')
    uid = str(uid).strip().lower() or 'p1'
    # p1 / p1a / p1b → nuestro bando si user.id == p1
    return not slot.startswith(uid)


def _side_from_slot(battle: Battle, slot: str):
    """Resuelve p1/p1a/p1b vs p2/p2b al Side user u opponent."""
    slot_key = (slot or '').split(':')[0].strip().lower()
    uid = str(getattr(getattr(battle, 'user', None), 'id', None) or 'p1').strip().lower() or 'p1'
    return battle.user if slot_key.startswith(uid) else battle.opponent


def _is_opponent_slot(battle: Battle, slot: str) -> bool:
    slot_key = (slot or '').split(':')[0].strip().lower()
    uid = str(getattr(getattr(battle, 'user', None), 'id', None) or 'p1').strip().lower() or 'p1'
    return not slot_key.startswith(uid)


############################################################################################################################################################
### OK - Identifica el tipo de |request| y le asigna un valor positivo en caso sea el tipo correspondiente: active, forceSwitch & wait                   ###
############################################################################################################################################################
def request(battle: Battle, split_msg: list) -> Battle:
    # |request|{"active":[...],"side":{...},"rqid":"4"}
    # |request|{"forceSwitch":[true,false],"side":{...},"rqid":"14"}
    # |request|{"wait":true,"side":{...},"rqid":"5"}
    battle_json = json.loads(split_msg[2].strip('\'')) # Se divide el JSON por lineas en battle_json

    # Obtiene el rqid
    battle.rqid = battle_json[constants.RQID]

    # ¿battle_json tiene "forceSwitch"? Se activa cuando uso U-TURN
    if battle_json.get(constants.FORCE_SWITCH):
        battle.force_switch = True
    else:
        battle.force_switch = False

    # ¿battle_json tiene "active"? Se activa cuando uso U-TURN
    if battle_json.get(constants.ACTIVE):
        battle.active = True
    else:
        battle.active = False

    # ¿battle_json tiene "wait"? ¿Se activa cuando el oponente usa U-TURN?
    if battle_json.get(constants.WAIT):
        battle.wait = True
    else:
        battle.wait = False

    # Se almacena el JSON siempre (incl. wait) para depuración / paridad con Showdown
    battle.request_json = battle_json

    return battle


############################################################################################################################################################
### OK - Se realiza el cambio de un PKMN activo a uno de reserva                                                                                         ###
############################################################################################################################################################
def switch_or_drag(battle: Battle, split_msg: list) -> Battle:
    # |switch|p1a: Incineroar|Incineroar, L50, M|190/202
    # |switch|p2a: Dragonite|Dragonite, L50, M|28/198 par|[from] Flip Turn
    # |drag|p2b: Whimsicott|Whimsicott, L50, F|100/100
    # |switch|p2a: Blastoise|Blastoise-Mega, L50, M|62/100 par   (puede ser Illusion)
    if len(split_msg) < 4:
        return battle

    raw_name = split_msg[3].partition(",")[0].strip()
    if raw_name.lower() == 'mimikyu':
        raw_name = "mimikyu-disguised"
    namePkmn = constants.standardName.get(raw_name.lower(), raw_name.lower())

    opponent_turn = is_opponent(battle, split_msg)
    side = battle.opponent if opponent_turn else battle.user

    slot_str = split_msg[2].split(':')[0].strip()  # Ej: "p1a" o "p2b"
    is_active1 = slot_str.endswith('a')

    # =========================================================================
    # CASO 1: CAMBIO REALIZADO POR EL OPONENTE
    # =========================================================================
    if opponent_turn:
        current_active = side.active1 if is_active1 else side.active2

        # Extraemos la vida y estado no volátil reales mandados por el servidor
        hp_status_raw = split_msg[4].strip() if len(split_msg) > 4 else ""
        hpa_val, hpb_val, new_non_volatile = _parse_hp_and_status_from_switch(hp_status_raw)

        # 0. ILLUSION: si el entrante parece otra especie pero coincide con Zoroark
        # conocido (HP/estado) y ya existe la especie real, sacamos a Zoroark.
        illusion_idx = _find_illusion_zoroark_reserve_index(
            side, namePkmn, hpa_val, new_non_volatile
        )
        if illusion_idx != -1:
            pkmnAux = deepcopy(current_active)
            pkmnAux = _revert_special_pokemon(pkmnAux, side)

            new_active = deepcopy(side.reserved[illusion_idx])
            new_active.is_zoroark = True
            # PS sigue mostrando namePkmn (disfraz); guardamos para reporte dual
            new_active.illusion_disguise = _base_species_name(namePkmn)
            new_active.illusion_protocol_revealed = False
            _apply_switch_condition(new_active, hpa_val, hpb_val, new_non_volatile)
            new_active.volatile_statuses = defaultdict(lambda: 0)
            new_active.boosts = defaultdict(lambda: 0)
            new_active.is_drowsy = False
            _clear_all_field_temp_flags(new_active, entering=True)
            new_active.perish_count = 0

            if is_active1:
                side.active1 = new_active
            else:
                side.active2 = new_active

            if pkmnAux is not None:
                side.reserved[illusion_idx] = deepcopy(pkmnAux)
            else:
                side.reserved.pop(illusion_idx)

            _resolve_duplicate_illusion_reserves(side)
            print(
                f"El oponente ha cambiado de Pokémon a {new_active.name} "
                f"(Illusion deducida) ({new_active.HPA_battle_actual}/{new_active.HPB_battle_base}) : {slot_str}"
            )
            return battle

        # 1. PASO A: Buscamos PRIMERO si el Pokémon que entra YA ESTÁ en la reserva (reserved)
        # Preferir copia no-KO si hay varias de la misma especie.
        found_in_reserved_idx = -1
        fainted_reserved_idx = -1
        for contador, pkmn in enumerate(side.reserved):
            if pkmn is None or not _same_party_species(namePkmn, getattr(pkmn, 'name', '')):
                continue
            if getattr(pkmn, 'is_fainted', False) or int(getattr(pkmn, 'HPA_battle_actual', 1) or 0) <= 0:
                if fainted_reserved_idx < 0:
                    fainted_reserved_idx = contador
                continue
            found_in_reserved_idx = contador
            break
        if found_in_reserved_idx < 0:
            found_in_reserved_idx = fainted_reserved_idx

        # Si ya estaba en la reserva, hacemos un intercambio directo (Swap)
        if found_in_reserved_idx != -1:
            pkmnAux = deepcopy(current_active)
            pkmnAux = _revert_special_pokemon(pkmnAux, side)
            
            pkmn_from_reserved = side.reserved[found_in_reserved_idx]
            new_active = deepcopy(pkmn_from_reserved)
            _apply_switch_identity(new_active, namePkmn)

            # AUTORIDAD DEL LOG: HP / faint / status del mensaje de switch
            _apply_switch_condition(new_active, hpa_val, hpb_val, new_non_volatile)

            new_active.volatile_statuses = defaultdict(lambda: 0)
            new_active.boosts = defaultdict(lambda: 0)
            new_active.is_drowsy = False
            _clear_all_field_temp_flags(new_active, entering=True)
            new_active.perish_count = 0

            if is_active1:
                side.active1 = new_active
            else:
                side.active2 = new_active

            # El Pokémon saliente ocupa el lugar de la reserva que dejó el entrante
            if pkmnAux is not None:
                side.reserved[found_in_reserved_idx] = deepcopy(pkmnAux)
            _resolve_duplicate_illusion_reserves(side)
            print(f"El oponente ha cambiado de Pokémon a {new_active.name} ({new_active.HPA_battle_actual}/{new_active.HPB_battle_base}) : {slot_str}")

        # 2. PASO B: Si NO estaba en la reserva, lo buscamos en 'opponents' (Team Preview)
        else:
            for pkmn in side.opponents:
                if pkmn is not None and _same_party_species(namePkmn, getattr(pkmn, 'name', '')):
                    pkmnAux = deepcopy(current_active)
                    pkmnAux = _revert_special_pokemon(pkmnAux, side)
                    
                    new_active = deepcopy(pkmn)
                    _apply_switch_identity(new_active, namePkmn)

                    # AUTORIDAD DEL LOG: HP / faint / status del mensaje de switch
                    _apply_switch_condition(new_active, hpa_val, hpb_val, new_non_volatile)

                    new_active.volatile_statuses = defaultdict(lambda: 0)
                    new_active.boosts = defaultdict(lambda: 0)
                    new_active.is_drowsy = False
                    _clear_all_field_temp_flags(new_active, entering=True)
                    new_active.perish_count = 0

                    # Confirmación del Pokémon
                    if not getattr(new_active, 'is_opponent_confirmed', False):
                        side.opponentPkmnConfirmed += 1
                        new_active.is_opponent_confirmed = True

                    # ENRIQUECIMIENTO ZOROARK / OTS: Hereda objeto, habilidad y naturaleza
                    new_active.objectPkmn = getattr(pkmn, 'objectPkmn', '')
                    new_active.abilityPkmn = getattr(pkmn, 'abilityPkmn', '')
                    new_active.nature = getattr(pkmn, 'nature', None)

                    if is_active1:
                        side.active1 = new_active
                    else:
                        side.active2 = new_active

                    print(f"El oponente ha revelado a {new_active.name} ({new_active.HPA_battle_actual}/{new_active.HPB_battle_base}) : {slot_str}")

                    # El Pokémon saliente recién pasa a 'reserved' al dejar el campo
                    if pkmnAux is not None:
                        reserved_copy = deepcopy(pkmnAux)
                        # El entrante no estaba en reserved, por lo que el slot
                        # que abandona el campo siempre debe ocupar una reserva.
                        # No deduplicamos por especie: dos instancias distintas
                        # pueden compartir especie y la lista de preview no es
                        # la reserva real.
                        side.reserved.append(reserved_copy)
                    _resolve_duplicate_illusion_reserves(side)
                    break

        slot_now = side.active1 if is_active1 else side.active2
        if not _same_party_species(namePkmn, getattr(slot_now, 'name', '') or ''):
            other = side.active2 if is_active1 else side.active1
            if other is not None and _same_party_species(namePkmn, getattr(other, 'name', '')):
                if is_active1:
                    side.active1, side.active2 = other, slot_now
                else:
                    side.active2, side.active1 = other, slot_now
                incoming = side.active1 if is_active1 else side.active2
                _apply_switch_identity(incoming, namePkmn)
                _apply_switch_condition(incoming, hpa_val, hpb_val, new_non_volatile)

        # Autoridad del |switch|: si el slot sigue KO pero el log trae HP>0, revivir
        # in-place (evita foe2.fainted fantasma tras fallos de matching).
        slot_now = side.active1 if is_active1 else side.active2
        if (
            slot_now is not None
            and hpa_val > 0
            and (
                getattr(slot_now, 'is_fainted', False)
                or int(getattr(slot_now, 'HPA_battle_actual', 0) or 0) <= 0
                or not _same_party_species(namePkmn, getattr(slot_now, 'name', '') or '')
            )
        ):
            if _same_party_species(namePkmn, getattr(slot_now, 'name', '') or ''):
                _apply_switch_identity(slot_now, namePkmn)
                _apply_switch_condition(slot_now, hpa_val, hpb_val, new_non_volatile)
                slot_now.volatile_statuses = defaultdict(lambda: 0)
                slot_now.boosts = defaultdict(lambda: 0)
                slot_now.is_drowsy = False
                _clear_all_field_temp_flags(slot_now, entering=True)
            else:
                # Último recurso: clonar desde preview OTS o fabricar shell.
                donor = None
                for pkmn in list(getattr(side, 'opponents', []) or []) + list(
                    getattr(side, 'reserved', []) or []
                ):
                    if pkmn is not None and _same_party_species(
                        namePkmn, getattr(pkmn, 'name', '')
                    ):
                        donor = pkmn
                        break
                if donor is not None:
                    outgoing = deepcopy(slot_now)
                    outgoing = _revert_special_pokemon(outgoing, side)
                    new_active = deepcopy(donor)
                    _apply_switch_identity(new_active, namePkmn)
                    _apply_switch_condition(new_active, hpa_val, hpb_val, new_non_volatile)
                    new_active.volatile_statuses = defaultdict(lambda: 0)
                    new_active.boosts = defaultdict(lambda: 0)
                    new_active.is_drowsy = False
                    _clear_all_field_temp_flags(new_active, entering=True)
                    new_active.perish_count = 0
                    if is_active1:
                        side.active1 = new_active
                    else:
                        side.active2 = new_active
                    if outgoing is not None:
                        side.reserved.append(deepcopy(outgoing))

        _resolve_duplicate_illusion_reserves(side)

    # =========================================================================
    # CASO 2: CAMBIO REALIZADO POR EL USUARIO
    # =========================================================================
    else:
        hp_status_raw_u = split_msg[4].strip() if len(split_msg) > 4 else ""
        hpa_u, hpb_u, nvs_u = (
            _parse_hp_and_status_from_switch(hp_status_raw_u)
            if hp_status_raw_u
            else (-1, -1, None)
        )

        found_user_idx = -1
        fainted_user_idx = -1
        for contador, pkmn in enumerate(side.reserved):
            if pkmn is None or not _same_party_species(namePkmn, getattr(pkmn, 'name', '')):
                continue
            if getattr(pkmn, 'is_fainted', False) or int(getattr(pkmn, 'HPA_battle_actual', 1) or 0) <= 0:
                if fainted_user_idx < 0:
                    fainted_user_idx = contador
                continue
            found_user_idx = contador
            break
        if found_user_idx < 0:
            found_user_idx = fainted_user_idx

        slot_target = side.active1 if is_active1 else side.active2
        if (
            found_user_idx < 0
            and slot_target is not None
            and _same_party_species(namePkmn, getattr(slot_target, 'name', ''))
        ):
            _apply_switch_identity(slot_target, namePkmn)
            if hpa_u >= 0:
                _apply_switch_condition(slot_target, hpa_u, hpb_u, nvs_u)
            else:
                slot_target.is_fainted = False
            slot_target.volatile_statuses = defaultdict(lambda: 0)
            slot_target.boosts = defaultdict(lambda: 0)
            slot_target.is_drowsy = False
            _clear_all_field_temp_flags(slot_target, entering=True)
            slot_target.perish_count = 0
            return battle

        if found_user_idx >= 0:
            aux = deepcopy(side.active1 if is_active1 else side.active2)
            aux = _revert_special_pokemon(aux, side)

            new_active = deepcopy(side.reserved[found_user_idx])
            _apply_switch_identity(new_active, namePkmn)
            if hpa_u >= 0:
                _apply_switch_condition(new_active, hpa_u, hpb_u, nvs_u)
            else:
                new_active.is_fainted = False
            new_active.volatile_statuses = defaultdict(lambda: 0)
            new_active.boosts = defaultdict(lambda: 0)
            new_active.is_drowsy = False
            _clear_all_field_temp_flags(new_active, entering=True)
            new_active.perish_count = 0

            if is_active1:
                side.active1 = new_active
            else:
                side.active2 = new_active

            side.reserved[found_user_idx] = deepcopy(aux)
        else:
            # El entrante ya está en el otro slot (a/b cruzados).
            slot_now = side.active1 if is_active1 else side.active2
            other = side.active2 if is_active1 else side.active1
            if other is not None and _same_party_species(namePkmn, getattr(other, 'name', '')):
                if is_active1:
                    side.active1, side.active2 = other, slot_now
                else:
                    side.active2, side.active1 = other, slot_now
                incoming = side.active1 if is_active1 else side.active2
                _apply_switch_identity(incoming, namePkmn)
                if hpa_u >= 0:
                    _apply_switch_condition(incoming, hpa_u, hpb_u, nvs_u)
                else:
                    incoming.is_fainted = False

        # Autoridad del |switch| propio: especie/HP del log mandan sobre el slot.
        slot_now = side.active1 if is_active1 else side.active2
        if (
            slot_now is None
            or not _same_party_species(namePkmn, getattr(slot_now, 'name', '') or '')
            or (
                hpa_u > 0
                and (
                    getattr(slot_now, 'is_fainted', False)
                    or int(getattr(slot_now, 'HPA_battle_actual', 0) or 0) <= 0
                )
            )
        ):
            donor = None
            donor_idx = -1
            for idx, pkmn in enumerate(list(getattr(side, 'reserved', []) or [])):
                if pkmn is not None and _same_party_species(
                    namePkmn, getattr(pkmn, 'name', '')
                ):
                    donor = pkmn
                    donor_idx = idx
                    break
            if donor is not None:
                outgoing = deepcopy(slot_now) if slot_now is not None else None
                if outgoing is not None:
                    outgoing = _revert_special_pokemon(outgoing, side)
                new_active = deepcopy(donor)
                _apply_switch_identity(new_active, namePkmn)
                if hpa_u >= 0:
                    _apply_switch_condition(new_active, hpa_u, hpb_u, nvs_u)
                else:
                    new_active.is_fainted = False
                new_active.volatile_statuses = defaultdict(lambda: 0)
                new_active.boosts = defaultdict(lambda: 0)
                new_active.is_drowsy = False
                _clear_all_field_temp_flags(new_active, entering=True)
                new_active.perish_count = 0
                if is_active1:
                    side.active1 = new_active
                else:
                    side.active2 = new_active
                if donor_idx >= 0:
                    side.reserved[donor_idx] = (
                        deepcopy(outgoing) if outgoing is not None else None
                    )
            elif slot_now is not None and _same_party_species(
                namePkmn, getattr(slot_now, 'name', '') or ''
            ):
                _apply_switch_identity(slot_now, namePkmn)
                if hpa_u >= 0:
                    _apply_switch_condition(slot_now, hpa_u, hpb_u, nvs_u)

        # Limpieza de nulos en reservados del usuario
        side.reserved = [p for p in side.reserved if p is not None]

    return battle


############################################################################################################################################################
### OK - Marcar Pokémon debilitado                                                                                                                       ###
############################################################################################################################################################
def faint(battle: Battle, split_msg: list) -> Battle:
    # |faint|p1a: Tornadus
    # |faint|p2b: Dragonite
    opponent_turn = is_opponent(battle, split_msg)
    side = battle.opponent if opponent_turn else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    
    # Identificamos si es la posición A o B
    is_active1 = slot_str.endswith('a')
    pkmn = side.active1 if is_active1 else side.active2

    if pkmn:
        # Ditto transformado: revertir a Ditto base antes de marcar faint
        if getattr(pkmn, 'is_ditto', False) and _canonical_species_name(getattr(pkmn, 'name', '')) != 'ditto':
            pkmn = _revert_special_pokemon(pkmn, side)
            if is_active1:
                side.active1 = pkmn
            else:
                side.active2 = pkmn

        pkmn.HPA_battle_actual = 0
        pkmn.is_fainted = True
        pkmn.boosts = defaultdict(lambda: 0)
        pkmn.volatile_statuses = defaultdict(lambda: 0)
        pkmn.non_volatile_statuses = defaultdict(lambda: 0)
        pkmn.is_drowsy = False
        pkmn.perish_count = 0
        _clear_all_field_temp_flags(pkmn)

        if _is_zoroark(pkmn):
            pkmn.is_zoroark = True
            pkmn.is_opponent_confirmed = True
            _keep_highest_hp_reserve_for_species(side, pkmn.name)

    return battle


############################################################################################################################################################
### OK - Actualización de HP por daño o curación                                                                                                         ###
############################################################################################################################################################
def heal_or_damage(battle: Battle, split_msg: list) -> Battle:
    # |-damage|p2a: Zoroark|40/100 brn|[from] brn
    # |-damage|p2a: Tornadus|91/100|[from] item: Life Orb
    # |-damage|p1a: Raichu|135/167 tox|[from] psn
    # |-damage|p2a: Blastoise|50/100y
    # |-heal|p2: Dragonite|50/100y|[from] move: Revival Blessing
    # |-heal|p1: Farigiraf|111/222|[from] move: Revival Blessing
    if len(split_msg) < 4:
        return battle

    opponent_turn = is_opponent(battle, split_msg)
    side = battle.opponent if opponent_turn else battle.user
    slot_full = split_msg[2].strip()
    slot_str = slot_full.split(':')[0].strip()
    from_parts = [part.lower() for part in split_msg[4:]]
    from_revival_blessing = any(
        'revival blessing' in part or 'revivalblessing' in part.replace(' ', '')
        for part in from_parts
    )

    # Revival Blessing (y curas de banquillo) llegan como "p2: Species" sin slot a/b.
    is_party_target = (
        from_revival_blessing
        or (
            ':' in slot_full
            and not slot_str.endswith('a')
            and not slot_str.endswith('b')
        )
    )

    if is_party_target:
        raw_name = slot_full.split(':', 1)[1].strip() if ':' in slot_full else ''
        species_name = constants.standardName.get(raw_name.lower(), raw_name.lower())
        pkmn = _find_party_pokemon_for_heal(
            side,
            species_name,
            prefer_fainted=from_revival_blessing,
        )
    else:
        pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if not pkmn:
        return battle

    # HP antes del evento (para Δ daño / inferencia de SP)
    try:
        old_hp = int(getattr(pkmn, 'HPA_battle_actual', 0) or 0)
        old_max = int(getattr(pkmn, 'HPB_battle_base', 0) or 0)
    except (TypeError, ValueError):
        old_hp, old_max = 0, 0

    is_damage_event = len(split_msg) > 1 and split_msg[1].strip() == '-damage'

    hp_str = split_msg[3].strip()

    # Comprobamos si el Pokémon cayó debilitado
    if constants.FNT in hp_str or hp_str.startswith("0"):
        pkmn.HPA_battle_actual = 0
        pkmn.is_fainted = True
    else:
        parts = hp_str.split('/')
        if len(parts) >= 2:
            # Extraemos únicamente los dígitos del HP actual
            current_digits = re.findall(r'\d+', parts[0])
            if current_digits:
                current_hp_val = float(current_digits[0])
                pkmn.HPA_battle_actual = int(math.ceil(current_hp_val))

            # Extraemos únicamente los dígitos del HP máximo (incluso si viene pegado como "100y")
            max_digits = re.findall(r'\d+', parts[1])
            if max_digits:
                max_hp_val = float(max_digits[0])
                pkmn.HPB_battle_base = int(math.ceil(max_hp_val))

        if from_revival_blessing:
            pkmn.is_fainted = False
            pkmn.volatile_statuses = defaultdict(lambda: 0)
            pkmn.boosts = defaultdict(lambda: 0)
            print(
                f"[REVIVAL BLESSING] {pkmn.name} revive con "
                f"{pkmn.HPA_battle_actual}/{pkmn.HPB_battle_base}"
            )

    if is_damage_event and not is_party_target:
        _maybe_infer_from_damage(battle, split_msg, pkmn, old_hp, old_max)

    return battle


############################################################################################################################################################
### OK - Intercambio de posiciones activas en el campo (ej: Ally Switch)                                                                                 ###
############################################################################################################################################################
def swap(battle: Battle, split_msg: list) -> Battle:
    # |swap|p1a: Gothitelle|1|[from] move: Ally Switch
    # |swap|p2b: Indeedee|0
    side = battle.opponent if is_opponent(battle, split_msg) else battle.user

    # Intercambio de referencias instantáneo en O(1) sin deepcopy
    side.active1, side.active2 = side.active2, side.active1

    return battle


############################################################################################################################################################
# OK - Aumento de estadísticas                                                                                                                           ###
############################################################################################################################################################
def boost(battle: Battle, split_msg: list) -> Battle:
    # |-boost|p2b: Celesteela|spe|1
    raw_stat = split_msg[3].strip()
    stat = constants.STAT_ABBREVIATION_LOOKUPS.get(raw_stat, raw_stat)
    amount = int(split_msg[4].strip())

    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn:
        pkmn.boosts[stat] = min(pkmn.boosts[stat] + amount, constants.MAX_BOOSTS)

    return battle


############################################################################################################################################################
# OK - Disminución de estadísticas                                                                                                                       ###
############################################################################################################################################################
def unboost(battle: Battle, split_msg: list) -> Battle:
    # |-unboost|p2a: Whimsicott|spe|1
    raw_stat = split_msg[3].strip()
    stat = constants.STAT_ABBREVIATION_LOOKUPS.get(raw_stat, raw_stat)
    amount = int(split_msg[4].strip())

    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn:
        pkmn.boosts[stat] = max(pkmn.boosts[stat] - amount, -1 * constants.MAX_BOOSTS)

    return battle


############################################################################################################################################################
# OK - Cambio de estado alterado persistente (ej: 'brn', 'slp', 'par')                                                                                   ###
############################################################################################################################################################
def status(battle: Battle, split_msg: list) -> Battle:
    # |-status|p2a: Slowbro|brn
    # |-status|p2b: Maushold|slp|[from] move: Rest
    # |-status|p1a: Torkoal|par|[from] move: Body Slam
    # |-status|p2a: Incineroar|slp|[from] move: Yawn
    if len(split_msg) < 4:
        return battle

    status_name = split_msg[3].strip().lower()

    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn and hasattr(pkmn, 'non_volatile_statuses'):
        # Un Pokémon solo puede tener un estado mayor a la vez (brn/par/slp/frz/psn/tox).
        if not isinstance(pkmn.non_volatile_statuses, (dict, defaultdict)):
            pkmn.non_volatile_statuses = defaultdict(lambda: 0)
        if status_name in _MAJOR_NON_VOLATILE:
            for key in list(pkmn.non_volatile_statuses.keys()):
                if key in _MAJOR_NON_VOLATILE:
                    del pkmn.non_volatile_statuses[key]
        pkmn.non_volatile_statuses[status_name] = True
        # Al caer dormido por Yawn (u otra fuente), el aviso drowsy ya no aplica.
        if status_name == 'slp':
            _set_drowsy(pkmn, False)

    return battle


############################################################################################################################################################
# OK - Inicia alterado volatil (ej: 'moveyawn', 'confusion', 'substitute')                                                                               ###
############################################################################################################################################################
def start_volatile_status(battle: Battle, split_msg: list) -> Battle:
    # |-start|p2a: Incineroar|move: Yawn|[of] p1a: Gastrodon
    # |-start|p1b: Whimsicott|confusion
    # |-start|p1a: Aegislash|Substitute
    # |-start|p2a: Gengar|perish3|[silent]
    # |-start|p2a: Gengar|perish2
    # |-start|p1a: Incineroar|perish1
    # |-start|p1a: Incineroar|perish0
    
    # Validamos que el mensaje tenga la estructura esperada
    if len(split_msg) < 4:
        return battle

    # Extraemos y normalizamos el estado volátil (quitando prefijos como 'move:')
    raw_status = split_msg[3].split(":")[-1].strip()
    volatile_status = normalize_name(raw_status)

    # Determinamos a qué bando y Pokémon activo pertenece el estado
    opponent_turn = is_opponent(battle, split_msg)
    side = battle.opponent if opponent_turn else battle.user

    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn and hasattr(pkmn, 'volatile_statuses'):
        # BLINDAJE: Si volatile_statuses no es un dict/defaultdict (ej: si era una list []),
        # lo re-inicializamos como diccionario seguro
        if not isinstance(pkmn.volatile_statuses, (dict, defaultdict)):
            pkmn.volatile_statuses = defaultdict(lambda: 0)

        perish_n = _parse_perish_count(volatile_status)
        if perish_n is not None:
            # Autoridad del log: perish3 → perish2 → perish1 → perish0 (luego faint).
            _set_perish_count(pkmn, perish_n)
            return battle

        # Registramos el estado volátil en el diccionario
        pkmn.volatile_statuses[volatile_status] = True

        # Yawn: el mon sigue despierto y puede actuar/cambiar hasta el final del
        # siguiente turno, cuando llegue |-status|...|slp|[from] move: Yawn.
        if _is_yawn_volatile_name(volatile_status):
            pkmn.is_drowsy = True
            # Clave canónica para el volátil (Showdown normaliza a 'yawn').
            pkmn.volatile_statuses['yawn'] = True
        elif volatile_status in _CHARGE_IDS:
            _set_charged(pkmn, True)
        elif volatile_status in _DESTINY_BOND_IDS:
            _set_destiny_bonded(pkmn, True)
        elif volatile_status in _HELPING_HAND_IDS:
            _set_helping_hand(pkmn, True)
        elif volatile_status in _PROTECT_SINGLETURN:
            _set_protected(pkmn, True, volatile_status)
        elif volatile_status in ('flashfire', 'slowstart', 'electromorphosis'):
            pkmn.ability_on = True
        elif volatile_status == 'typechange' and len(split_msg) > 4:
            _apply_typechange(pkmn, split_msg[4].strip())

    return battle


############################################################################################################################################################
# OK - Fin alterado volatil (ej: 'moveyawn')                                                                                                             ###
############################################################################################################################################################
def end_volatile_status(battle: Battle, split_msg: list) -> Battle:
    # |-end|p1a: Aerodactyl|Encore
    # |-end|p2a: Zoroark|Illusion
    # |-end|p2a: Incineroar|move: Yawn
    # |-end|p2a: Gengar|perishsong   (poco frecuente; el clear habitual es al switch)
    
    # Validamos que el mensaje tenga la longitud necesaria para evitar errores de índice
    if len(split_msg) < 4:
        return battle

    # Extraemos y normalizamos el nombre del estado volátil que ha terminado
    raw_status = split_msg[3].split(":")[-1]
    volatile_status = normalize_name(raw_status)

    # Identificamos al bando y al Pokémon activo afectado
    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn and hasattr(pkmn, 'volatile_statuses'):
        # Si el estado volátil está registrado, lo eliminamos
        if volatile_status in pkmn.volatile_statuses:
            del pkmn.volatile_statuses[volatile_status]
        if _is_yawn_volatile_name(volatile_status):
            _set_drowsy(pkmn, False)
        if volatile_status.startswith('perish') or volatile_status == 'perishsong':
            _clear_perish(pkmn)
        if volatile_status in _PROTECT_SINGLETURN:
            _set_protected(pkmn, False)
        if volatile_status in _HELPING_HAND_IDS:
            _set_helping_hand(pkmn, False)
        if volatile_status in _DESTINY_BOND_IDS:
            _set_destiny_bonded(pkmn, False)
        if volatile_status in _CHARGE_IDS:
            _set_charged(pkmn, False)
        if volatile_status in ('flashfire', 'slowstart', 'electromorphosis'):
            pkmn.ability_on = False
        if volatile_status in _REDIRECT_SINGLETURN:
            _set_redirecting(pkmn, False)

    return battle


############################################################################################################################################################
# OK - Revelación de Illusion (Zoroark sustituye al disfraz en el campo)                                                                                 ###
############################################################################################################################################################
def replace_illusion(battle: Battle, split_msg: list, *, protocol_reveal: bool = True) -> Battle:
    # |replace|p2a: Zoroark|Zoroark, L50, M
    # |replace|p2b: Zoroark|Zoroark-Hisui, L50, F
    # Suele ir precedido de |-end|p2a: <Disfraz>|Illusion
    # protocol_reveal=False: deducción por move ilegal (PS aún muestra el disfraz)

    if len(split_msg) < 4:
        return battle

    raw_name = split_msg[3].partition(",")[0].strip()
    real_name = constants.standardName.get(raw_name.lower(), raw_name.lower())

    opponent_turn = is_opponent(battle, split_msg)
    side = battle.opponent if opponent_turn else battle.user

    slot_str = split_msg[2].split(':')[0].strip()
    is_active1 = slot_str.endswith('a')
    disguised_active = side.active1 if is_active1 else side.active2

    # Preservamos los estados acumulados durante el combate mientras estuvo disfrazado
    accumulated_volatile = deepcopy(disguised_active.volatile_statuses) if disguised_active and hasattr(disguised_active, 'volatile_statuses') else defaultdict(lambda: 0)
    accumulated_non_volatile = deepcopy(disguised_active.non_volatile_statuses) if disguised_active and hasattr(disguised_active, 'non_volatile_statuses') else defaultdict(lambda: 0)
    accumulated_boosts = deepcopy(disguised_active.boosts) if disguised_active and hasattr(disguised_active, 'boosts') else defaultdict(lambda: 0)
    current_hpa = disguised_active.HPA_battle_actual if disguised_active else 0
    current_hpb = disguised_active.HPB_battle_base if disguised_active else 0

    # 1. El Pokémon disfraz (ej. Dragonite) NUNCA estuvo en el campo: lo devolvemos intacto a la reserva
    if disguised_active and _canonical_species_name(disguised_active.name) != _canonical_species_name(real_name):
        disguised_active.is_fainted = False
        disguised_active.HPA_battle_actual = disguised_active.HPB_battle_base
        disguised_active.volatile_statuses = defaultdict(lambda: 0)
        disguised_active.non_volatile_statuses = defaultdict(lambda: 0)
        disguised_active.boosts = defaultdict(lambda: 0)

        # Si la especie disfrazada ya está conocida en el campo o reserva, no la volvemos a guardar.
        # Importante: la lista de preview (opponents) NO debe invalidar la reserva real que acaba de salir
        # y, sobre todo, NO se debe borrar toda la reserva de esa especie; solo debe quitarse la copia exacta
        # que era el disfraz del Zoroark, no la entrada ya confirmada del mismo Pokémon real.
        _remove_exact_reserve_entry(side, disguised_active)
        if not _species_exists_on_side(side, disguised_active.name, exclude=disguised_active):
            side.reserved.append(disguised_active)

    # 2. Buscamos la ficha completa de Zoroark en 'opponents' (OTS/Preview) para recuperar Objeto, Habilidad y Naturaleza
    ots_zoroark = None
    for p in side.opponents:
        if p and _canonical_species_name(p.name) == _canonical_species_name(real_name):
            ots_zoroark = p
            break

    # 3. Buscamos a Zoroark en 'reserved' si ya existía allí
    zoroark_instance = None
    zoroark_idx = -1
    for i, p in enumerate(side.reserved):
        if p and _canonical_species_name(p.name) == _canonical_species_name(real_name):
            zoroark_instance = p
            zoroark_idx = i
            break

    if zoroark_instance:
        # Si estaba en la reserva, lo removemos de allí porque pasa al campo activo.
        # Importante: no se limpia toda la reserva por especie; solo se elimina la referencia exacta
        # del Zoroark real que estaba ahí, no cualquier otros Pokémon ya confirmados del mismo nombre.
        side.reserved.pop(zoroark_idx)
        _remove_exact_reserve_entry(side, zoroark_instance)
    else:
        # Si es del oponente o no estaba instanciado, creamos la nueva instancia con los datos de ots_zoroark
        obj_item = normalize_name(getattr(ots_zoroark, 'objectPkmn', '') or '') if ots_zoroark else ''
        raw_ability = getattr(ots_zoroark, 'abilityPkmn', '') if ots_zoroark else ''
        obj_ability = normalize_name(raw_ability) if raw_ability else ''
        raw_nature = getattr(ots_zoroark, 'nature', None) if ots_zoroark else None
        obj_nature = normalize_name(raw_nature) if raw_nature else None
        
        # Asignamos los 4 movimientos si existen en ots_zoroark
        m1 = ots_zoroark.moves[0] if ots_zoroark and len(ots_zoroark.moves) > 0 else None
        m2 = ots_zoroark.moves[1] if ots_zoroark and len(ots_zoroark.moves) > 1 else None
        m3 = ots_zoroark.moves[2] if ots_zoroark and len(ots_zoroark.moves) > 2 else None
        m4 = ots_zoroark.moves[3] if ots_zoroark and len(ots_zoroark.moves) > 3 else None

        zoroark_instance = Pokemon(real_name, 0, opponent_turn, obj_item, obj_ability, m1, m2, m3, m4, obj_nature)

    # Confirmación del oponente: incrementar solo la primera vez que se revela.
    if opponent_turn and not getattr(zoroark_instance, 'is_opponent_confirmed', False):
        side.opponentPkmnConfirmed += 1
        zoroark_instance.is_opponent_confirmed = True
    elif opponent_turn:
        zoroark_instance.is_opponent_confirmed = True

    # 4. Asignamos Datos Reales de Zoroark (Objeto, Habilidad, Naturaleza y Stats Base)
    if ots_zoroark:
        # No reintroducir un item ya consumido/robado: OTS se sincroniza en -enditem.
        ots_item = getattr(ots_zoroark, 'objectPkmn', None)
        if ots_item is not None and str(ots_item).strip() != '':
            zoroark_instance.objectPkmn = ots_item
        else:
            zoroark_instance.objectPkmn = None
        zoroark_instance.abilityPkmn = getattr(ots_zoroark, 'abilityPkmn', zoroark_instance.abilityPkmn)
        zoroark_instance.nature = getattr(ots_zoroark, 'nature', zoroark_instance.nature)
        
        # RESTAURACIÓN DE STATS BASE REALES DE ZOROARK (Descartando las del disfraz)
        zoroark_instance.HPB_battle_base = 100 if opponent_turn else getattr(ots_zoroark, 'HPB_battle_base', current_hpb)
        zoroark_instance.ATK_battle_base = getattr(ots_zoroark, 'ATK_battle_base', 0)
        zoroark_instance.DEF_battle_base = getattr(ots_zoroark, 'DEF_battle_base', 0)
        zoroark_instance.SPA_battle_base = getattr(ots_zoroark, 'SPA_battle_base', 0)
        zoroark_instance.SPD_battle_base = getattr(ots_zoroark, 'SPD_battle_base', 0)
        zoroark_instance.VEL_battle_base = getattr(ots_zoroark, 'VEL_battle_base', 0)

    # 5. Mantenemos el HP actual y HEREDAMOS los estados acumulados y boosts del combate
    zoroark_instance.HPA_battle_actual = current_hpa
    zoroark_instance.volatile_statuses = accumulated_volatile
    zoroark_instance.non_volatile_statuses = accumulated_non_volatile
    zoroark_instance.boosts = accumulated_boosts
    zoroark_instance.is_zoroark = True
    # Conservar disfraz para reporte dual
    if disguised_active and _canonical_species_name(getattr(disguised_active, 'name', '')) != _canonical_species_name(real_name):
        zoroark_instance.illusion_disguise = getattr(disguised_active, 'name', None)
    elif getattr(disguised_active, 'illusion_disguise', None):
        zoroark_instance.illusion_disguise = disguised_active.illusion_disguise
    else:
        zoroark_instance.illusion_disguise = getattr(zoroark_instance, 'illusion_disguise', None)

    if protocol_reveal:
        zoroark_instance.illusion_protocol_revealed = True
    else:
        # Deducción temprana: no pisar si PS ya reveló en un |replace| previo
        if not getattr(zoroark_instance, 'illusion_protocol_revealed', False):
            zoroark_instance.illusion_protocol_revealed = False
        # Heredar flag del activo disfrazado si ya estaba revelado
        if disguised_active and getattr(disguised_active, 'illusion_protocol_revealed', False):
            zoroark_instance.illusion_protocol_revealed = True

    # Asignamos a Zoroark como el activo real
    if is_active1:
        side.active1 = zoroark_instance
    else:
        side.active2 = zoroark_instance

    # Limpieza final: solo eliminamos entradas exactas de la misma referencia que ya no pertenecen a la reserva.
    _remove_exact_reserve_entry(side, zoroark_instance)

    # Lo observado mientras iba disfrazado se atribuyó a la especie equivocada:
    # esas conclusiones eran de Zoroark, así que se descartan.
    disguise = getattr(zoroark_instance, 'illusion_disguise', None)
    if opponent_turn and disguise:
        battle.stat_beliefs.forget(disguise)

    print(f"[ILLUSION REVEALED] {real_name} se ha revelado en {slot_str} con HP: {current_hpa}")

    return battle


############################################################################################################################################################
# OK - Curación de estado alterado                                                                                                                       ###
############################################################################################################################################################
def curestatus(battle: Battle, split_msg: list) -> Battle:
    # |-curestatus|p2a: Kyogre|slp|[msg]
    # |-curestatus|p1b: Farigiraf|par|[msg]
    # |-curestatus|p2a: Blastoise|brn|[msg]
    if len(split_msg) < 3:
        return battle

    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2
    cured = split_msg[3].strip().lower() if len(split_msg) > 3 else ''

    if pkmn and hasattr(pkmn, 'non_volatile_statuses'):
        # El modelo de estado vive en non_volatile_statuses (no en un atributo .status).
        if not isinstance(pkmn.non_volatile_statuses, (dict, defaultdict)):
            pkmn.non_volatile_statuses = defaultdict(lambda: 0)
        if cured and cured in pkmn.non_volatile_statuses:
            del pkmn.non_volatile_statuses[cured]
        else:
            pkmn.non_volatile_statuses = defaultdict(lambda: 0)

    return battle


############################################################################################################################################################
# OK - Establecimiento / eliminación de clima                                                                                                            ###
############################################################################################################################################################

# Rocas que extienden el clima de 5 → 8 turnos (el Pokémon fuente debe llevarlas).
_WEATHER_EXTENDER_ITEMS = {
    'sunnyday': 'heatrock',
    'desolateland': 'heatrock',
    'raindance': 'damprock',
    'primordialsea': 'damprock',
    'sandstorm': 'smoothrock',
    'snowscape': 'icyrock',
    'hail': 'icyrock',
}

_SCREEN_SIDE_CONDITIONS = frozenset({'reflect', 'lightscreen', 'auroraveil'})


def _normalized_held_item(pkmn) -> str:
    raw = getattr(pkmn, 'objectPkmn', None) or ''
    return normalize_name(raw) if raw else ''


def _resolve_active_from_ident(battle: Battle, ident: str):
    """Resuelve 'p1a: Torkoal' / 'p2b' al Pokémon activo correspondiente."""
    if not ident:
        return None
    slot = ident.split(':')[0].strip()
    if not slot:
        return None
    side = _side_from_slot(battle, slot)
    return side.active1 if slot.endswith('a') else side.active2


def _extract_of_ident(split_msg: list) -> str:
    """Extrae el ident tras [of] si existe (ej: '[of] p1a: Torkoal' → 'p1a: Torkoal')."""
    for part in split_msg[3:]:
        if '[of]' in part.lower():
            # Soporta '[of] p1a: Torkoal' y variantes con espacios.
            after = part.split('[of]', 1)[-1].strip()
            if after.startswith(']'):
                after = after[1:].strip()
            return after
    return ''


def _side_active_holds_item(side, item_id: str) -> bool:
    item_id = normalize_name(item_id) if item_id else ''
    if not item_id:
        return False
    for pkmn in (getattr(side, 'active1', None), getattr(side, 'active2', None)):
        if pkmn and _normalized_held_item(pkmn) == item_id:
            return True
    return False


def _weather_duration_turns(battle: Battle, weather_name: str, split_msg: list) -> int:
    """
    Gen 6+ (VGC2022): clima por movimiento o habilidad dura 5 turnos, o 8 si el
    Pokémon fuente lleva la roca correspondiente (Heat/Damp/Smooth/Icy Rock).
    """
    extender = _WEATHER_EXTENDER_ITEMS.get(weather_name)
    if not extender:
        return 5

    of_ident = _extract_of_ident(split_msg)
    source = _resolve_active_from_ident(battle, of_ident) if of_ident else None
    if source and _normalized_held_item(source) == extender:
        return 8

    # Sin [of] fiable: si algún activo en el campo lleva la roca, asumimos extensión.
    for side in (battle.user, battle.opponent):
        if _side_active_holds_item(side, extender):
            return 8
    return 5


def weather(battle: Battle, split_msg: list) -> Battle:
    # |-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal
    # |-weather|RainDance|[from] move: Rain Dance|[of] p2b: Pelipper
    # |-weather|Sandstorm
    # |-weather|Snowscape|[upkeep]
    # |-weather|none

    raw_weather = split_msg[2].split(':')[-1].strip()
    weather_name = normalize_name(raw_weather)
    is_upkeep = any(part.strip().lower() == "[upkeep]" for part in split_msg[3:])

    if weather_name in ("none", "null"):
        battle.weather = 'none'
        battle.weatherTurn = 0
        return battle

    # Upkeep tardío tras |-weather|none no debe resucitar el clima (sticky rain/sun).
    if is_upkeep:
        if battle.weatherTurn <= 0 or (battle.weather or 'none') in ('none', 'null', ''):
            return battle
        battle.weather = weather_name
        return battle

    battle.weather = weather_name
    battle.weatherTurn = _weather_duration_turns(battle, weather_name, split_msg)
    return battle


# Solo uno de estos cuatro a la vez. Rooms (trickroom/wonderroom) sí pueden
# coexistir entre sí y con un terreno.
_EXCLUSIVE_TERRAINS = frozenset({
    'electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain',
})


############################################################################################################################################################
# OK - Inicio de Terreno / Room (Trick/Wonder). Terrenos exclusivos se reemplazan.                                                                       ###
############################################################################################################################################################
def fieldstart(battle: Battle, split_msg: list) -> Battle:
    # |-fieldstart|move: Wonder Room|[of] p1b: Starmie
    # |-fieldstart|move: Electric Terrain
    # |-fieldstart|move: Trick Room|[of] p2b: Farigiraf
    # PS a veces no manda -fieldend del terreno anterior antes del nuevo.

    raw_field = split_msg[2].split(':')[-1].strip()
    field_name = normalize_name(raw_field)
    if field_name in _EXCLUSIVE_TERRAINS:
        for other in _EXCLUSIVE_TERRAINS:
            if other != field_name:
                battle.terrain.pop(other, None)
    battle.terrain[field_name] = 5

    return battle


############################################################################################################################################################
# OK - Fin de Terreno, incluye Trick Room                                                                                                                ###
############################################################################################################################################################
def fieldend(battle: Battle, split_msg: list) -> Battle:
    # |-fieldend|move: Trick Room

    raw_field = split_msg[2].split(':')[-1].strip()
    field_name = normalize_name(raw_field)
    battle.terrain.pop(field_name, None)

    return battle


############################################################################################################################################################
# OK - Activación de efectos de bando (Tailwind, Reflect, Light Screen, etc.)                                                                            ###
############################################################################################################################################################
def sidestart(battle: Battle, split_msg: list) -> Battle:
    # |-sidestart|p1: PopplioPeru98|move: Tailwind
    # |-sidestart|p2: Opponent|move: Light Screen
    # |-sidestart|p1: Agent|move: Reflect
    # |-sidestart|p2: Opponent|move: Aurora Veil
    raw_condition = next(
        (
            part.split(':', 1)[1].strip()
            for part in split_msg[3:]
            if ':' in part and part.split(':', 1)[0].strip().lower() in ('move', 'ability')
        ),
        split_msg[-1].strip() if len(split_msg) > 3 else '',
    )
    condition = normalize_name(raw_condition)

    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    # Gen 8+: Tailwind = 4. Reflect / Light Screen / Aurora Veil = 5, o 8 con Light Clay
    # en un activo del bando (quien lanza el move está en campo).
    if condition == 'tailwind':
        duration = 4
    elif condition in _SCREEN_SIDE_CONDITIONS:
        duration = 8 if _side_active_holds_item(side, 'lightclay') else 5
    else:
        duration = 5
    side.side_conditions[condition] = duration
    return battle


############################################################################################################################################################
# OK - Finalización de efectos de bando (Tailwind, Reflect, Light Screen, etc.)                                                                          ###
############################################################################################################################################################
def sideend(battle: Battle, split_msg: list) -> Battle:
    # |-sideend|p2: PopplioPeru98|move: Tailwind
    # |-sideend|p2: popplioperu98|move: Light Screen
    raw_condition = split_msg[3].split(':')[-1].strip()
    condition = normalize_name(raw_condition)

    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    side.side_conditions.pop(condition, None)
    return battle


############################################################################################################################################################
# OK - Establece o cambia el objeto equipado de un Pokémon (ej: por movimientos como Trick o Switcheroo)                                               ###
############################################################################################################################################################
def set_item(battle: Battle, split_msg: list) -> Battle:
    # |-item|p2a: Torkoal|Choice Scarf|[from] move: Trick
    # |-item|p1b: Ditto|Sitrus Berry|[from] move: Trick
    # |-item|p2b: Farigiraf|Leftovers|[from] ability: Frisk
    
    # Validamos que el mensaje tenga la longitud mínima para extraer el slot y el objeto
    if len(split_msg) < 4:
        return battle

    # Extraemos el nombre del objeto (ej: "Choice Scarf") y lo normalizamos
    item_name = normalize_name(split_msg[3].strip())

    # Determinamos a qué bando y Pokémon activo pertenece el cambio de objeto
    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn and hasattr(pkmn, 'objectPkmn'):
        pkmn.objectPkmn = item_name

    return battle


def _clear_item_everywhere(side, item_name: str) -> None:
    """Item Clause: limpia el objeto en activos, reserva y plantilla OTS."""
    item = normalize_name(item_name or '')
    if not item:
        return
    for pkmn in _iter_side_pokemon(side, include_opponents=True):
        if normalize_name(getattr(pkmn, 'objectPkmn', '') or '') == item:
            pkmn.objectPkmn = None


############################################################################################################################################################
# OK - Remueve o consume el objeto equipado de un Pokémon (ej: Knock Off, Focus Sash gastado, Baya)                                                      ###
############################################################################################################################################################
def remove_item(battle: Battle, split_msg: list) -> Battle:
    # |-enditem|p1b: Kingambit|Focus Sash
    # |-enditem|p2a: Rillaboom|Grassy Seed|[from] item: Grassy Seed
    # |-enditem|p1b: Ditto|Sitrus Berry|[eat]
    # |-enditem|p2a: Dragonite|Expert Belt|[from] move: Corrosive Gas  (puede ser Illusion)
    
    # Validamos que el mensaje tenga la longitud mínima requerida
    if len(split_msg) < 3:
        return battle

    item_name = normalize_name(split_msg[3].strip()) if len(split_msg) > 3 else ''

    # Focus Sash/Band: el |-damage| siguiente NO es un roll completo (queda a 1 HP).
    if item_name in ('focussash', 'focusband'):
        battle.pending_damage_sash = True

    # Determinamos a qué bando y Pokémon activo pertenece la pérdida del objeto
    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn and hasattr(pkmn, 'objectPkmn'):
        # Limpiamos el objeto equipado del Pokémon activo
        pkmn.objectPkmn = None

    # Sincroniza plantillas OTS / reserva (crítico con Illusion: el log muestra el item real)
    if item_name:
        _clear_item_everywhere(side, item_name)

    return battle


############################################################################################################################################################
# OK - Establece la habilidad revelada o cambiada de un Pokémon (ej: al entrar al campo con Intimidate, Drizzle, Unnerve, o por Skill Swap)              ###
############################################################################################################################################################
def set_opponent_ability_from_ability_tag(battle: Battle, split_msg: list) -> Battle:
    # |-ability|p1a: Aerodactyl|Unnerve
    # |-ability|p2a: Incineroar|Intimidate|[from] ability: Intimidate|[of] p2a: Incineroar
    
    # Validamos la longitud mínima del mensaje para extraer el slot y la habilidad
    if len(split_msg) < 4:
        return battle

    # Extraemos el nombre de la habilidad (ej: "Unnerve", "Intimidate")
    ability_name = split_msg[3].strip()

    # Identificamos el bando (oponente o usuario) y el slot activo
    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn and hasattr(pkmn, 'abilityPkmn'):
        # Asignamos la habilidad revelada/actualizada al Pokémon
        pkmn.abilityPkmn = normalize_name(ability_name)

    return battle


############################################################################################################################################################
# OK - Cambio de forma permanente (MegaEvolución, Mimikyu Busted, etc.)                                                                                  ###
############################################################################################################################################################
def form_change(battle: Battle, split_msg: list) -> Battle:
    # |detailschange|p1a: Aerodactyl|Aerodactyl-Mega, L50, M
    # |detailschange|p1a: Mimikyu|Mimikyu-Busted, L50, F
    # |-formechange|p1b: Aegislash|Aegislash-Blade|
    
    # Validamos que el mensaje tenga la estructura esperada
    if len(split_msg) < 4:
        return battle

    raw_name = split_msg[3].partition(",")[0].strip()
    
    # Normalización del nombre del Pokémon según las constantes del proyecto
    if raw_name.lower() in ('mimikyu', 'mimikyu-disguised'):
        name_pkmn = 'mimikyu-busted'
    else:
        name_pkmn = constants.standardName.get(raw_name.lower(), raw_name.lower())

    opponent_turn = is_opponent(battle, split_msg)
    side = battle.opponent if opponent_turn else battle.user

    slot_str = split_msg[2].split(':')[0].strip()
    is_active1 = slot_str.endswith('a')
    current_active = side.active1 if is_active1 else side.active2

    if not current_active:
        return battle

    # Caso especial para Mimikyu: solo actualizamos su nombre a 'mimikyu-busted'
    if name_pkmn == 'mimikyu-busted':
        current_active.name = 'mimikyu-busted'
    else:
        # Para Mega-Evoluciones u otras formas permanentes:
        # Conservamos estado de combate (HP, status, boosts, confirmacion) e instanciamos la nueva forma.
        aux_volatile = deepcopy(getattr(current_active, 'volatile_statuses', defaultdict(lambda: 0)))
        aux_boosts = deepcopy(getattr(current_active, 'boosts', defaultdict(lambda: 0)))
        aux_non_volatile = deepcopy(getattr(current_active, 'non_volatile_statuses', defaultdict(lambda: 0)))
        aux_hpa = getattr(current_active, 'HPA_battle_actual', None)
        aux_hpb = getattr(current_active, 'HPB_battle_base', None)
        aux_confirmed = bool(getattr(current_active, 'is_opponent_confirmed', False))
        aux_drowsy = bool(getattr(current_active, 'is_drowsy', False))
        aux_perish = int(getattr(current_active, 'perish_count', 0) or 0)
        aux_moves = list(getattr(current_active, 'moves', [None, None, None, None]))
        aux_pp = list(getattr(current_active, 'movesPP', [-1, -1, -1, -1]))
        while len(aux_moves) < 4:
            aux_moves.append(None)
        while len(aux_pp) < 4:
            aux_pp.append(-1)

        mega_ability = _mega_forme_ability(name_pkmn)
        ability = mega_ability or normalize_name(getattr(current_active, 'abilityPkmn', '') or '')

        new_pkmn = Pokemon(
            name_pkmn,
            0,
            opponent_turn,
            current_active.objectPkmn,
            ability,
            aux_moves[0],
            aux_moves[1],
            aux_moves[2],
            aux_moves[3],
            getattr(current_active, 'nature', None),
        )

        # getStatsBattle() pone HPA=100 en oponentes: restauramos el estado real.
        if aux_hpb is not None:
            new_pkmn.HPB_battle_base = aux_hpb
        if aux_hpa is not None:
            new_pkmn.HPA_battle_actual = aux_hpa
        new_pkmn.volatile_statuses = aux_volatile
        new_pkmn.boosts = aux_boosts
        new_pkmn.non_volatile_statuses = aux_non_volatile
        new_pkmn.movesPP = aux_pp
        new_pkmn.is_opponent_confirmed = aux_confirmed
        new_pkmn.is_drowsy = aux_drowsy
        new_pkmn.perish_count = aux_perish
        if 'mega' in _canonical_species_name(name_pkmn):
            new_pkmn.is_mega_evo = True
            if mega_ability:
                new_pkmn.abilityPkmn = mega_ability

        if is_active1:
            side.active1 = new_pkmn
        else:
            side.active2 = new_pkmn

    return battle


############################################################################################################################################################
# OK -  Habilidad Imposter / Transformación (Ditto)                                                                                                      ###
############################################################################################################################################################
def transform(battle: Battle, split_msg: list) -> Battle:
    # Ejemplo de mensaje: |-transform|p1b: Ditto|p2a: Incineroar|[from] ability: Imposter
    
    # Validamos que el mensaje contenga tanto el origen (Ditto) como el objetivo a copiar
    if len(split_msg) < 4:
        return battle

    # 1. Identificamos al Pokémon que se transforma (source / Ditto)
    transform_opponent = is_opponent(battle, split_msg)
    source_side = battle.opponent if transform_opponent else battle.user
    source_slot_str = split_msg[2].split(':')[0].strip()
    source_is_active1 = source_slot_str.endswith('a')
    source_pkmn = source_side.active1 if source_is_active1 else source_side.active2

    # 2. Identificamos al Pokémon que está siendo copiado (target)
    target_slot_full = split_msg[3].split(':')[0].strip() # Ej: "p2a" o "p1b"
    target_is_opponent = _is_opponent_slot(battle, target_slot_full)
    target_side = battle.opponent if target_is_opponent else battle.user
    target_is_active1 = target_slot_full.endswith('a')
    target_pkmn = target_side.active1 if target_is_active1 else target_side.active2

    # 3. Si ambos existen, realizamos la transformación
    if source_pkmn and target_pkmn:
        # Guardamos la información única de Ditto que NUNCA cambia al transformarse
        original_item = source_pkmn.objectPkmn
        original_hpa = source_pkmn.HPA_battle_actual
        original_hpb = source_pkmn.HPB_battle_base
        original_non_volatile = deepcopy(source_pkmn.non_volatile_statuses)
        original_volatile = deepcopy(source_pkmn.volatile_statuses)
        original_drowsy = bool(getattr(source_pkmn, 'is_drowsy', False))
        original_perish = int(getattr(source_pkmn, 'perish_count', 0) or 0)

        # Creamos una copia exacta de los atributos del objetivo (tipos, movimientos, etc.)
        new_source = deepcopy(target_pkmn)

        # Reforzamos la identidad de Ditto y aquella parte que debe mantenerse inmutable
        new_source.name = target_pkmn.name
        new_source.objectPkmn = original_item
        # Habilidad efectiva = la del objetivo (como en el |request| de Showdown)
        target_ability = normalize_name(getattr(target_pkmn, 'abilityPkmn', '') or '')
        new_source.abilityPkmn = target_ability or 'imposter'
        new_source.HPA_battle_actual = original_hpa
        new_source.HPB_battle_base = original_hpb
        new_source.is_ditto = True
        new_source.is_zoroark = False  # Nunca heredar bandera Illusion del objetivo
        new_source.is_mega_evo = False
        new_source.can_mega_evo = False
        new_source.non_volatile_statuses = original_non_volatile
        new_source.volatile_statuses = original_volatile
        new_source.is_drowsy = original_drowsy
        new_source.perish_count = original_perish

        stat_names = ('ATK', 'DEF', 'SPA', 'SPD', 'VEL')

        # El Ditto del agente copia las bases visibles del rival, pero sus valores actuales
        # quedan desconocidos. El Ditto rival copia los valores actuales visibles del agente,
        # pero sus bases no son observables. En ambos casos sus HP siguen siendo los originales.
        new_source.boosts = deepcopy(target_pkmn.boosts)
        if not transform_opponent:
            for stat_name in stat_names:
                setattr(new_source, f'{stat_name}_battle_base', getattr(target_pkmn, f'{stat_name}_battle_base'))
                setattr(new_source, f'{stat_name}_battle_actual', -1)
        else:
            for stat_name in stat_names:
                setattr(new_source, f'{stat_name}_battle_base', -1)
                setattr(new_source, f'{stat_name}_battle_actual', getattr(target_pkmn, f'{stat_name}_battle_actual'))

        # Reemplazamos al Pokémon transformado en su slot activo correspondiente
        if source_is_active1:
            source_side.active1 = new_source
        else:
            source_side.active2 = new_source

    return battle


############################################################################################################################################################
# OK -  Mega-evolución de un Pokémon en el campo                                                                                                         ###
############################################################################################################################################################
def mega(battle: Battle, split_msg: list) -> Battle:
    # |-mega|p1a: Aerodactyl|Aerodactyl|Aerodactylite
    # |-mega|p2a: Blastoise|Blastoise|Blastoisinite
    # |-mega|p1b: Charizard|Charizard|Charizardite Y
    # La forma y stats definitivas suelen llegar justo después en |detailschange|.

    # Validamos que el mensaje tenga la estructura esperada
    if len(split_msg) < 3:
        return battle

    # Determinamos a qué bando y slot activo pertenece el Pokémon que megaevoluciona
    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn and hasattr(pkmn, 'is_mega_evo'):
        # Establecemos la bandera de Mega Evolución en True
        pkmn.is_mega_evo = True
        # Si el detalle de forma aún no llegó, prepara la habilidad mega por el nombre base.
        # detailschange suele sobrescribir con la forma -Mega y su habilidad real.
        forme_guess = _canonical_species_name(getattr(pkmn, 'name', ''))
        if not forme_guess.endswith('-mega') and not forme_guess.endswith('-mega-x') and not forme_guess.endswith('-mega-y'):
            for candidate in (f'{forme_guess}-mega', f'{forme_guess}-mega-x', f'{forme_guess}-mega-y'):
                mega_ability = _mega_forme_ability(candidate)
                if mega_ability:
                    pkmn.abilityPkmn = mega_ability
                    break
        else:
            mega_ability = _mega_forme_ability(forme_guess)
            if mega_ability:
                pkmn.abilityPkmn = mega_ability

    return battle


############################################################################################################################################################
# OK - Terastallize (Champions / gen9)                                                                                                                    ###
############################################################################################################################################################
def terastallize(battle: Battle, split_msg: list) -> Battle:
    # |-terastallize|p1a: Incineroar|Fire
    # |-terastallize|p2b: Rillaboom|Grass
    if len(split_msg) < 4:
        return battle
    pkmn = _resolve_active_from_ident(battle, split_msg[2].strip())
    if pkmn is None:
        return battle
    tera = split_msg[3].strip()
    if tera:
        pkmn.tera_type = tera
        pkmn.is_terastallized = True
        # Tipo efectivo para el calc mientras está terastallizado.
        pkmn.type1 = tera
        pkmn.type2 = None
    return battle


############################################################################################################################################################
# OK -  Restablece las estadísticas (boosts) de TODOS los Pokémon activos en el campo (ej: por efecto de Haze / Niebla)                                  ###
############################################################################################################################################################
def clearallboost(battle: Battle, _=None) -> Battle:
    # |-clearallboost
    # Agrupamos todos los Pokémon activos posibles en el campo de dobles
    active_pokemons = (
        battle.user.active1,
        battle.user.active2,
        battle.opponent.active1,
        battle.opponent.active2,
    )

    for pkmn in active_pokemons:
        if pkmn and hasattr(pkmn, 'boosts'):
            # Limpiamos o reseteamos a 0 todos los boosts acumulados
            # Como boosts es un defaultdict, vaciarlo o poner cada valor a 0 restablece las estadísticas
            pkmn.boosts.clear()

    return battle


############################################################################################################################################################
# OK -  Restablece todas las estadísticas (boosts) a 0 de un Pokémon específico (ej: por efecto de Clear Smog)                                           ###
############################################################################################################################################################
def clearboost(battle: Battle, split_msg: list) -> Battle:
    # Ejemplo de mensaje: |-clearboost|p2b: Farigiraf
    
    # Validamos que el mensaje contenga la información del slot del Pokémon
    if len(split_msg) < 3:
        return battle

    # Determinamos a qué bando y slot activo pertenece el Pokémon afectado
    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn and hasattr(pkmn, 'boosts'):
        # Limpiamos todos los boosts acumulados (vuelven a 0)
        pkmn.boosts.clear()

    return battle


############################################################################################################################################################
# OK - Solo limpia boosts negativos (White Herb → |-clearnegativeboost|)                                                                                 ###
############################################################################################################################################################
def clearnegativeboost(battle: Battle, split_msg: list) -> Battle:
    # |-clearnegativeboost|p1a: Salamence|[silent]
    # |-clearnegativeboost|p2b: Incineroar|[from] item: White Herb
    if len(split_msg) < 3:
        return battle

    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn and hasattr(pkmn, 'boosts'):
        for key, val in list(pkmn.boosts.items()):
            if int(val or 0) < 0:
                pkmn.boosts[key] = 0

    return battle


############################################################################################################################################################
# OK - Solo limpia boosts positivos (Mirror Herb / similares → |-clearpositiveboost|)                                                                    ###
############################################################################################################################################################
def clearpositiveboost(battle: Battle, split_msg: list) -> Battle:
    # |-clearpositiveboost|p1a: Pokemon|[from] item: Mirror Herb  (raro; paridad PS)
    if len(split_msg) < 3:
        return battle

    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn and hasattr(pkmn, 'boosts'):
        for key, val in list(pkmn.boosts.items()):
            if int(val or 0) > 0:
                pkmn.boosts[key] = 0

    return battle


############################################################################################################################################################
# OK - Copia las estadísticas (boosts) de un Pokémon a otro (ej: por efecto del movimiento Psych Up / Más Psique)                                        ###
############################################################################################################################################################
def copyboost(battle: Battle, split_msg: list) -> Battle:
    # Ejemplo de mensaje: |-copyboost|p2b: Farigiraf|p1b: Chimecho|[from] move: Psych Up
    
    # Validamos que el mensaje contenga tanto el receptor como la fuente de los boosts
    if len(split_msg) < 4:
        return battle

    # 1. Identificamos al Pokémon que RECIBE/COPIA los boosts (origen del mensaje: split_msg[2])
    receiver_opponent = is_opponent(battle, split_msg)
    receiver_side = battle.opponent if receiver_opponent else battle.user
    receiver_slot = split_msg[2].split(':')[0].strip()
    receiver_pkmn = receiver_side.active1 if receiver_slot.endswith('a') else receiver_side.active2

    # 2. Identificamos al Pokémon del cual SE COPIAN los boosts (fuente: split_msg[3])
    # Nota: El target puede ser del propio bando o del oponente, por lo que evaluamos su ID de jugador
    target_slot_full = split_msg[3].split(':')[0].strip()  # Ej: "p1b" o "p2a"
    target_is_opponent = _is_opponent_slot(battle, target_slot_full)
    target_side = battle.opponent if target_is_opponent else battle.user
    target_slot = target_slot_full.endswith('a')
    target_pkmn = target_side.active1 if target_slot else target_side.active2

    # 3. Si ambos Pokémon existen en el campo, realizamos la copia de boosts
    if receiver_pkmn and target_pkmn and hasattr(receiver_pkmn, 'boosts') and hasattr(target_pkmn, 'boosts'):
        # Copiamos los valores asegurando que sea una copia independiente para evitar referencias compartidas
        receiver_pkmn.boosts = deepcopy(target_pkmn.boosts)

    return battle


############################################################################################################################################################
# OK - Intercambia boosts entre dos Pokémon (Guard Swap / Heart Swap / …)                                                                                ###
############################################################################################################################################################
def swapboost(battle: Battle, split_msg: list) -> Battle:
    # |-swapboost|p2a: Indeedee|p1b: Torkoal|def, spd|[from] move: Guard Swap
    # |-swapboost|p1a: Magearna|p2a: Incineroar|atk, def, spa, spd, spe, accuracy, evasion|[from] move: Heart Swap
    if len(split_msg) < 5:
        return battle

    source = _resolve_active_from_ident(battle, split_msg[2].strip())
    target = _resolve_active_from_ident(battle, split_msg[3].strip())
    if not source or not target:
        return battle
    if not hasattr(source, 'boosts') or not hasattr(target, 'boosts'):
        return battle

    raw_stats = split_msg[4].strip()
    if not raw_stats or raw_stats.startswith('['):
        return battle

    for token in raw_stats.split(','):
        abbr = token.strip().lower()
        if not abbr:
            continue
        key = constants.STAT_ABBREVIATION_LOOKUPS.get(abbr, abbr)
        src_val = int(source.boosts.get(key, 0) or 0)
        tgt_val = int(target.boosts.get(key, 0) or 0)
        source.boosts[key] = tgt_val
        target.boosts[key] = src_val

    print(
        f"[SWAPBOOST] {getattr(source, 'name', '?')} <-> {getattr(target, 'name', '?')} "
        f"stats={raw_stats}"
    )
    return battle


def _stat_for_guard_split(pkmn, actual_attr: str, base_attr: str):
    """Valor usable para promediar Guard Split (actual si >0; si no, base de especie)."""
    if pkmn is None:
        return None
    try:
        actual = int(getattr(pkmn, actual_attr, -1) or -1)
    except (TypeError, ValueError):
        actual = -1
    if actual > 0:
        return actual
    try:
        base = int(getattr(pkmn, base_attr, -1) or -1)
    except (TypeError, ValueError):
        base = -1
    return base if base > 0 else None


def _apply_guard_split(source, target) -> bool:
    """
    Promedia Def/SpD como PS (storedStats). Persistimos en *_battle_actual.
    Al salir del campo el rival vuelve a -1 (base) vía _clear_reserve_effects;
    el usuario se corrige con el siguiente |request|.
    """
    if source is None or target is None:
        return False

    src_def = _stat_for_guard_split(source, 'DEF_battle_actual', 'DEF_battle_base')
    tgt_def = _stat_for_guard_split(target, 'DEF_battle_actual', 'DEF_battle_base')
    src_spd = _stat_for_guard_split(source, 'SPD_battle_actual', 'SPD_battle_base')
    tgt_spd = _stat_for_guard_split(target, 'SPD_battle_actual', 'SPD_battle_base')

    applied = False
    if src_def is not None and tgt_def is not None:
        new_def = (src_def + tgt_def) // 2
        source.DEF_battle_actual = new_def
        target.DEF_battle_actual = new_def
        applied = True
    if src_spd is not None and tgt_spd is not None:
        new_spd = (src_spd + tgt_spd) // 2
        source.SPD_battle_actual = new_spd
        target.SPD_battle_actual = new_spd
        applied = True
    if applied:
        source.guard_split_active = True
        target.guard_split_active = True
    return applied


############################################################################################################################################################
# OK - Activación de efectos/movimientos especiales (Skill Swap, Guard Split, …)                                                                         ###
############################################################################################################################################################
def activate(battle: Battle, split_msg: list) -> Battle:
    # |-activate|p2b: Starmie|move: Skill Swap|Drought|Natural Cure|[of] p1b: Torkoal
    # |-activate|p2b: Indeedee|move: Guard Split|[of] p1b: Torkoal
    # |-activate|p1a: Indeedee|ability: Psychic Surge
    if len(split_msg) < 4:
        return battle

    trigger_effect = split_msg[3].strip().lower()

    # --- Skill Swap: intercambio de habilidades (PS incluye ambas habilidades) ---
    if 'skill swap' in trigger_effect:
        if len(split_msg) < 7:
            return battle

        pkmn1 = _resolve_active_from_ident(battle, split_msg[2].strip())
        of_part = _extract_of_ident(split_msg)
        if not of_part:
            return battle
        pkmn2 = _resolve_active_from_ident(battle, of_part)

        if pkmn1 and pkmn2:
            new_ability_pkmn1 = normalize_name(split_msg[4].strip())
            new_ability_pkmn2 = normalize_name(split_msg[5].strip())
            pkmn1.abilityPkmn = new_ability_pkmn1
            pkmn2.abilityPkmn = new_ability_pkmn2
            print(
                f"[SKILL SWAP] {pkmn1.name} ahora tiene {new_ability_pkmn1} | "
                f"{pkmn2.name} ahora tiene {new_ability_pkmn2}"
            )
        return battle

    # Flash Fire / Electromorphosis: |-activate|pXa: Mon|ability: Flash Fire
    if 'flash fire' in trigger_effect or 'electromorphosis' in trigger_effect:
        pkmn = _resolve_active_from_ident(battle, split_msg[2].strip())
        if pkmn is not None:
            pkmn.ability_on = True
        return battle

    # --- Guard Split: promediar Def / SpD (PS solo manda -activate, sin números) ---
    if 'guard split' in trigger_effect:
        source = _resolve_active_from_ident(battle, split_msg[2].strip())
        of_part = _extract_of_ident(split_msg)
        target = _resolve_active_from_ident(battle, of_part) if of_part else None
        if _apply_guard_split(source, target):
            print(
                f"[GUARD SPLIT] {getattr(source, 'name', '?')} <-> {getattr(target, 'name', '?')} "
                f"DEF={getattr(source, 'DEF_battle_actual', '?')} "
                f"SPD={getattr(source, 'SPD_battle_actual', '?')}"
            )
        return battle

    # Protect en un objetivo de spread: ese mon no recibe |-damage|; no tumbar
    # el pending entero (los otros hits del spread sí aportan SP).
    if any(pid in normalize_name(trigger_effect) for pid in _PROTECT_SINGLETURN):
        return battle

    # Sturdy / Disguise: daño truncado como Focus Sash.
    if 'sturdy' in trigger_effect or 'disguise' in trigger_effect:
        battle.pending_damage_sash = True

    return battle


############################################################################################################################################################
### OK - Actualiza el turno y decrementa contadores de clima, terrenos, trick room y estados temporales                                                  ###
############################################################################################################################################################
def turn(battle: Battle, split_msg: list) -> Battle:
    # |turn|1
    # |turn|3
    # Al llegar |turn|N (N>1) decrementamos contadores de clima / terrenos / side conditions.
    if len(split_msg) < 3:
        return battle

    # Actualizamos el número de turno actual de la batalla
    new_turn = int(split_msg[2].strip())
    battle.turn = new_turn

    # El orden de acción es evidencia de un solo turno: se reinicia en cada uno.
    battle.turn_actions = []
    battle.turn_speed_snapshot = None
    battle.pending_damage_ctx = None
    battle.pending_crit_target = None
    battle.pending_damage_sash = False

    # Decremento de duraciones (para AI). NO borrar clima/terreno aquí:
    # poke-env / PS limpian con |-weather|none y |-fieldend|; expirar en |turn|
    # un tick antes deja fields vacíos mientras PE aún los tiene (y al revés
    # con upkeep sticky). Mantener al menos 1 hasta el fin de protocolo.
    if new_turn > 1 and battle.weatherTurn > 1:
        battle.weatherTurn -= 1

    for terr, turns in list(battle.terrain.items()):
        left = int(turns or 0)
        if left <= 0:
            battle.terrain.pop(terr, None)
            continue
        if new_turn > 1:
            if left > 1:
                battle.terrain[terr] = left - 1
            elif left == 1:
                # PS / poke-env limpian al expirar; quedarse en 1 deja psychicterrain
                # fantasma cuando PE ya no tiene fields (robust e2e t8).
                battle.terrain.pop(terr, None)

    # Decremento de condiciones de bando (Side Conditions) y limpieza de estados
    for side in (battle.user, battle.opponent):
        # Aseguramos que side_conditions sea un diccionario válido para evitar caídas
        if not isinstance(side.side_conditions, dict):
            # Si se convirtió en una lista u otro objeto, lo restablecemos a un diccionario limpio
            side.side_conditions = defaultdict(lambda: 0)

        for condition in ('tailwind', 'reflect', 'lightscreen', 'auroraveil'):
            left = int(side.side_conditions.get(condition, 0) or 0)
            # Igual que clima/terreno: no borrar en 0; |-sideend| es la autoridad.
            if left > 1:
                side.side_conditions[condition] = left - 1

        # Yawn / is_drowsy: NO se limpia aquí. Debe sobrevivir al |turn| del turno
        # de decisión y solo caer con |-status|slp, |-end|Yawn, switch o faint.

    # Protect / Helping Hand / roost: solo viven dentro del turno que termina.
    # Destiny Bond y Charge NO se limpian aquí (cruzan |turn| hasta move/switch).
    _clear_all_ephemeral_turn_flags(battle)

    # Stakeout: just_switched solo vale el turno de entrada.
    for side in (battle.user, battle.opponent):
        for attr in ('active1', 'active2'):
            mon = getattr(side, attr, None)
            if mon is not None:
                mon.just_switched = False

    return battle


############################################################################################################################################################
# OK - Nombres de cuenta desde |player|p1|Nick|avatar|                                                                                                   ###
############################################################################################################################################################
def set_player(battle: Battle, split_msg: list) -> Battle:
    # |player|p1|Deep1ODDQN|167|
    # |player|p2|popplioperu98|winona|
    if len(split_msg) < 4:
        return battle

    player_id = split_msg[2].strip()
    account = split_msg[3].strip()
    if not account:
        return battle

    user_id = getattr(battle.user, 'id', None)
    opp_id = getattr(battle.opponent, 'id', None)

    if user_id and player_id == user_id:
        battle.user.account_name = account
    elif opp_id and player_id == opp_id:
        battle.opponent.account_name = account
    elif player_id.endswith('1') and not battle.user.account_name:
        # Antes de conocer side.id: p1 suele ser el retador o el host; se corrige luego
        battle.user.account_name = account
    elif player_id.endswith('2') and not battle.opponent.account_name:
        battle.opponent.account_name = account

    return battle


def _normalized_move_set(moves) -> set:
    return {normalize_name(m) for m in (moves or []) if m}


def _ots_moves_for_species(side, species_name: str) -> set:
    """Movimientos registrados en showteam/OTS para la especie (forma base)."""
    species = _base_species_name(species_name)
    if not species:
        return set()
    collected = set()
    for p in getattr(side, 'opponents', []) or []:
        if p is None:
            continue
        if _base_species_name(getattr(p, 'name', '')) == species:
            collected |= _normalized_move_set(getattr(p, 'moves', None))
    return collected


def _zoroark_still_available(side) -> bool:
    """True si Illusion sigue siendo posible (Zoroark vivo o aún no sacado)."""
    field_or_reserve = [
        getattr(side, 'active1', None),
        getattr(side, 'active2', None),
        *getattr(side, 'reserved', []),
    ]
    seen = [p for p in field_or_reserve if p is not None and _is_zoroark(p)]
    if seen:
        return any(not getattr(p, 'is_fainted', False) for p in seen)
    return any(_is_zoroark(p) for p in (getattr(side, 'opponents', []) or []))


def _ots_zoroark_details(side) -> str:
    for p in getattr(side, 'opponents', []) or []:
        if p is not None and _is_zoroark(p):
            raw = getattr(p, 'name', 'zoroark') or 'zoroark'
            # Showdown details usan mayúscula inicial / Hisui
            if raw.lower() == 'zoroark-hisui':
                return 'Zoroark-Hisui, L50'
            return 'Zoroark, L50'
    return 'Zoroark, L50'


############################################################################################################################################################
# OK - Observa |move|: si el mon oponente usa un move fuera de su OTS y hay Zoroark, deduce Illusion                                                     ###
############################################################################################################################################################
############################################################################################################################################################
### OK - Orden de acción del turno → evidencia de velocidad del rival                                                                                    ###
############################################################################################################################################################

# Efectos que reordenan el turno: si aparecen, el orden observado ya no refleja
# la velocidad y se descarta el turno completo.
_ORDER_ALTERING_EFFECTS = ('after you', 'quash', 'instruct')


def _speed_context_for(battle: Battle, side, pkmn) -> SpeedContext:
    """Modificadores de velocidad afirmables para un Pokémon en campo."""
    statuses = getattr(pkmn, 'non_volatile_statuses', None) or {}
    boosts = getattr(pkmn, 'boosts', None) or {}
    side_conditions = getattr(side, 'side_conditions', None) or {}
    spe_boost = int(
        boosts.get(constants.SPEED, 0)
        or boosts.get('spe', 0)
        or boosts.get('speed', 0)
        or 0
    )
    return SpeedContext(
        boost=spe_boost,
        item=getattr(pkmn, 'objectPkmn', None),
        ability=getattr(pkmn, 'abilityPkmn', None),
        paralyzed=bool(statuses.get('par')),
        tailwind=int(side_conditions.get('tailwind', 0) or 0) > 0,
    )


def _speed_obs_unreliable(battle: Battle, side, pkmn, ctx: SpeedContext) -> bool:
    """True si no conviene estrechar Spe (hazards / boosts desincronizados)."""
    if ctx.is_uncertain:
        return True
    side_conditions = getattr(side, 'side_conditions', None) or {}
    sticky = int(side_conditions.get('stickyweb', 0) or 0) > 0
    # Webs activas pero Spe>=0: probablemente no registramos el -1 de Sticky Web.
    if sticky and int(ctx.boost or 0) >= 0:
        return True
    return False


def _record_action_for_speed(battle: Battle, split_msg: list) -> None:
    """
    Anota que un Pokémon acaba de actuar y, contra lo ya anotado en este turno,
    estrecha la velocidad del rival.

    El contexto de velocidad se toma en el instante de cada |move| (no al inicio
    del turno): Tailwind / Trick Room aplicados mid-turn sí afectan a los que
    actúan después, y congelar el snapshot del primer move excluye Spe reales
    (p.ej. Kingambit 71 bajo TW del aliado vs Archaludon 104).
    """
    tags = ' '.join(split_msg[4:]).lower()

    # Un |move| con [from] es un movimiento llamado (Sleep Talk, Copycat...), no
    # la acción elegida: no aporta información de orden.
    if '[from]' in tags or '[still]' in tags:
        return
    if any(effect in tags for effect in _ORDER_ALTERING_EFFECTS):
        battle.turn_actions = []
        battle.turn_speed_snapshot = 'discarded'
        return
    if battle.turn_speed_snapshot == 'discarded':
        return

    slot_str = split_msg[2].split(':')[0].strip()
    if not slot_str:
        return
    from_opponent = _is_opponent_slot(battle, slot_str)
    side = battle.opponent if from_opponent else battle.user
    pkmn = side.active1 if slot_str.endswith('a') else side.active2
    if pkmn is None:
        return

    ability = getattr(pkmn, 'abilityPkmn', None)
    # Terreno activo (Grassy Glide +1 bajo Grassy Terrain).
    active_terrain = None
    for terr_name, turns in (battle.terrain or {}).items():
        if terr_name in ('trickroom', 'wonderroom', 'magicroom'):
            continue
        if int(turns or 0) > 0:
            active_terrain = terr_name
            break
    priority = move_priority(split_msg[3].strip(), ability, active_terrain)
    if priority is None:
        return  # movimiento desconocido: mejor no asumir prioridad 0

    # Marca de turno iniciado (para limpiar en |turn|); ya no congelamos TW/TR.
    if not battle.turn_actions:
        battle.turn_speed_snapshot = {'live': True}

    ctx = _speed_context_for(battle, side, pkmn)
    trick_room = int(battle.terrain.get('trickroom', 0) or 0) > 0

    current = {
        'slot': slot_str,
        'is_opponent': from_opponent,
        'pkmn': pkmn,
        'priority': priority,
        'ctx': ctx,
        'trick_room': trick_room,
    }

    for earlier in battle.turn_actions:
        # Solo comparamos bandos distintos y misma prioridad.
        if earlier['is_opponent'] == from_opponent:
            continue
        if earlier['priority'] != priority:
            continue
        # Si Trick Room se activó/desactivó entre las dos acciones, el orden
        # ya no es comparable con una sola regla.
        if bool(earlier.get('trick_room')) != trick_room:
            continue

        if earlier['is_opponent']:
            opponent_action, user_action, opponent_first = earlier, current, True
        else:
            opponent_action, user_action, opponent_first = current, earlier, False

        user_pkmn = user_action['pkmn']
        try:
            user_stat_speed = int(getattr(user_pkmn, 'VEL_battle_actual', -1) or -1)
        except (TypeError, ValueError):
            user_stat_speed = -1
        if user_stat_speed <= 0:
            continue

        if (
            _speed_obs_unreliable(battle, battle.opponent, opponent_action['pkmn'], opponent_action['ctx'])
            or _speed_obs_unreliable(battle, battle.user, user_action['pkmn'], user_action['ctx'])
        ):
            continue

        battle.stat_beliefs.observe_speed_order(
            opponent_action['pkmn'],
            opponent_action['ctx'],
            user_action['ctx'].speed_of(user_stat_speed),
            user_action['ctx'],
            opponent_first=opponent_first,
            trick_room=trick_room,
        )

    battle.turn_actions.append(current)


def note_crit(battle: Battle, split_msg: list = None) -> Battle:
    # |-crit|p1a: Farigiraf
    # Solo invalida el próximo |-damage| de ESE objetivo (spreads: los otros hits valen).
    # Si ya había un crit pendiente de otro mon, se acumula en un set.
    if split_msg and len(split_msg) >= 3:
        target = _resolve_active_from_ident(battle, split_msg[2].strip())
        prev = getattr(battle, 'pending_crit_target', None)
        if target is None:
            battle.pending_crit_target = True
        elif prev is True:
            pass  # ya invalida el próximo daño cualquiera
        elif prev is None:
            battle.pending_crit_target = target
        elif isinstance(prev, set):
            prev.add(id(target))
            battle.pending_crit_target = prev
        else:
            # Había un mon; ahora hay dos (spread con varios crits).
            battle.pending_crit_target = {id(prev), id(target)}
    else:
        battle.pending_crit_target = True  # legacy: siguiente daño cualquiera
    return battle


def _major_status_id(pkmn) -> str:
    statuses = getattr(pkmn, 'non_volatile_statuses', None) or {}
    for key in ('brn', 'par', 'slp', 'frz', 'psn', 'tox'):
        if statuses.get(key):
            return key
    return ''


def _boost_val(pkmn, const_key: str) -> int:
    boosts = getattr(pkmn, 'boosts', None) or {}
    try:
        return int(boosts.get(const_key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _side_has(side, condition: str) -> bool:
    conds = getattr(side, 'side_conditions', None) or {}
    try:
        return int(conds.get(condition, 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _active_terrain_name(battle: Battle) -> Optional[str]:
    for name in (
        'electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain',
    ):
        if int(battle.terrain.get(name, 0) or 0) > 0:
            return name
    return None


def _known_user_stats(pkmn) -> dict:
    def _v(attr, default=1):
        try:
            val = int(getattr(pkmn, attr, -1) or -1)
        except (TypeError, ValueError):
            return default
        return val if val > 0 else default

    return {
        'hp': _v('HPB_battle_base', 100),
        'atk': _v('ATK_battle_actual'),
        'def': _v('DEF_battle_actual'),
        'spa': _v('SPA_battle_actual'),
        'spd': _v('SPD_battle_actual'),
        'spe': _v('VEL_battle_actual'),
    }


def _illusion_blocks_damage_inference(pkmn) -> bool:
    """True si el mon es un disfraz Illusion aún opaco (no atribuir a esa especie)."""
    if pkmn is None:
        return False
    if getattr(pkmn, 'illusion_disguise', None) and not getattr(
        pkmn, 'illusion_protocol_revealed', False
    ):
        return not getattr(pkmn, 'is_zoroark', False)
    return False


def _build_hit_damage_context(
    battle: Battle,
    attacker,
    defender,
    *,
    is_spread: bool,
    defender_hp_fraction: Optional[float] = None,
) -> DamageContext:
    """Contexto afirmable para UN hit (screens/FG/boosts del defensor de ese golpe)."""
    weather = battle.weather if battle.weather and battle.weather != 'none' else None
    def_side = battle.user
    if defender is battle.opponent.active1 or defender is battle.opponent.active2:
        def_side = battle.opponent
    elif defender is battle.user.active1 or defender is battle.user.active2:
        def_side = battle.user
    elif getattr(defender, 'is_opponent', False):
        def_side = battle.opponent

    has_aurora = _side_has(def_side, 'auroraveil')
    frac = defender_hp_fraction
    if frac is None:
        frac = _hp_fraction_before_hit(defender)

    atk_side = battle.opponent if getattr(attacker, 'is_opponent', False) else battle.user
    if attacker is battle.opponent.active1 or attacker is battle.opponent.active2:
        atk_side = battle.opponent
    elif attacker is battle.user.active1 or attacker is battle.user.active2:
        atk_side = battle.user

    sun = normalize_name(str(weather or '')) in ('sunnyday', 'desolateland')
    flower = (
        normalize_name(getattr(attacker, 'abilityPkmn', '') or '') == 'flowergift'
        or _ally_has_ability(atk_side, attacker, ('flowergift',))
    ) and sun

    return DamageContext(
        atk_boost=_boost_val(attacker, constants.ATTACK),
        def_boost=_boost_val(defender, constants.DEFENSE),
        spa_boost=_boost_val(attacker, constants.SPECIAL_ATTACK),
        spd_boost=_boost_val(defender, constants.SPECIAL_DEFENSE),
        attacker_item=getattr(attacker, 'objectPkmn', None),
        defender_item=getattr(defender, 'objectPkmn', None),
        attacker_ability=getattr(attacker, 'abilityPkmn', None),
        defender_ability=getattr(defender, 'abilityPkmn', None),
        attacker_status=_major_status_id(attacker),
        defender_status=_major_status_id(defender),
        weather=weather,
        terrain=_active_terrain_name(battle),
        wonder_room=int(battle.terrain.get('wonderroom', 0) or 0) > 0,
        magic_room=int(battle.terrain.get('magicroom', 0) or 0) > 0,
        gravity=int(battle.terrain.get('gravity', 0) or 0) > 0,
        reflect=_side_has(def_side, 'reflect') and not has_aurora,
        light_screen=_side_has(def_side, 'lightscreen') and not has_aurora,
        aurora_veil=has_aurora,
        helping_hand=bool(getattr(attacker, 'is_helping_hand', False)),
        friend_guard=_ally_has_friend_guard(def_side, defender),
        charge=bool(getattr(attacker, 'is_charged', False)),
        is_spread=is_spread,
        is_crit=False,
        defender_hp_fraction=frac,
        attacker_ability_on=_ability_on_for_calc(attacker, side=atk_side, foe=defender),
        defender_ability_on=_ability_on_for_calc(defender, side=def_side, foe=attacker),
        attacker_types=_live_types(attacker),
        defender_types=_live_types(defender),
        attacker_tera=(
            getattr(attacker, 'tera_type', None)
            if getattr(attacker, 'is_terastallized', False) else None
        ),
        defender_tera=(
            getattr(defender, 'tera_type', None)
            if getattr(defender, 'is_terastallized', False) else None
        ),
        is_battery=_ally_has_ability(atk_side, attacker, ('battery',)),
        is_power_spot=_ally_has_ability(atk_side, attacker, ('powerspot',)),
        is_steely_spirit=_ally_has_ability(atk_side, attacker, ('steelyspirit',)),
        is_flower_gift=flower,
        is_beads_of_ruin=_active_has_ability(battle, ('beadsofruin',)),
        is_sword_of_ruin=_active_has_ability(battle, ('swordofruin',)),
        is_tablets_of_ruin=_active_has_ability(battle, ('tabletsofruin',)),
        is_vessel_of_ruin=_active_has_ability(battle, ('vesselofruin',)),
        attacker_allies_fainted=_count_side_fainted(atk_side),
    )


def _set_pending_damage_from_move(battle: Battle, split_msg: list) -> None:
    """
    Congela el ataque para el/los |-damage| siguientes.

    - Single-target: un hit cruzado; se limpia tras usarlo.
    - Spread ([spread]): varios |-damage| (foes y a veces aliados); cada hit
      cruzado aporta Atk/SpA o Def/SpD del rival involucrado.
    """
    tags = ' '.join(split_msg[4:]).lower()
    if '[from]' in tags or '[still]' in tags:
        battle.pending_damage_ctx = None
        battle.pending_crit_target = None
        return

    if len(split_msg) < 4:
        return

    move_name = split_msg[3].strip()
    category = move_category(move_name)
    if category not in ('p', 's'):
        battle.pending_damage_ctx = None
        battle.pending_crit_target = None
        return

    atk_slot = split_msg[2].split(':')[0].strip()
    if not atk_slot:
        return
    atk_opp = _is_opponent_slot(battle, atk_slot)
    atk_side = battle.opponent if atk_opp else battle.user
    attacker = atk_side.active1 if atk_slot.endswith('a') else atk_side.active2
    if attacker is None:
        return

    if _illusion_blocks_damage_inference(attacker):
        battle.pending_damage_ctx = None
        battle.pending_crit_target = None
        return

    # Nuevo golpe: limpia crits del move anterior.
    battle.pending_crit_target = None
    battle.pending_damage_sash = False
    is_spread = _move_is_spread(move_name, tags)

    battle.pending_damage_ctx = {
        'move_name': move_name,
        'category': category,
        'attacker': attacker,
        'opponent_is_attacker': atk_opp,
        'is_spread': is_spread,
        # Charge se consume en observe_move tras el |move|; congelar aquí.
        'attacker_charge': bool(getattr(attacker, 'is_charged', False)),
        'attacker_helping_hand': bool(getattr(attacker, 'is_helping_hand', False)),
        # Hits ya usados (id del objeto Pokemon) para no doblar el mismo mon.
        'hit_ids': set(),
    }


def _maybe_infer_from_damage(battle: Battle, split_msg: list, pkmn, old_hp: int, old_max: int) -> None:
    """Tras aplicar |-damage|, intenta estrechar Atk/SpA/Def/SpD del rival."""
    pending = getattr(battle, 'pending_damage_ctx', None)
    if not pending:
        return

    # Focus Sash/Band o Sturdy: el daño observado está truncado a 1 HP.
    if getattr(battle, 'pending_damage_sash', False):
        battle.pending_damage_sash = False
        if not pending.get('is_spread'):
            battle.pending_damage_ctx = None
        return

    # Residuales / items: no son el golpe del move pendiente.
    from_parts = [part.lower() for part in split_msg[4:]]
    if any('[from]' in part for part in from_parts):
        return

    # Crit solo para este objetivo (no tumbar el flag de otro mon del spread).
    crit_target = getattr(battle, 'pending_crit_target', None)
    if crit_target is not None:
        is_this_crit = False
        if crit_target is True:
            is_this_crit = True
            battle.pending_crit_target = None
        elif isinstance(crit_target, set):
            if id(pkmn) in crit_target:
                is_this_crit = True
                crit_target.discard(id(pkmn))
                battle.pending_crit_target = crit_target or None
        elif crit_target is pkmn:
            is_this_crit = True
            battle.pending_crit_target = None
        if is_this_crit:
            # Spread: mantener pending para otros hits no-crit.
            if not pending.get('is_spread'):
                battle.pending_damage_ctx = None
            return

    attacker = pending.get('attacker')
    if attacker is None or pkmn is None:
        return

    atk_opp = bool(pending['opponent_is_attacker'])
    def_opp = bool(getattr(pkmn, 'is_opponent', False))
    # Preferir identidad de slot activa sobre el flag del objeto.
    if pkmn is battle.user.active1 or pkmn is battle.user.active2:
        def_opp = False
    elif pkmn is battle.opponent.active1 or pkmn is battle.opponent.active2:
        def_opp = True

    # Solo hits cruzados (atacante rival ↔ defensor nuestro, o al revés).
    if atk_opp == def_opp:
        return

    if _illusion_blocks_damage_inference(pkmn):
        return

    if getattr(pkmn, 'is_protected', False):
        return

    volatiles = getattr(pkmn, 'volatile_statuses', None) or {}
    if volatiles.get('substitute'):
        return

    hit_ids = pending.setdefault('hit_ids', set())
    pid = id(pkmn)
    if pid in hit_ids:
        return
    hit_ids.add(pid)

    try:
        new_hp = int(getattr(pkmn, 'HPA_battle_actual', 0) or 0)
        new_max = int(getattr(pkmn, 'HPB_battle_base', 0) or 0)
    except (TypeError, ValueError):
        return

    opponent_is_attacker = atk_opp
    damage_is_percent = opponent_is_attacker is False

    if damage_is_percent:
        if old_hp <= 0 or new_max <= 0:
            return
        delta_pct = max(0, int(old_hp) - int(new_hp))
        if pkmn.is_fainted or new_hp <= 0:
            delta_pct = max(delta_pct, 1)
        observed = delta_pct
    else:
        if old_hp <= 0:
            return
        observed = max(0, int(old_hp) - int(new_hp))
        if observed <= 0 and (pkmn.is_fainted or new_hp <= 0):
            observed = int(old_hp)

    if observed <= 0:
        return

    is_ko = bool(getattr(pkmn, 'is_fainted', False) or new_hp <= 0)

    # Fracción de HP ANTES del golpe (Multiscale); old_hp ya lo capturó heal_or_damage.
    pre_frac = None
    try:
        if def_opp:
            pre_frac = max(0.0, min(1.0, float(old_hp) / 100.0))
        elif old_max > 0:
            pre_frac = max(0.0, min(1.0, float(old_hp) / float(old_max)))
    except (TypeError, ValueError):
        pre_frac = None

    ctx = _build_hit_damage_context(
        battle,
        attacker,
        pkmn,
        is_spread=bool(pending.get('is_spread')),
        defender_hp_fraction=pre_frac,
    )
    # Snapshot del |move|: Charge/HH pueden caer antes del |-damage|.
    if pending.get('attacker_charge'):
        ctx.charge = True
    if pending.get('attacker_helping_hand'):
        ctx.helping_hand = True

    user_mon = pkmn if opponent_is_attacker else attacker
    opp_mon = attacker if opponent_is_attacker else pkmn
    known_stats = _known_user_stats(user_mon)

    battle.stat_beliefs.observe_damage(
        opponent_pkmn=opp_mon,
        move_name=pending['move_name'],
        category=pending['category'],
        opponent_is_attacker=opponent_is_attacker,
        observed_damage=observed,
        damage_is_percent=damage_is_percent,
        known_stats=known_stats,
        known_species=getattr(user_mon, 'name', '') or 'Magikarp',
        known_nature=getattr(user_mon, 'nature', None),
        ctx=ctx,
        known_hp=known_stats.get('hp'),
        is_ko=is_ko,
    )

    # Single-target: un solo hit útil. Spread: seguir hasta el próximo |move|/|turn|.
    if not pending.get('is_spread'):
        battle.pending_damage_ctx = None


def observe_move(battle: Battle, split_msg: list) -> Battle:
    # |move|p2a: Dragonite|Calm Mind|p2a: Dragonite
    # |move|p2b: Pawmot|Coaching|p2a: Dragonite
    # Típico T1: el "lead" usa un move del set de Zoroark (no del disfraz) → Illusion
    if len(split_msg) < 4:
        return battle

    actor = _resolve_active_from_ident(battle, split_msg[2].strip())
    move_name = split_msg[3].strip()
    move_nid = normalize_name(move_name)
    still = any(part.strip().lower() == '[still]' for part in split_msg[4:])

    # Destiny Bond cae al usar otro move (no al propio Destiny Bond).
    if (
        actor is not None
        and not still
        and getattr(actor, 'is_destiny_bonded', False)
        and move_nid not in _DESTINY_BOND_IDS
    ):
        _set_destiny_bonded(actor, False)

    # El orden de acción se registra para AMBOS bandos: es la evidencia con la
    # que se estrecha la velocidad del rival.
    _record_action_for_speed(battle, split_msg)
    # Congela contexto de daño para el |-damage| inmediato del move.
    _set_pending_damage_from_move(battle, split_msg)

    # Charge se consume tras un move Eléctrico (después de congelar el ctx).
    if (
        actor is not None
        and not still
        and getattr(actor, 'is_charged', False)
        and _move_is_electric(move_name)
    ):
        _set_charged(actor, False)

    # mustrecharge cae al actuar (turno de recarga o cualquier move).
    if actor is not None and not still and getattr(actor, 'must_recharge', False):
        _set_must_recharge(actor, False)

    # Solo oponente: el agente no usa Illusion de forma opaca en nuestro estado
    if not is_opponent(battle, split_msg):
        return battle

    # Ignorar moves que no llegaron a ejecutarse
    if any(part.strip().lower() == '[still]' for part in split_msg[4:]):
        return battle

    move_id = normalize_name(split_msg[3].strip())
    if not move_id or move_id == 'struggle':
        return battle

    side = battle.opponent
    slot_str = split_msg[2].split(':')[0].strip()
    is_active1 = slot_str.endswith('a')
    pkmn = side.active1 if is_active1 else side.active2
    if pkmn is None:
        return battle

    # Ya revelado / ya tratado como Zoroark
    if _is_zoroark(pkmn) or _canonical_species_name(getattr(pkmn, 'name', '')).startswith('zoroark'):
        return battle

    if not _zoroark_still_available(side):
        return battle

    known_moves = _ots_moves_for_species(side, getattr(pkmn, 'name', ''))
    if not known_moves:
        known_moves = _normalized_move_set(getattr(pkmn, 'moves', None))
    if not known_moves:
        return battle

    if move_id in known_moves:
        return battle

    # Move ilegal para la especie OTS → debe ser Illusion (Champions con showteam)
    print(
        f"[ILLUSION DEDUCED BY MOVE] {getattr(pkmn, 'name', '?')} uso '{move_id}' "
        f"(no esta en OTS {sorted(known_moves)}) -> Zoroark"
    )
    details = _ots_zoroark_details(side)
    return replace_illusion(
        battle,
        ['', 'replace', f'{slot_str}: Zoroark', details],
        protocol_reveal=False,
    )


def must_recharge(battle: Battle, split_msg: list) -> Battle:
    # |-mustrecharge|p1a: Archaludon
    if len(split_msg) < 3:
        return battle
    pkmn = _resolve_active_from_ident(battle, split_msg[2].strip())
    if pkmn is not None:
        _set_must_recharge(pkmn, True)
    return battle


############################################################################################################################################################
### OK - Lookup para cada una de las funciones de actualización de entorno                                                                               ###
############################################################################################################################################################
BATTLE_MODIFIERS_LOOKUP = {
    'request': request,
    'player': set_player,
    'switch': switch_or_drag,
    'faint': faint,
    'drag': switch_or_drag,
    'swap': swap,
    'move': observe_move,
    '-heal': heal_or_damage,
    '-damage': heal_or_damage,
    '-crit': note_crit,
    '-boost': boost,
    '-unboost': unboost,
    '-status': status,
    '-start': start_volatile_status,
    '-end': end_volatile_status,
    '-singleturn': singleturn,
    '-singlemove': singlemove,
    '-curestatus': curestatus,
    '-weather': weather,
    '-fieldstart': fieldstart,
    '-fieldend': fieldend,
    '-sidestart': sidestart,
    '-sideend': sideend,
    '-item': set_item,
    '-enditem': remove_item,
    '-ability': set_opponent_ability_from_ability_tag,
    'detailschange': form_change,
    'replace': replace_illusion,
    '-formechange': form_change,
    '-transform': transform,
    '-mega': mega,
    '-terastallize': terastallize,
    '-clearallboost': clearallboost,
    '-clearboost': clearboost,
    '-clearnegativeboost': clearnegativeboost,
    '-clearpositiveboost': clearpositiveboost,
    '-copyboost': copyboost,
    '-swapboost': swapboost,
    '-activate': activate,
    '-mustrecharge': must_recharge,
    'turn': turn
}


############################################################################################################################################################
### OK - Actualiza el entorno de batalla procesando los mensajes de actualización del entorno y tipo |request|                                           ###
############################################################################################################################################################
async def update_battle_environment(battle: Battle, msg: str, msg_request: str) -> None:
    full_msg = f"{msg}\n{msg_request}" if msg_request else msg # Se concatena ambos mensajes recibidos para procesar todas las líneas en un solo flujo
    msg_lines = full_msg.split('\n')

    for line in msg_lines: # Para cada una de las líneas de la concatenación
        split_msg = line.split('|')
        # Protocolo Showdown: "|ACTION|..." → al menos ['', 'ACTION', ...]
        if len(split_msg) < 2:
            continue

        action = split_msg[1].strip() # Se obtiene la acción

        function_to_call = BATTLE_MODIFIERS_LOOKUP.get(action) # Para obtener su función correspondiente
        if function_to_call is not None:
            battle = function_to_call(battle, split_msg) # Y se ejecuta...
            battle.error = False # Cada vez que se actualiza el entorno, entonces la acción anteriormente ejecutada no ha caído en Error