import json
import math
import constants
import re

from copy import deepcopy
from collections import defaultdict
from teams.team_converter import normalize_name
from showdown.battle import Pokemon
from showdown.battle import Battle

'''
    MEJORAS PENDIENTES:
    - TAL VEZ HAYA ALGUN MODIFICADOR PENDIENTE
'''

def _parse_hp_and_status_from_switch(hp_status_str: str):
    """
    Extrae (hpa, hpb, status_dict) desde la cadena de condición de un |switch| o |drag|.
    Ejemplos de entrada:
      '190/202'       -> (190, 202, {})
      '28/198 par'    -> (28, 198, {'par': True})
      '0 fnt'         -> (0, 0, {})
    """
    hp_status_str = hp_status_str.strip()
    
    if constants.FNT in hp_status_str or hp_status_str.startswith("0"):
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

    return 0, 0, {}


############################################################################################################################################################
### OK - Devuelve el valor booleando indicativo si la actualización del entorno afecta al oponente o al usuario                                          ###
############################################################################################################################################################
def is_opponent(battle: Battle,  split_msg: list) -> bool:
    # split_msg = ["","switch","p1a: Incineroar","Incineroar, L50, M","190/202"]
    # split_msg[2] = "p1a: Incineroar"
    return not split_msg[2].startswith(battle.user.id)


############################################################################################################################################################
### OK - Identifica el tipo de |request| y le asigna un valor positivo en caso sea el tipo correspondiente: active, forceSwitch & wait                   ###
############################################################################################################################################################
def request(battle: Battle, split_msg: list) -> Battle:
    
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

    # Se almacena el JSON para elegir una acción
    if not battle.wait:
        battle.request_json = battle_json

    return battle


############################################################################################################################################################
### OK - Se realiza el cambio de un PKMN activo a uno de reserva                                                                                         ###
############################################################################################################################################################
def switch_or_drag(battle: Battle, split_msg: list) -> Battle:
    # |switch|p1a: Incineroar|Incineroar, L50, M|190/202
    # |switch|p2a: Dragonite|Dragonite, L50, M|28/198 par|[from] Flip Turn
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

        # 1. PASO A: Buscamos PRIMERO si el Pokémon que entra YA ESTÁ en la reserva (reserved)
        found_in_reserved_idx = -1
        for contador, pkmn in enumerate(side.reserved):
            if pkmn is not None and namePkmn == pkmn.name:
                found_in_reserved_idx = contador
                break

        # Si ya estaba en la reserva, hacemos un intercambio directo (Swap)
        if found_in_reserved_idx != -1:
            pkmnAux = deepcopy(current_active)
            pkmnAux = _revert_special_pokemon(pkmnAux, side)
            
            pkmn_from_reserved = side.reserved[found_in_reserved_idx]
            new_active = deepcopy(pkmn_from_reserved)
            
            # AUTORIDAD DEL LOG: Actualizamos HP y Estado no Volátil con la info del mensaje
            new_active.HPA_battle_actual = hpa_val if hpa_val > 0 else new_active.HPA_battle_actual
            if hpb_val > 0:
                new_active.HPB_battle_base = hpb_val
            if new_non_volatile:
                new_active.non_volatile_statuses = new_non_volatile

            new_active.volatile_statuses = defaultdict(lambda: 0)
            new_active.boosts = defaultdict(lambda: 0)

            if is_active1:
                side.active1 = new_active
            else:
                side.active2 = new_active

            # El Pokémon saliente ocupa el lugar de la reserva que dejó el entrante
            if pkmnAux is not None:
                side.reserved[found_in_reserved_idx] = deepcopy(pkmnAux)
            print(f"El oponente ha cambiado de Pokémon a {new_active.name} ({new_active.HPA_battle_actual}/{new_active.HPB_battle_base}) : {slot_str}")

        # 2. PASO B: Si NO estaba en la reserva, lo buscamos en 'opponents' (Team Preview)
        else:
            for pkmn in side.opponents:
                if pkmn is not None and namePkmn == pkmn.name:
                    pkmnAux = deepcopy(current_active)
                    pkmnAux = _revert_special_pokemon(pkmnAux, side)
                    
                    new_active = deepcopy(pkmn)
                    new_active.name = namePkmn
                    
                    # AUTORIDAD DEL LOG: Asignamos HP y Estado no Volátil traídos por el switch
                    new_active.HPA_battle_actual = hpa_val if hpa_val > 0 else getattr(pkmn, 'HPA_battle_actual', 0)
                    if hpb_val > 0:
                        new_active.HPB_battle_base = hpb_val
                    if new_non_volatile:
                        new_active.non_volatile_statuses = new_non_volatile

                    new_active.volatile_statuses = defaultdict(lambda: 0)
                    new_active.boosts = defaultdict(lambda: 0)

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
                        if not any(p is pkmnAux for p in side.reserved if p is not None):
                            side.reserved.append(deepcopy(pkmnAux))
                    break

    # =========================================================================
    # CASO 2: CAMBIO REALIZADO POR EL USUARIO
    # =========================================================================
    else:
        for contador, pkmn in enumerate(side.reserved):
            if pkmn is not None and namePkmn == pkmn.name:
                aux = deepcopy(side.active1 if is_active1 else side.active2)
                aux = _revert_special_pokemon(aux, side)
                
                new_active = deepcopy(pkmn)
                new_active.name = namePkmn
                new_active.volatile_statuses = defaultdict(lambda: 0)
                new_active.boosts = defaultdict(lambda: 0)

                if is_active1:
                    side.active1 = new_active
                else:
                    side.active2 = new_active

                # Intercambio en Reservados
                side.reserved[contador] = deepcopy(aux)
                break

        # Limpieza de nulos en reservados del usuario
        side.reserved = [p for p in side.reserved if p is not None]

    return battle


############################################################################################################################################################
### OK - Marcar Pokémon debilitado                                                                                                                       ###
############################################################################################################################################################
def faint(battle: Battle, split_msg: list) -> Battle:
    # |faint|p1a: Tornadus
    opponent_turn = is_opponent(battle, split_msg)
    side = battle.opponent if opponent_turn else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    
    # Identificamos si es la posición A o B
    if slot_str.endswith('a'):
        pkmn = side.active1
    else:
        pkmn = side.active2

    if pkmn:
        pkmn.HPA_battle_actual = 0
        pkmn.is_fainted = True
        pkmn.boosts = defaultdict(lambda: 0)
        pkmn.volatile_statuses = defaultdict(lambda: 0)
        pkmn.non_volatile_statuses = defaultdict(lambda: 0)

    return battle


############################################################################################################################################################
### OK - Actualización de HP por daño o curación                                                                                                         ###
############################################################################################################################################################
def heal_or_damage(battle: Battle, split_msg: list) -> Battle:
    # |-damage|p2a: Zoroark|40/100 brn|[from] brn
    # |-damage|p2a: Tornadus|91/100|[from] item: Life Orb
    # |-damage|p1a: Raichu|135/167 tox|[from] psn
    # |-damage|p2a: Blastoise|50/100y
    if len(split_msg) < 4:
        return battle

    opponent_turn = is_opponent(battle, split_msg)
    side = battle.opponent if opponent_turn else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if not pkmn:
        return battle

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

    return battle


############################################################################################################################################################
### OK - Intercambio de posiciones activas en el campo (ej: Ally Switch)                                                                                 ###
############################################################################################################################################################
def swap(battle: Battle, split_msg: list) -> Battle:
    # |swap|p1a: Gothitelle|1|[from] move: Ally Switch
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
    if len(split_msg) < 4:
        return battle

    status_name = split_msg[3].strip().lower()

    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn and hasattr(pkmn, 'non_volatile_statuses'):
        # Si non_volatile_statuses es un dict o defaultdict, guardamos el estado con True
        if isinstance(pkmn.non_volatile_statuses, (dict, defaultdict)):
            pkmn.non_volatile_statuses[status_name] = True
        else:
            # Fallback por seguridad si en algún momento fue un string
            pkmn.non_volatile_statuses = {status_name: True}

    return battle


############################################################################################################################################################
# OK - Inicia alterado volatil (ej: 'moveyawn', 'confusion', 'substitute')                                                                               ###
############################################################################################################################################################
def start_volatile_status(battle: Battle, split_msg: list) -> Battle:
    # Ejemplo de mensaje: |-start|p2a: Incineroar|move: Yawn|[of] p1a: Gastrodon
    # Ejemplo de mensaje: |-start|p1b: Whimsicott|confusion
    
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

        # Registramos el estado volátil en el diccionario
        pkmn.volatile_statuses[volatile_status] = True

        # Lógica adicional para Substituto
        if volatile_status == 'substitute':
            pkmn.is_substitute = True

    return battle


############################################################################################################################################################
# OK - Fin alterado volatil (ej: 'moveyawn')                                                                                                             ###
############################################################################################################################################################
def end_volatile_status(battle: Battle, split_msg: list) -> Battle:
    # |-end|p1a: Aerodactyl|Encore
    # |-end|p2a: Zoroark|Illusion
    
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
            
        # Lógica adicional para Substituto
        if volatile_status == 'substitute':
            pkmn.is_substitute = False

    return battle


############################################################################################################################################################
# OK - Revelación de Illusion (Zoroark sustituye al disfraz en el campo)                                                                                 ###
############################################################################################################################################################
def replace_illusion(battle: Battle, split_msg: list) -> Battle:
    # |replace|p2a: Zoroark|Zoroark, L50, M

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
    if disguised_active and disguised_active.name != real_name:
        disguised_active.is_fainted = False
        disguised_active.volatile_statuses = defaultdict(lambda: 0)
        disguised_active.non_volatile_statuses = defaultdict(lambda: 0)
        disguised_active.boosts = defaultdict(lambda: 0)
        
        # Si el disfraz no estaba en la reserva, lo regresamos
        if not any(p and p.name == disguised_active.name for p in side.reserved):
            side.reserved.append(disguised_active)

    # 2. Buscamos la ficha completa de Zoroark en 'opponents' (OTS/Preview) para recuperar Objeto, Habilidad y Naturaleza
    ots_zoroark = None
    for p in side.opponents:
        if p and p.name == real_name:
            ots_zoroark = p
            break

    # 3. Buscamos a Zoroark en 'reserved' si ya existía allí
    zoroark_instance = None
    zoroark_idx = -1
    for i, p in enumerate(side.reserved):
        if p and p.name == real_name:
            zoroark_instance = p
            zoroark_idx = i
            break

    if zoroark_instance:
        # Si estaba en la reserva, lo removemos de allí porque pasa al campo activo
        side.reserved.pop(zoroark_idx)
    else:
        # Si es del oponente o no estaba instanciado, creamos la nueva instancia con los datos de ots_zoroark
        obj_item = normalize_name(getattr(ots_zoroark, 'objectPkmn', '') if ots_zoroark else '')
        obj_ability = normalize_name(getattr(ots_zoroark, 'abilityPkmn', '') if ots_zoroark else '')
        obj_nature = normalize_name(getattr(ots_zoroark, 'nature', None) if ots_zoroark else None)
        
        # Asignamos los 4 movimientos si existen en ots_zoroark
        m1 = ots_zoroark.moves[0] if ots_zoroark and len(ots_zoroark.moves) > 0 else None
        m2 = ots_zoroark.moves[1] if ots_zoroark and len(ots_zoroark.moves) > 1 else None
        m3 = ots_zoroark.moves[2] if ots_zoroark and len(ots_zoroark.moves) > 2 else None
        m4 = ots_zoroark.moves[3] if ots_zoroark and len(ots_zoroark.moves) > 3 else None

        zoroark_instance = Pokemon(real_name, 0, opponent_turn, obj_item, obj_ability, m1, m2, m3, m4, obj_nature)
        
        if opponent_turn:
            zoroark_instance.is_opponent_confirmed = True
            if not getattr(zoroark_instance, 'is_opponent_confirmed', False):
                side.opponentPkmnConfirmed += 1

    # 4. Asignamos Datos Reales de Zoroark (Objeto, Habilidad, Naturaleza y Stats Base)
    if ots_zoroark:
        zoroark_instance.objectPkmn = getattr(ots_zoroark, 'objectPkmn', zoroark_instance.objectPkmn)
        zoroark_instance.abilityPkmn = getattr(ots_zoroark, 'abilityPkmn', zoroark_instance.abilityPkmn)
        zoroark_instance.nature = getattr(ots_zoroark, 'nature', zoroark_instance.nature)
        
        # RESTAURACIÓN DE STATS BASE REALES DE ZOROARK (Descartando las del disfraz)
        zoroark_instance.HPB_battle_base = getattr(ots_zoroark, 'HPB_battle_base', current_hpb)
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

    # Asignamos a Zoroark como el activo real
    if is_active1:
        side.active1 = zoroark_instance
    else:
        side.active2 = zoroark_instance

    print(f"[ILLUSION REVEALED] {real_name} se ha revelado en {slot_str} con HP: {current_hpa}")

    return battle


############################################################################################################################################################
# OK - Curación de estado alterado                                                                                                                       ###
############################################################################################################################################################
def curestatus(battle: Battle, split_msg: list) -> Battle:
    # |-curestatus|p2a: Kyogre|slp|[msg]
    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn:
        pkmn.status = None

    return battle


############################################################################################################################################################
# OK - Establecimiento / eliminación de clima                                                                                                            ###
############################################################################################################################################################
def weather(battle: Battle, split_msg: list) -> Battle:
    # |-weather|Sandstorm
    # |-weather|none
    # |-weather|Snowscape|[upkeep]

    raw_weather = split_msg[2].split(':')[-1].strip()
    weather_name = normalize_name(raw_weather)

    battle.weather = weather_name
    battle.weatherTurn = 0 if weather_name in ("none", "null") else 5
    return battle


############################################################################################################################################################
# OK - Inicio de Terreno, incluye Trick Room                                                                                                             ###
############################################################################################################################################################
def fieldstart(battle: Battle, split_msg: list) -> Battle:
    # |-fieldstart|move: Wonder Room|[of] p1b: Starmie
    # |-fieldstart|move: Electric Terrain
    # |-fieldstart|move: Trick Room|[of] p2b: Farigiraf

    raw_field = split_msg[2].split(':')[-1].strip()
    field_name = normalize_name(raw_field)
    battle.terrain[field_name] = 5

    return battle


############################################################################################################################################################
# OK - Fin de Terreno, incluye Trick Room                                                                                                                ###
############################################################################################################################################################
def fieldend(battle: Battle, split_msg: list) -> Battle:
    # |-fieldend|move: Trick Room

    raw_field = split_msg[2].split(':')[-1].strip()
    field_name = normalize_name(raw_field)
    del battle.terrain[field_name]

    return battle


############################################################################################################################################################
# OK - Activación de efectos de bando (Tailwind, Reflect, Light Screen, etc.)                                                                            ###
############################################################################################################################################################
def sidestart(battle: Battle, split_msg: list) -> Battle:
    # |-sidestart|p1: PopplioPeru98|move: Tailwind
    raw_condition = split_msg[3].split(':')[-1].strip()
    condition = normalize_name(raw_condition)

    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    side.side_conditions[condition] = 5
    return battle


############################################################################################################################################################
# OK - Finalización de efectos de bando (Tailwind, Reflect, Light Screen, etc.)                                                                          ###
############################################################################################################################################################
def sideend(battle: Battle, split_msg: list) -> Battle:
    # |-sideend|p2: PopplioPeru98|move: Tailwind
    raw_condition = split_msg[3].split(':')[-1].strip()
    condition = normalize_name(raw_condition)

    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    side.side_conditions[condition] = 0
    return battle


############################################################################################################################################################
# OK - Establece o cambia el objeto equipado de un Pokémon (ej: por movimientos como Trick o Switcheroo)                                               ###
############################################################################################################################################################
def set_item(battle: Battle, split_msg: list) -> Battle:
    # |-item|p2a: Torkoal|Choice Scarf|[from] move: Trick
    # |-item|p1b: Ditto|Sitrus Berry|[from] move: Trick
    
    # Validamos que el mensaje tenga la longitud mínima para extraer el slot y el objeto
    if len(split_msg) < 4:
        return battle

    # Extraemos el nombre del objeto (ej: "Choice Scarf") y lo normalizamos si es necesario
    item_name = split_msg[3].strip()

    # Determinamos a qué bando y Pokémon activo pertenece el cambio de objeto
    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn and hasattr(pkmn, 'objectPkmn'):
        pkmn.objectPkmn = item_name

    return battle


############################################################################################################################################################
# OK - Remueve o consume el objeto equipado de un Pokémon (ej: Knock Off, Focus Sash gastado, Baya)                                                      ###
############################################################################################################################################################
def remove_item(battle: Battle, split_msg: list) -> Battle:
    # |-enditem|p1b: Kingambit|Focus Sash
    # |-enditem|p2a: Rillaboom|Grassy Seed|[from] item: Grassy Seed
    # |-enditem|p1b: Ditto|Sitrus Berry|[eat]
    
    # Validamos que el mensaje tenga la longitud mínima requerida
    if len(split_msg) < 3:
        return battle

    # Determinamos a qué bando y Pokémon activo pertenece la pérdida del objeto
    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn and hasattr(pkmn, 'objectPkmn'):
        # Limpiamos el objeto equipado del Pokémon
        pkmn.objectPkmn = None

    return battle


############################################################################################################################################################
# OK - Establece la habilidad revelada o cambiada de un Pokémon (ej: al entrar al campo con Intimidate, Drizzle, Unnerve, o por Skill Swap)              ###
############################################################################################################################################################
def set_opponent_ability_from_ability_tag(battle: Battle, split_msg: list) -> Battle:
    # Ejemplos de mensaje de Showdown:
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
        pkmn.abilityPkmn = ability_name

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
        # Conservamos estados volátiles previos (si los hubiera) e instanciamos la nueva forma
        aux_volatile = current_active.volatile_statuses
        
        # Creamos la nueva instancia del Pokémon con su nueva forma (ej. Aerodactyl-Mega)
        # Pasamos max_hp=0 para que recalcule sus estadísticas base de combate mediante getStatsBattle()
        new_pkmn = Pokemon(name_pkmn, 0, opponent_turn, current_active.objectPkmn, current_active.abilityPkmn, current_active.moves[0], current_active.moves[1], current_active.moves[2], current_active.moves[3], getattr(current_active, 'nature', None))
        
        # Mantenemos los HP actuales proporcionales o heredados y los estados volátiles
        new_pkmn.HPB_battle_base = current_active.HPB_battle_base
        new_pkmn.volatile_statuses = aux_volatile

        # Asignamos el nuevo Pokémon al slot activo correspondiente
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
    target_is_opponent = not target_slot_full.startswith(battle.user.id)
    target_side = battle.opponent if target_is_opponent else battle.user
    target_is_active1 = target_slot_full.endswith('a')
    target_pkmn = target_side.active1 if target_is_active1 else target_side.active2

    # 3. Si ambos existen, realizamos la transformación
    if source_pkmn and target_pkmn:
        # Guardamos la información única de Ditto que NUNCA cambia al transformarse
        original_item = source_pkmn.objectPkmn
        original_hpa = source_pkmn.HPA_battle_actual
        original_hpb = source_pkmn.HPB_battle_base

        # Creamos una copia exacta de los atributos del objetivo (tipos, habilidad, stats base, movimientos, etc.)
        new_source = deepcopy(target_pkmn)

        # MODIFICACIÓN 1: Restauramos el objeto propio de Ditto (no copia el del objetivo)
        new_source.objectPkmn = original_item

        # Mantenemos los HP originales de Ditto (HP no cambia en la transformación)
        new_source.HPA_battle_actual = original_hpa
        new_source.HPB_battle_base = original_hpb

        # MODIFICACIÓN 2: Reglas de estadísticas personalizadas para Ditto
        if not transform_opponent:  # Caso A: El Ditto es del BOT (Usuario)
            # Copia estadísticas base del oponente (ya heredadas en 'deepcopy')
            # Sus estadísticas actuales se llenan con -1
            new_source.ATK_battle_actual = -1
            new_source.DEF_battle_actual = -1
            new_source.SPA_battle_actual = -1
            new_source.SPD_battle_actual = -1
            new_source.VEL_battle_actual = -1
        else:                      # Caso B: El Ditto es del OPONENTE
            # Sus estadísticas base se convierten en -1
            new_source.ATK_battle_base = -1
            new_source.DEF_battle_base = -1
            new_source.SPA_battle_base = -1
            new_source.SPD_battle_base = -1
            new_source.VEL_battle_base = -1
            # Sus estadísticas actuales son las mías del Pokémon que copió (ya heredadas en 'deepcopy')

        # Marcamos explícitamente que es Ditto transformado
        new_source.is_ditto = True

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
    # Ejemplo de mensaje: |-mega|p1a: Aerodactyl|Aerodactyl|Aerodactylite
    
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
    target_is_opponent = not target_slot_full.startswith(battle.user.id)
    target_side = battle.opponent if target_is_opponent else battle.user
    target_slot = target_slot_full.endswith('a')
    target_pkmn = target_side.active1 if target_slot else target_side.active2

    # 3. Si ambos Pokémon existen en el campo, realizamos la copia de boosts
    if receiver_pkmn and target_pkmn and hasattr(receiver_pkmn, 'boosts') and hasattr(target_pkmn, 'boosts'):
        # Copiamos los valores asegurando que sea una copia independiente para evitar referencias compartidas
        receiver_pkmn.boosts = deepcopy(target_pkmn.boosts)

    return battle


############################################################################################################################################################
# OK - Activación de efectos/movimientos especiales (ej: Skill Swap para cambiar habilidades)                                                           ###
############################################################################################################################################################
def activate(battle: Battle, split_msg: list) -> Battle:
    # Ejemplo de mensaje: |-activate|p2b: Starmie|Skill Swap|Drought|Natural Cure|[of] p1b: Torkoal
    if len(split_msg) < 6:
        return battle

    trigger_effect = split_msg[3].strip().lower()

    # 1. Validamos que el activador sea estrictamente "skill swap"
    if "skill swap" in trigger_effect:
        # --- DATOS DEL POKÉMON 1 (Emisor / p2b: Starmie) ---
        pkmn1_slot_full = split_msg[2].strip()  # Ej: "p2b: Starmie"
        pkmn1_opponent = is_opponent(battle, ["", "", pkmn1_slot_full])
        pkmn1_side = battle.opponent if pkmn1_opponent else battle.user
        pkmn1_is_active1 = pkmn1_slot_full.split(':')[0].strip().endswith('a')
        pkmn1 = pkmn1_side.active1 if pkmn1_is_active1 else pkmn1_side.active2

        # --- DATOS DEL POKÉMON 2 (Receptor / [of] p1b: Torkoal) ---
        # Extraemos el identificador de bando (p1b) limpiando el "[of]" de la cadena
        of_part = split_msg[6].partition("[of] ")[2].strip()  # Ej: "p1b: Torkoal"
        if not of_part:
            return battle
            
        pkmn2_slot_full = of_part
        pkmn2_opponent = is_opponent(battle, ["", "", pkmn2_slot_full])
        pkmn2_side = battle.opponent if pkmn2_opponent else battle.user
        pkmn2_is_active1 = pkmn2_slot_full.split(':')[0].strip().endswith('a')
        pkmn2 = pkmn2_side.active1 if pkmn2_is_active1 else pkmn2_side.active2

        # --- INTERCAMBIO DE HABILIDADES EN MEMORIA ---
        if pkmn1 and pkmn2:
            new_ability_pkmn1 = normalize_name(split_msg[4].strip())  # "Drought"  
            new_ability_pkmn2 = normalize_name(split_msg[5].strip())  # "Natural Cure"

            pkmn1.abilityPkmn = new_ability_pkmn1
            pkmn2.abilityPkmn = new_ability_pkmn2

            print(f"[SKILL SWAP] {pkmn1.name} ahora tiene {new_ability_pkmn1} | {pkmn2.name} ahora tiene {new_ability_pkmn2}")

    return battle


############################################################################################################################################################
### OK - Actualiza el turno y decrementa contadores de clima, terrenos, trick room y estados temporales                                                  ###
############################################################################################################################################################
def turn(battle: Battle, split_msg: list) -> Battle:
    # Ejemplo de mensaje: |turn|3
    if len(split_msg) < 3:
        return battle

    # Actualizamos el número de turno actual de la batalla
    new_turn = int(split_msg[2].strip())
    battle.turn = new_turn

    # Decremento de duraciones globales del campo de batalla
    if battle.weatherTurn > 0:
        battle.weatherTurn -= 1

    # Decremento de turnos para cada terreno activo en el diccionario
    for terr, turns in list(battle.terrain.items()):
        if turns > 1:
            battle.terrain[terr] -= 1

    # Decremento de condiciones de bando (Side Conditions) y limpieza de estados
    for side in (battle.user, battle.opponent):
        # Aseguramos que side_conditions sea un diccionario válido para evitar caídas
        if not isinstance(side.side_conditions, dict):
            # Si se convirtió en una lista u otro objeto, lo restablecemos a un diccionario limpio
            side.side_conditions = defaultdict(lambda: 0)

        for condition in ('tailwind', 'reflect', 'lightscreen', 'auroraveil'):
            if side.side_conditions.get(condition, 0) > 0:
                side.side_conditions[condition] -= 1

        # Limpieza o gestión de estados volátiles al finalizar el turno
        for active in (side.active1, side.active2):
            if active and hasattr(active, 'volatile_statuses') and isinstance(active.volatile_statuses, dict):
                if constants.DROWSY in active.volatile_statuses:
                    del active.volatile_statuses[constants.DROWSY]
    
    return battle


############################################################################################################################################################
### OK - Lookup para cada una de las funciones de actualización de entorno                                                                               ###
############################################################################################################################################################
BATTLE_MODIFIERS_LOOKUP = {
    'request': request,
    'switch': switch_or_drag,
    'faint': faint,
    'drag': switch_or_drag,
    'swap': swap,
    '-heal': heal_or_damage,
    '-damage': heal_or_damage,
    '-boost': boost,
    '-unboost': unboost,
    '-status': status,
    '-start': start_volatile_status,
    '-end': end_volatile_status,
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
    '-clearallboost': clearallboost,
    '-clearboost': clearboost,
    '-copyboost': copyboost,
    '-activate': activate,
    'turn': turn
}


############################################################################################################################################################
### OK - Actualiza el entorno de batalla procesando los mensajes de actualización del entorno y tipo |request|                                           ###
############################################################################################################################################################
async def update_battle_environment(battle: Battle, msg: str, msg_request: str) -> None:
    print("update_battle_environment")
    full_msg = f"{msg}\n{msg_request}" if msg_request else msg # Se concatena ambos mensajes recibidos para procesar todas las líneas en un solo flujo
    msg_lines = full_msg.split('\n')

    for line in msg_lines: # Para cada una de las líneas de la concatenación
        split_msg = line.split('|')
        if len(split_msg) < 2: # Solo consideramos los que tengan 3 elementos separados por |
            continue

        action = split_msg[1].strip() # Se obtiene la acción

        function_to_call = BATTLE_MODIFIERS_LOOKUP.get(action) # Para obtener su función correspondiente
        if function_to_call is not None:
            battle = function_to_call(battle, split_msg) # Y se ejecuta...
            battle.error = False # Cada vez que se actualiza el entorno, entonces la acción anteriormente ejecutada no ha caído en Error