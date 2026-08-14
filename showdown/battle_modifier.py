import json
import math
import constants

from copy import deepcopy
from collections import defaultdict
from teams.team_converter import normalize_name
from showdown.battle import Pokemon
from showdown.battle import Battle

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
    # |switch|POKEMON|DETAILS|HP STATUS  o  |drag|POKEMON|DETAILS|HP STATUS
    raw_name = split_msg[3].partition(",")[0].strip()
    if raw_name.lower() == 'mimikyu':
        raw_name = "mimikyu-disguised"
    name_pkmn = constants.standardName.get(raw_name.lower(), raw_name.lower())

    opponent_turn = is_opponent(battle, split_msg)
    side = battle.opponent if opponent_turn else battle.user

    slot_str = split_msg[2].split(':')[0].strip()
    is_active1 = slot_str.endswith('a')

    # Buscamos el Pokémon en la reserva
    for i, pkmn in enumerate(side.reserved):
        if pkmn is not None and getattr(pkmn, 'name', None) == name_pkmn:
            # Reseteamos boosts y estados volátiles del Pokémon que entra a luchar
            pkmn.volatile_statuses = []
            pkmn.boosts = defaultdict(lambda: 0)

            if opponent_turn:
                if not getattr(pkmn, 'opponent_confirmed', False):
                    side.opponentPkmnConfirmed += 1
                pkmn.opponent_confirmed = True

            # Intercambiamos referencias directamente en O(1)
            if is_active1:
                side.active1, side.reserved[i] = pkmn, side.active1
            else:
                side.active2, side.reserved[i] = pkmn, side.active2

            # Si es el primer turno / selección inicial (5 o 6 Pokémon), eliminamos el sobrante
            if len(side.reserved) in (5, 6):
                side.reserved.pop(i)
                
            break

    # Limpieza de reserva
    if opponent_turn:
        if sum(1 for p in side.reserved if p and getattr(p, 'opponent_confirmed', False)) == 2:
            side.reserved = [p for p in side.reserved if p and getattr(p, 'opponent_confirmed', False)]
    else:
        side.reserved = [p for p in side.reserved if p is not None]

    return battle

############################################################################################################################################################
### OK - Marcar Pokémon debilitado                                                                                                                       ###
############################################################################################################################################################
def faint(battle: Battle, split_msg: list) -> Battle:
    # |faint|p1b: Tornadus
    opponent_turn = is_opponent(battle, split_msg)
    side = battle.opponent if opponent_turn else battle.user

    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn:
        pkmn.HPB = 0

    return battle


############################################################################################################################################################
### OK - Actualización de HP por daño o curación                                                                                                         ###
############################################################################################################################################################
def heal_or_damage(battle: Battle, split_msg: list) -> Battle:
    # |-damage|p2a: Tornadus|91/100|[from] item: Life Orb
    opponent_turn = is_opponent(battle, split_msg)
    side = battle.opponent if opponent_turn else battle.user

    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if not pkmn:
        return battle

    hp_str = split_msg[3].strip()

    if constants.FNT in hp_str or hp_str.startswith("0 "):
        pkmn.HPB = 0
    else:
        parts = hp_str.split('/')
        current_hp_val = float(parts[0])

        if opponent_turn:
            # Para el oponente, Showdown reporta porcentajes ("91/100")
            hp_percentage = current_hp_val / 100.0
            pkmn.HPB = int(pkmn.max_hp * hp_percentage)
        else:
            # Para el usuario, Showdown reporta HP absoluto ("168/282")
            max_hp_val = float(parts[1].split()[0])
            pkmn.HPB = int(math.ceil(current_hp_val))
            pkmn.max_hp = int(math.ceil(max_hp_val))

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
    status_name = split_msg[3].strip()

    side = battle.opponent if is_opponent(battle, split_msg) else battle.user
    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn:
        pkmn.non_volatile_statuses = status_name

    return battle


############################################################################################################################################################
# OK - Inicia alterado volatil (ej: 'moveyawn')                                                                                                          ###
############################################################################################################################################################
def start_volatile_status(battle: Battle, split_msg: list) -> Battle:
    # Ejemplo de mensaje: |-start|p2a: Incineroar|move: Yawn|[of] p1a: Gastrodon
    
    # Validamos que el mensaje tenga la estructura esperada para extraer el estado volátil
    if len(split_msg) < 4:
        return battle

    # Extraemos y normalizamos el estado volátil (quitando prefijos como 'move:')
    raw_status = split_msg[3].split(":")[-1]
    volatile_status = normalize_name(raw_status)

    # Determinamos a qué bando y Pokémon activo pertenece el estado
    opponent_turn = is_opponent(battle, split_msg)
    side = battle.opponent if opponent_turn else battle.user

    slot_str = split_msg[2].split(':')[0].strip()
    pkmn = side.active1 if slot_str.endswith('a') else side.active2

    if pkmn and hasattr(pkmn, 'volatile_statuses'):
        # Al ser volatile_statuses un defaultdict, almacenamos/activamos el estado con valor 1 (o el contador que prefieras)
        pkmn.volatile_statuses[volatile_status] = 1

    return battle


############################################################################################################################################################
# OK - Fin alterado volatil (ej: 'moveyawn')                                                                                                             ###
############################################################################################################################################################
def end_volatile_status(battle: Battle, split_msg: list) -> Battle:
    # Ejemplo de mensaje: |-end|p1a: Aerodactyl|Encore
    
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
        # Si el estado volátil está registrado en el diccionario, lo eliminamos
        if volatile_status in pkmn.volatile_statuses:
            del pkmn.volatile_statuses[volatile_status]

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
    # |-weather|Sandstorm  o  |-weather|none
    raw_weather = split_msg[2].split(':')[-1].strip()
    weather_name = normalize_name(raw_weather)

    battle.weather = weather_name
    battle.weatherTurn = 0 if weather_name in ("none", "null") else 5
    return battle


############################################################################################################################################################
# OK - Inicio de Terreno, incluye Trick Room                                                                                                             ###
############################################################################################################################################################
def fieldstart(battle: Battle, split_msg: list) -> Battle:
    # |-fieldstart|move: Grassy Terrain
    raw_field = split_msg[2].split(':')[-1].strip()
    field_name = normalize_name(raw_field)

    if field_name == constants.TRICK_ROOM:
        battle.trick_room = 5
    else:
        battle.terrain = field_name
        battle.terrainTurn = 5

    return battle


############################################################################################################################################################
# OK - Fin de Terreno, incluye Trick Room                                                                                                                ###
############################################################################################################################################################
def fieldend(battle: Battle, split_msg: list) -> Battle:
    raw_field = split_msg[2].split(':')[-1].strip()
    field_name = normalize_name(raw_field)

    if field_name == constants.TRICK_ROOM:
        battle.trick_room = 0
    else:
        battle.terrain = None
        battle.terrainTurn = 0

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
    # Ejemplo de mensaje: |-item|p2a: Torkoal|Choice Scarf|[from] move: Trick
    
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
    # Ejemplo de mensaje: |-enditem|p1b: Kingambit|Focus Sash
    # Ejemplo de mensaje: |-enditem|p2a: Rillaboom|Grassy Seed|[from] item: Grassy Seed
    
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
    # Ejemplo de mensaje: |detailschange|p1a: Aerodactyl|Aerodactyl-Mega, L50, M
    # Ejemplo de mensaje: |detailschange|p1a: Mimikyu|Mimikyu-Busted, L50, F
    
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
        new_pkmn = Pokemon(name_pkmn, 0, opponent_turn, current_active.objectPkmn, current_active.abilityPkmn, current_active.move1, current_active.move2, current_active.move3, current_active.move4, getattr(current_active, 'nature', None))
        
        # Mantenemos los HP actuales proporcionales o heredados y los estados volátiles
        new_pkmn.HPB = current_active.HPB
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
    # Nota: El target puede estar en el bando contrario o incluso en el mismo (ej. Ally Transform)
    # Por lo general, Showdown indica el bando en split_msg[3]. Analizamos su slot:
    target_slot_full = split_msg[3].split(':')[0].strip() # Ej: "p2a" o "p1b"
    target_is_opponent = not target_slot_full.startswith(battle.user.id)
    target_side = battle.opponent if target_is_opponent else battle.user
    target_is_active1 = target_slot_full.endswith('a')
    target_pkmn = target_side.active1 if target_is_active1 else target_side.active2

    # 3. Si ambos existen, realizamos la transformación
    if source_pkmn and target_pkmn:
        # Guardamos el HP actual de Ditto antes de transformarse
        aux_hp = source_pkmn.HPB
        aux_max_hp = source_pkmn.max_hp

        # Creamos una copia exacta (deep copy) de las estadísticas y atributos del objetivo
        new_source = deepcopy(target_pkmn)
        
        # Mantenemos el HP actual y el HP máximo que ya tenía Ditto en combate
        new_source.HPB = aux_hp
        new_source.max_hp = aux_max_hp
        
        # Marcamos explícitamente que sigue siendo Ditto transformado
        new_source.isDitto = 1

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

    if pkmn and hasattr(pkmn, 'isMegaEvo'):
        # Establecemos la bandera de Mega Evolución en True
        pkmn.isMegaEvo = True

    return battle


############################################################################################################################################################
# OK -  Restablece las estadísticas (boosts) de TODOS los Pokémon activos en el campo (ej: por efecto de Haze / Niebla)                                  ###
############################################################################################################################################################
def clearallboost(battle: Battle, _=None) -> Battle:
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
### OK - Actualiza el turno y decrementa contadores de clima, terrenos, trick room y estados temporales                                                  ###
############################################################################################################################################################
def turn(battle: Battle, split_msg: list) -> Battle:
    # Ejemplo de mensaje: |turn|1
    
    # Validamos que el mensaje contenga el número de turno
    if len(split_msg) < 3:
        return battle

    # Actualizamos el número de turno actual de la batalla
    new_turn = int(split_msg[2].strip())
    battle.turn = new_turn

    # Decremento de duraciones globales del campo de batalla
    if battle.weatherTurn > 0:
        battle.weatherTurn -= 1
    if battle.terrainTurn > 0:
        battle.terrainTurn -= 1
    if battle.trick_room > 0:
        battle.trick_room -= 1

    # Decremento de condiciones de bando (Side Conditions) y limpieza de estados en ambos lados
    for side in (battle.user, battle.opponent):
        # Reducimos en 1 turno la duración de las pantallas,tailwind, etc.
        for condition in ('tailwind', 'reflect', 'lightscreen', 'auroraveil'):
            if side.side_conditions[condition] > 0:
                side.side_conditions[condition] -= 1

        # Limpieza o gestión de estados volátiles al finalizar el turno (ej: Drowsy por Yawn)
        for active in (side.active1, side.active2):
            if active and hasattr(active, 'volatile_statuses'):
                # Al ser volatile_statuses un defaultdict/diccionario, eliminamos la clave si está presente
                if constants.DROWSY in active.volatile_statuses:
                    del active.volatile_statuses[constants.DROWSY]
    
    return battle


############################################################################################################################################################
### OK - Estimamos las Estadísticas de Batalla Actuales de cada uno de los PKMN Oponentes Activos, en base a sus Estadísticas Base, Naturaleza y Boosts  ###
############################################################################################################################################################
def updateStatsOpponentPKMN(pkmn: Pokemon) -> None:
    # 1. El HP actual se estima directamente del valor HPB ya actualizado en batalla (por heal_or_damage)
    pkmn.HPA_battle_actual = max(0, pkmn.HPB)

    # 2. Inicializamos las demás estadísticas con sus valores base
    atk = pkmn.ATK_battle_base
    defe = pkmn.DEF_battle_base
    spa = pkmn.SPA_battle_base
    spd = pkmn.SPD_battle_base
    vel = pkmn.VEL_battle_base

    # 3. Aplicamos los efectos de la Naturaleza si existe y es conocida en el OTS
    if pkmn.nature:
        nature_lower = pkmn.nature.lower().strip()
        effects = constants.NATURE_EFFECTS.get(nature_lower, {})

        # Aplicamos el aumento (+10%)
        plus_stat = effects.get("plus")
        if plus_stat == "attack": atk = math.floor(atk * 1.1)
        elif plus_stat == "defense": defe = math.floor(defe * 1.1)
        elif plus_stat == "special-attack": spa = math.floor(spa * 1.1)
        elif plus_stat == "special-defense": spd = math.floor(spd * 1.1)
        elif plus_stat == "speed": vel = math.floor(vel * 1.1)

        # Aplicamos la disminución (-10%)
        minus_stat = effects.get("minus")
        if minus_stat == "attack": atk = math.floor(atk * 0.9)
        elif minus_stat == "defense": defe = math.floor(defe * 0.9)
        elif minus_stat == "special-attack": spa = math.floor(spa * 0.9)
        elif minus_stat == "special-defense": spd = math.floor(spd * 0.9)
        elif minus_stat == "speed": vel = math.floor(vel * 0.9)

    # 4. Obtenemos los multiplicadores de boosts directamente de forma ultra rápida
    atk_mult = constants.BOOST_MULTIPLIERS.get(pkmn.boosts.get("attack", 0), 1.0)
    def_mult = constants.BOOST_MULTIPLIERS.get(pkmn.boosts.get("defense", 0), 1.0)
    spa_mult = constants.BOOST_MULTIPLIERS.get(pkmn.boosts.get("special-attack", 0), 1.0)
    spd_mult = constants.BOOST_MULTIPLIERS.get(pkmn.boosts.get("special-defense", 0), 1.0)
    vel_mult = constants.BOOST_MULTIPLIERS.get(pkmn.boosts.get("speed", 0), 1.0)

    # 5. Asignamos los valores de batalla reales calculados y redondeados hacia abajo
    pkmn.ATK_battle_actual = math.floor(atk * atk_mult)
    pkmn.DEF_battle_actual = math.floor(defe * def_mult)
    pkmn.SPA_battle_actual = math.floor(spa * spa_mult)
    pkmn.SPD_battle_actual = math.floor(spd * spd_mult)
    pkmn.VEL_battle_actual = math.floor(vel * vel_mult)


############################################################################################################################################################
### OK - Estimamos las Estadísticas de Batalla Actuales de TODOS los PKMN Oponentes Activos                                                              ###
############################################################################################################################################################
async def updateStatsOpponents(battle: Battle) -> None:
    if battle.opponent.active1 is not None:
        updateStatsOpponentPKMN(battle.opponent.active1)
    if battle.opponent.active2 is not None:
        updateStatsOpponentPKMN(battle.opponent.active2)


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
    'replace': form_change,
    '-formechange': form_change,
    '-transform': transform,
    '-mega': mega,
    '-clearallboost': clearallboost,
    '-clearboost': clearboost,
    '-copyboost': copyboost,
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
        if len(split_msg) < 2: # Solo consideramos los que tengan 3 elementos separados por |
            continue

        action = split_msg[1].strip() # Se obtiene la acción

        function_to_call = BATTLE_MODIFIERS_LOOKUP.get(action) # Para obtener su función correspondiente
        if function_to_call is not None:
            battle = function_to_call(battle, split_msg) # Y se ejecuta...
            battle.error = False # Cada vez que se actualiza el entorno, entonces la acción anteriormente ejecutada no ha caído en Error