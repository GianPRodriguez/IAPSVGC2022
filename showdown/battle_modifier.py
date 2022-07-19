import re
import json
from copy import deepcopy
import logging
from collections import defaultdict

import constants
from data import all_move_json
from data import pokedex
from showdown.battle import Pokemon
from showdown.engine.helpers import normalize_name
from showdown.engine.helpers import get_pokemon_info_from_condition
from showdown.engine.helpers import calculate_stats
from showdown.engine.find_state_instructions import get_effective_speed
from showdown.engine.damage_calculator import calculate_damage
from showdown.engine.objects import boost_multiplier_lookup


logger = logging.getLogger(__name__)

# ¿El cambio es hecho por el opponente?
def is_opponent(battle,  split_msg):
    # split_msg = ["","switch","p1a: Incineroar","Incineroar, L50, M","190/202"]
    # split_msg[2] = "p1a: Incineroar"
    return not split_msg[2].startswith(battle.user.id)


# Obtiene el JSON y rqid... Además, avisa si es por force_switch o es solo un wait
def request(battle, split_msg):
    # split_msg = ["","request","{"active":[{"moves":[{"move":"Fake Out","id":"fakeout","pp":16,"maxpp":16,..."]
    if len(split_msg) >= 2:
        # Se divide el JSON en battle_json
        battle_json = json.loads(split_msg[2].strip('\''))
        # Obtiene el rqid, que al parecer no se utilizará
        battle.rqid = battle_json[constants.RQID]
        # ¿battle_json tiene "forceSwitch"? Se activa cuando uso U-TURN
        if battle_json.get(constants.FORCE_SWITCH):
            logger.debug("Action: forceSwitch")
            battle.force_switch = True
        else:
            battle.force_switch = False
        # ¿battle_json tiene "wait"? ¿Se activa cuando el opponente usa U-TURN?
        if battle_json.get(constants.WAIT):
            logger.debug("Action: wait")
            battle.wait = True
        else:
            battle.wait = False
        # Se almacena el JSON para elegir una accióm
        if not battle.wait:
            battle.request_json = battle_json
    return battle


# Avanza el tiempo...
def inactive(battle, split_msg):
    # split_msg = ["","inactive","Time left: 55 sec this turn "," 410 sec total"]
    regex_string = "(\d+) sec this turn"
    if split_msg[2].startswith(constants.TIME_LEFT):
        capture = re.search(regex_string, split_msg[2])
        time_left = int(capture.group(1))
        logger.debug("Tiempo restante: {} segundos...".format(time_left))
    return battle


# Ya no avanza el tiempo...
def inactiveoff(battle, _):
    logger.debug("Se ha detenido el tiempo...")
    return battle


# Ha habido un cambio de Pokémon...
def switch_or_drag(battle, split_msg):
    # |switch|POKEMON|DETAILS|HP STATUS or |drag|POKEMON|DETAILS|HP STATUS
    # |switch|p1a: Incineroar|Incineroar, L50, M|190/202
    # ¿El cambio es hecho por el opponente?
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                contador = 0
                for pkmn in battle.opponent.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        aux2 = battle.opponent.active1
                        battle.opponent.active1 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.opponent.active1.volatile_statuses = []
                        battle.opponent.active1.boosts = defaultdict(lambda: 0)
                        battle.opponent.active1.opponent_confirmed = True
                        # Fin - Limpieza de Stats
                        battle.opponent.active1.name = split_msg[3].partition(",")[0]
                        pkmn = deepcopy(aux2)
                        battle.opponent.active1.volatile_statuses = []
                        battle.opponent.active1.boosts = defaultdict(lambda: 0)
                        logger.debug("El oponente ha cambiado de Pokémon a " + str(battle.opponent.active1.name) + ": p2a")
                        # Si es el primer turno...
                        if len(battle.opponent.reserved) == 6 or len(battle.opponent.reserved) == 5: 
                            battle.opponent.reserved.pop(contador)
                    contador += 1
                # ¿Ya se conocen los 4 Pokémon del Oponente?
                contadorAux = 0
                for pkmn in battle.opponent.reserved:
                    if pkmn.opponent_confirmed == True:
                        contadorAux += 1
                if contadorAux == 2:
                    for pkmn in battle.opponent.reserved:
                        if pkmn.opponent_confirmed == False:
                            battle.opponent.reserved.remove(pkmn)
            else:
                contador = 0
                for pkmn in battle.opponent.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        aux2 = battle.opponent.active2
                        battle.opponent.active2 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.opponent.active2.volatile_statuses = []
                        battle.opponent.active2.boosts = defaultdict(lambda: 0)
                        battle.opponent.active1.opponent_confirmed = True
                        # Fin - Limpieza de Stats
                        battle.opponent.active2.name = split_msg[3].partition(",")[0]
                        pkmn = deepcopy(aux2)
                        battle.opponent.active2.volatile_statuses = []
                        battle.opponent.active2.boosts = defaultdict(lambda: 0)
                        logger.debug("El oponente ha cambiado de Pokémon a " + str(battle.opponent.active2.name) + ": p2b")
                        if len(battle.opponent.reserved) == 6 or len(battle.opponent.reserved) == 5: 
                            battle.opponent.reserved.pop(contador)
                    contador += 1
                # ¿Ya se conocen los 4 Pokémon del Oponente?
                contadorAux = 0
                for pkmn in battle.opponent.reserved:
                    if pkmn.opponent_confirmed == True:
                        contadorAux += 1
                if contadorAux == 2:
                    for pkmn in battle.opponent.reserved:
                        if pkmn.opponent_confirmed == False:
                            battle.opponent.reserved.remove(pkmn)
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                contador = 0
                for pkmn in battle.opponent.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        aux2 = battle.opponent.active1
                        battle.opponent.active1 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.opponent.active1.volatile_statuses = []
                        battle.opponent.active1.boosts = defaultdict(lambda: 0)
                        battle.opponent.active1.opponent_confirmed = True
                        # Fin - Limpieza de Stats
                        battle.opponent.active1.name = split_msg[3].partition(",")[0]
                        pkmn = deepcopy(aux2)
                        battle.opponent.active1.volatile_statuses = []
                        battle.opponent.active1.boosts = defaultdict(lambda: 0)
                        logger.debug("El oponente ha cambiado de Pokémon a " + str(battle.opponent.active1.name) + ": p1a")
                        if len(battle.opponent.reserved) == 6 or len(battle.opponent.reserved) == 5: 
                            battle.opponent.reserved.pop(contador)
                    contador += 1
                # ¿Ya se conocen los 4 Pokémon del Oponente?
                contadorAux = 0
                for pkmn in battle.opponent.reserved:
                    if pkmn.opponent_confirmed == True:
                        contadorAux += 1
                if contadorAux == 2:
                    for pkmn in battle.opponent.reserved:
                        if pkmn.opponent_confirmed == False:
                            battle.opponent.reserved.remove(pkmn)
            else:
                contador = 0
                for pkmn in battle.opponent.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        aux2 = battle.opponent.active2
                        battle.opponent.active2 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.opponent.active2.volatile_statuses = []
                        battle.opponent.active2.boosts = defaultdict(lambda: 0)
                        battle.opponent.active1.opponent_confirmed = True
                        # Fin - Limpieza de Stats
                        battle.opponent.active2.name = split_msg[3].partition(",")[0]
                        pkmn = deepcopy(aux2)
                        battle.opponent.active2.volatile_statuses = []
                        battle.opponent.active2.boosts = defaultdict(lambda: 0)
                        logger.debug("El oponente ha cambiado de Pokémon a " + str(battle.opponent.active2.name) + ": p1b")
                        if len(battle.opponent.reserved) == 6 or len(battle.opponent.reserved) == 5: 
                            battle.opponent.reserved.pop(contador)
                    contador += 1
                # ¿Ya se conocen los 4 Pokémon del Oponente?
                contadorAux = 0
                for pkmn in battle.opponent.reserved:
                    if pkmn.opponent_confirmed == True:
                        contadorAux += 1
                if contadorAux == 2:
                    for pkmn in battle.opponent.reserved:
                        if pkmn.opponent_confirmed == False:
                            battle.opponent.reserved.remove(pkmn)          
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                contador = 0
                for pkmn in battle.user.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        aux = battle.user.active1
                        battle.user.active1 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.user.active1.volatile_statuses = []
                        battle.user.active1.boosts = defaultdict(lambda: 0)
                        # Fin - Limpieza de Stats
                        battle.user.active1.name = split_msg[3].partition(",")[0]
                        pkmn = deepcopy(aux)
                        battle.user.active1.volatile_statuses = []
                        battle.user.active1.boosts = defaultdict(lambda: 0)
                        logger.debug("El usuario ha cambiado de Pokémon a " + str(battle.user.active1.name) + ": p1a")
                        # Intercambio en Reservados
                        battle.user.reserved[contador] = deepcopy(pkmn)
                    contador+=1
            else:
                contador = 0
                for pkmn in battle.user.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        aux = battle.user.active2
                        battle.user.active2 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.user.active2.volatile_statuses = []
                        battle.user.active2.boosts = defaultdict(lambda: 0)
                        # Fin - Limpieza de Stats
                        battle.user.active2.name = split_msg[3].partition(",")[0]
                        pkmn = deepcopy(aux)
                        battle.user.active2.volatile_statuses = []
                        battle.user.active2.boosts = defaultdict(lambda: 0)
                        logger.debug("El usuario ha cambiado de Pokémon a " + str(battle.user.active2.name) + ": p1b")
                        # Intercambio en Reservados
                        battle.user.reserved[contador] = deepcopy(pkmn)
                    contador+=1   
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                contador = 0
                for pkmn in battle.user.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        aux = battle.user.active1
                        battle.user.active1 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.user.active1.volatile_statuses = []
                        battle.user.active1.boosts = defaultdict(lambda: 0)
                        # Fin - Limpieza de Stats
                        battle.user.active1.name = split_msg[3].partition(",")[0]
                        pkmn = deepcopy(aux)
                        battle.user.active1.volatile_statuses = []
                        battle.user.active1.boosts = defaultdict(lambda: 0)
                        logger.debug("El usuario ha cambiado de Pokémon a " + str(battle.user.active1.name) + ": p2a")
                        # Intercambio en Reservados
                        battle.user.reserved[contador] = deepcopy(pkmn)
                    contador+=1
            else:
                contador = 0
                for pkmn in battle.user.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        aux = battle.user.active2
                        battle.user.active2 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.user.active2.volatile_statuses = []
                        battle.user.active2.boosts = defaultdict(lambda: 0)
                        # Fin - Limpieza de Stats
                        battle.user.active2.name = split_msg[3].partition(",")[0]
                        pkmn = deepcopy(aux)
                        battle.user.active2.volatile_statuses = []
                        battle.user.active2.boosts = defaultdict(lambda: 0)
                        logger.debug("El usuario ha cambiado de Pokémon a " + str(battle.user.active2.name) + ": p2b") 
                        # Intercambio en Reservados
                        battle.user.reserved[contador] = deepcopy(pkmn)
                    contador+=1
    return battle


# Aumento o disminución de HP...
def heal_or_damage(battle, split_msg):
    # split_msg = ["","-damage","p1a: Bibarel","168/282","[from] item: Life Orb"]
    # split_msg = ["","-damage","p2a: Scizor","49/100"]
    # split_msg = ["","-heal","p1a: Aerodactyl","221/265"]
    # split_msg = ["","-damage","p2a: Indeedee","0 fnt"]
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                if constants.FNT in split_msg[3]:
                    # Si el Pokémon se ha debilitado debido a un movimiento
                    battle.opponent.active1.hp = 0
                else:
                    # Si el Pokémon aún tiene HP, se obtiene el nuevo %
                    new_hp_percentage = float(split_msg[3].split('/')[0]) / 100
                    # Se establece el nuevo %
                    battle.opponent.active1.hp = battle.opponent.active1.max_hp * new_hp_percentage
            else:
                if constants.FNT in split_msg[3]:
                        # Si el Pokémon se ha debilitado debido a un movimiento
                        battle.opponent.active2.hp = 0
                    else:
                        # Si el Pokémon aún tiene HP, se obtiene el nuevo %
                        new_hp_percentage = float(split_msg[3].split('/')[0]) / 100
                        # Se establece el nuevo %
                        battle.opponent.active2.hp = battle.opponent.active2.max_hp * new_hp_percentage
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if constants.FNT in split_msg[3]:
                    # Si el Pokémon se ha debilitado debido a un movimiento
                    battle.opponent.active1.hp = 0
                else:
                    # Si el Pokémon aún tiene HP, se obtiene el nuevo %
                    new_hp_percentage = float(split_msg[3].split('/')[0]) / 100
                    # Se establece el nuevo %
                    battle.opponent.active1.hp = battle.opponent.active1.max_hp * new_hp_percentage
            else:
                    if constants.FNT in split_msg[3]:
                        # Si el Pokémon se ha debilitado debido a un movimiento
                        battle.opponent.active2.hp = 0
                    else:
                        # Si el Pokémon aún tiene HP, se obtiene el nuevo %
                        new_hp_percentage = float(split_msg[3].split('/')[0]) / 100
                        # Se establece el nuevo %
                        battle.opponent.active2.hp = battle.opponent.active2.max_hp * new_hp_percentage
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if constants.FNT in split_msg[3]:
                    battle.user.active1.hp = 0
                else:
                    battle.user.active1.hp = float(split_msg[3].split('/')[0])
                    battle.user.active1.max_hp = float(split_msg[3].split('/')[1].split()[0])
            else:
                if constants.FNT in split_msg[3]:
                    battle.user.active2.hp = 0
                else:
                    battle.user.active2.hp = float(split_msg[3].split('/')[0])
                    battle.user.active2.max_hp = float(split_msg[3].split('/')[1].split()[0])
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                if constants.FNT in split_msg[3]:
                    battle.user.active1.hp = 0
                else:
                    battle.user.active1.hp = float(split_msg[3].split('/')[0])
                    battle.user.active1.max_hp = float(split_msg[3].split('/')[1].split()[0])
            else:
                if constants.FNT in split_msg[3]:
                    battle.user.active2.hp = 0
                else:
                    battle.user.active2.hp = float(split_msg[3].split('/')[0])
                    battle.user.active2.max_hp = float(split_msg[3].split('/')[1].split()[0])
    return battle


# Pokémon debilitado...
def faint(battle, split_msg):
    # split_msg = ["","faint","p1b: Tornadus"]
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                battle.opponent.active1.hp = 0
            else:
                battle.opponent.active2.hp = 0
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                battle.opponent.active1.hp = 0
            else:
                battle.opponent.active2.hp = 0
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                battle.user.active1.hp = 0
            else:
                battle.user.active2.hp = 0
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                battle.user.active1.hp = 0
            else:
                battle.user.active2.hp = 0
    return battle

# Se realiza un movimiento...
def move(battle, split_msg):
    # split_msg = ["","move","p2a: Scizor","Bug Bite","p1a: Vileplume"]
    # split_msg = ["","move","p1b: Zacian","Behemoth Blade","p2b: Zekrom"]
    # No importa para el agente los movimientos registrados
    return battle


# Aumento de estadísticas
def boost(battle, split_msg):
    # split_msg = ["","-boost","p2b: Celesteela","spe","1"]
    # Se obtiene el nombre de la estadística que ha incrementado... stat = "SPEED"
    stat = constants.STAT_ABBREVIATION_LOOKUPS[split_msg[3].strip()]
    # Se obtiene la cantidad en que ha aumentado... amount = 1
    amount = int(split_msg[4].strip())
    # Se define en qué Pokémon se realiza el incremento en la estadística...
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                battle.opponent.active1.boosts[stat] = min(battle.opponent.active1.boosts[stat] + amount, constants.MAX_BOOSTS)
            else:
                battle.opponent.active2.boosts[stat] = min(battle.opponent.active2.boosts[stat] + amount, constants.MAX_BOOSTS)
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                battle.opponent.active1.boosts[stat] = min(battle.opponent.active1.boosts[stat] + amount, constants.MAX_BOOSTS)
            else:
                battle.opponent.active2.boosts[stat] = min(battle.opponent.active2.boosts[stat] + amount, constants.MAX_BOOSTS)
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                battle.user.active1.boosts[stat] = min(battle.user.active1.boosts[stat] + amount, constants.MAX_BOOSTS)
            else:
                battle.user.active2.boosts[stat] = min(battle.user.active2.boosts[stat] + amount, constants.MAX_BOOSTS)
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                battle.user.active1.boosts[stat] = min(battle.user.active1.boosts[stat] + amount, constants.MAX_BOOSTS)
            else:
                battle.user.active2.boosts[stat] = min(battle.user.active2.boosts[stat] + amount, constants.MAX_BOOSTS)
    return battle


# Disminución de estadísticas
def unboost(battle, split_msg):
    # split_msg = ["","-unboost","p2a: Whimsicott","spe","1"]
    # Se obtiene el nombre de la estadística que ha disminuido... stat = "SPEED"
    stat = constants.STAT_ABBREVIATION_LOOKUPS[split_msg[3].strip()]
    # Se obtiene la cantidad en que ha aumentado... amount = 1
    amount = int(split_msg[4].strip())
    # Se define en qué Pokémon se realiza el decremento en la estadística...
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                battle.opponent.active1.boosts[stat] = max(battle.opponent.active1.boosts[stat] - amount, -1*constants.MAX_BOOSTS)
            else:
                battle.opponent.active2.boosts[stat] = max(battle.opponent.active2.boosts[stat] - amount, -1*constants.MAX_BOOSTS)
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                battle.opponent.active1.boosts[stat] = max(battle.opponent.active1.boosts[stat] - amount, -1*constants.MAX_BOOSTS)
            else:
                battle.opponent.active2.boosts[stat] = max(battle.opponent.active2.boosts[stat] - amount, -1*constants.MAX_BOOSTS)
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                battle.user.active1.boosts[stat] = max(battle.user.active1.boosts[stat] - amount, -1*constants.MAX_BOOSTS)
            else:
                battle.user.active2.boosts[stat] = max(battle.user.active2.boosts[stat] - amount, -1*constants.MAX_BOOSTS)
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                battle.user.active1.boosts[stat] = max(battle.user.active1.boosts[stat] - amount, -1*constants.MAX_BOOSTS)
            else:
                battle.user.active2.boosts[stat] = max(battle.user.active2.boosts[stat] - amount, -1*constants.MAX_BOOSTS)
    return battle


# Un Pokémon ha cambiado de status "permanente" + slp
def status(battle, split_msg):
    # split_msg = ["","-status","p2a: Slowbro","brn"]
    status_name = split_msg[3].strip()
    # status_name = "brn"
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                battle.opponent.active1.status = status_name
            else:
                battle.opponent.active2.status = status_name
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                battle.opponent.active1.status = status_name
            else:
                battle.opponent.active2.status = status_name
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                battle.user.active1.status = status_name
            else:
                battle.user.active2.status = status_name
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                battle.user.active1.status = status_name
            else:
                battle.user.active2.status = status_name
    return battle


# Se activa una habilidad...
def activate(battle, split_msg):
    # split_msg = ["","-activate","p2b: Indeedee","move: Psychic Terrain"]
    return battle


# Se prepara para un movimiento...
def prepare(battle, split_msg):
    return battle


# Estado volátil... Solo considero Dynamax...
def start_volatile_status(battle, split_msg):
    # split_msg = ["","-start","p2a: Kyogre","Dynamax"]
    # split_msg = ["","-start","p2a: Incineroar","move: Yawn","[of] p1a: Gastrodon"]
    # split_msg = ["","-start","p1a: Charizard","Dynamax","Gmax"]
    volatile_status = normalize_name(split_msg[3].split(":")[-1])
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                if volatile_status == constants.DYNAMAX:
                    battle.opponent.active1.hp *= 2
                    battle.opponent.active1.max_hp *= 2
                    logger.debug("{} ha sido afectado por el fenómeno dinamax - se duplica su HP a {}/{}".format(battle.opponent.active1.name, battle.opponent.active1.hp, battle.opponent.active1.max_hp))
            else:
                if volatile_status == constants.DYNAMAX:
                    battle.opponent.active2.hp *= 2
                    battle.opponent.active2.max_hp *= 2
                    logger.debug("{} ha sido afectado por el fenómeno dinamax - se duplica su HP a {}/{}".format(battle.opponent.active2.name, battle.opponent.active2.hp, battle.opponent.active2.max_hp))
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if volatile_status == constants.DYNAMAX:
                    battle.opponent.active1.hp *= 2
                    battle.opponent.active1.max_hp *= 2
                    logger.debug("{} ha sido afectado por el fenómeno dinamax - se duplica su HP a {}/{}".format(battle.opponent.active1.name, battle.opponent.active1.hp, battle.opponent.active1.max_hp))
            else:
                if volatile_status == constants.DYNAMAX:
                    battle.opponent.active2.hp *= 2
                    battle.opponent.active2.max_hp *= 2
                    logger.debug("{} ha sido afectado por el fenómeno dinamax - se duplica su HP a {}/{}".format(battle.opponent.active2.name, battle.opponent.active2.hp, battle.opponent.active2.max_hp))
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if volatile_status == constants.DYNAMAX:
                    battle.user.active1.hp *= 2
                    battle.user.active1.max_hp *= 2
                    logger.debug("{} ha sido afectado por el fenómeno dinamax - se duplica su HP a {}/{}".format(battle.user.active1.name, battle.user.active1.hp, battle.user.active1.max_hp))
            else:
                if volatile_status == constants.DYNAMAX:
                    battle.user.active2.hp *= 2
                    battle.user.active2.max_hp *= 2
                    logger.debug("{} ha sido afectado por el fenómeno dinamax - se duplica su HP a {}/{}".format(battle.user.active2.name, battle.user.active2.hp, battle.user.active2.max_hp))
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                if volatile_status == constants.DYNAMAX:
                    battle.user.active1.hp *= 2
                    battle.user.active1.max_hp *= 2
                    logger.debug("{} ha sido afectado por el fenómeno dinamax - se duplica su HP a {}/{}".format(battle.user.active1.name, battle.user.active1.hp, battle.user.active1.max_hp))
            else:
            if volatile_status == constants.DYNAMAX:
                    battle.user.active2.hp *= 2
                    battle.user.active2.max_hp *= 2
                    logger.debug("{} ha sido afectado por el fenómeno dinamax - se duplica su HP a {}/{}".format(battle.user.active2.name, battle.user.active2.hp, battle.user.active2.max_hp))
    return battle


# Fin estado volátil...
def end_volatile_status(battle, split_msg):
    # split_msg = ["","-end","p2a: Kyogre","Dynamax"]
    volatile_status = normalize_name(split_msg[3].split(":")[-1])
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                if volatile_status == constants.DYNAMAX:
                    battle.opponent.active1.hp /= 2
                    battle.opponent.active1.max_hp /= 2
                    logger.debug("{} ha terminado el fenómeno dinamax - se reduce a la mitad su HP a {}/{}".format(battle.opponent.active1.name, battle.opponent.active1.hp, battle.opponent.active1.max_hp))
            else:
                if volatile_status == constants.DYNAMAX:
                    battle.opponent.active2.hp /= 2
                    battle.opponent.active2.max_hp /= 2
                    logger.debug("{} ha terminado el fenómeno dinamax - se reduce a la mitad su HP a {}/{}".format(battle.opponent.active2.name, battle.opponent.active2.hp, battle.opponent.active2.max_hp))
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if volatile_status == constants.DYNAMAX:
                    battle.opponent.active1.hp /= 2
                    battle.opponent.active1.max_hp /= 2
                    logger.debug("{} ha terminado el fenómeno dinamax - se reduce a la mitad su HP a {}/{}".format(battle.opponent.active1.name, battle.opponent.active1.hp, battle.opponent.active1.max_hp))
            else:
                if volatile_status == constants.DYNAMAX:
                    battle.opponent.active2.hp /= 2
                    battle.opponent.active2.max_hp /= 2
                    logger.debug("{} ha terminado el fenómeno dinamax - se reduce a la mitad su HP a {}/{}".format(battle.opponent.active2.name, battle.opponent.active2.hp, battle.opponent.active2.max_hp))
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if volatile_status == constants.DYNAMAX:
                    battle.user.active1.hp /= 2
                    battle.user.active1.max_hp /= 2
                    logger.debug("{} ha terminado el fenómeno dinamax - se reduce a la mitad su HP a {}/{}".format(battle.user.active1.name, battle.user.active1.hp, battle.user.active1.max_hp))
            else:
                if volatile_status == constants.DYNAMAX:
                    battle.user.active2.hp /= 2
                    battle.user.active2.max_hp /= 2
                    logger.debug("{} ha terminado el fenómeno dinamax - se reduce a la mitad su HP a {}/{}".format(battle.user.active2.name, battle.user.active2.hp, battle.user.active2.max_hp))
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                if volatile_status == constants.DYNAMAX:
                    battle.user.active1.hp /= 2
                    battle.user.active1.max_hp /= 2
                    logger.debug("{} ha terminado el fenómeno dinamax - se reduce a la mitad su HP a {}/{}".format(battle.user.active1.name, battle.user.active1.hp, battle.user.active1.max_hp))
            else:
            if volatile_status == constants.DYNAMAX:
                    battle.user.active2.hp /= 2
                    battle.user.active2.max_hp /= 2
                    logger.debug("{} ha terminado el fenómeno dinamax - se reduce a la mitad su HP a {}/{}".format(battle.user.active2.name, battle.user.active2.hp, battle.user.active2.max_hp))
    return battle


# Un estado se cura...
def curestatus(battle, split_msg):
    # split_msg = ["","-curestatus","p2a: Kyogre","slp","[msg]"]
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                battle.opponent.active1.status = None
                logger.debug("{} se ha curado...".format(battle.opponent.active1.name))
            else:
                battle.opponent.active2.status = None
                logger.debug("{} se ha curado...".format(battle.opponent.active2.name))
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                battle.opponent.active1.status = None
                logger.debug("{} se ha curado...".format(battle.opponent.active1.name))
            else:
                battle.opponent.active2.status = None
                logger.debug("{} se ha curado...".format(battle.opponent.active2.name))
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                battle.user.active1.status = None
                logger.debug("{} se ha curado...".format(battle.user.active1.name))
            else:
                battle.user.active2.status = None
                logger.debug("{} se ha curado...".format(battle.user.active2.name))
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                battle.user.active1.status = None
                logger.debug("{} se ha curado...".format(battle.user.active1.name))
            else:
                battle.user.active2.status = None
                logger.debug("{} se ha curado...".format(battle.user.active2.name))
    return battle


# The Pokémon POKEMON has used a move that cures its team of status effects, like Heal Bell.
def cureteam(battle, split_msg):
    # No se usa...
    return battle


# Se establece un Clima... no me importa quién lo realiza
# split_msg = "|-weather|Sandstorm"
# split_msg = "|-weather|none"
def weather(battle, split_msg):
    # weather_name = "Sandstorm" || weather_name = "none"
    weather_name = normalize_name(split_msg[2].split(':')[-1].strip())
    # Mensaje en el CMD
    # logger.debug("El clima {} ha empezado...".format(weather_name))
    # Se establece en class Battle:
    battle.weather = weather_name
    if weather_name == "none":
        self.weatherTurn = 0
    return battle


# Inicia un Field...
def fieldstart(battle, split_msg):
    # split_msg = ["","-fieldstart","move: Grassy Terrain","[from] ability: Grassy Surge","[of] p2a: Rillaboom"]
    # split_msg = ["","-fieldstart","move: Electric Terrain"]
    field_name = normalize_name(split_msg[2].split(':')[-1].strip())
    # ¿Qué Field es...?
    if field_name == constants.TRICK_ROOM:
        # Trick Room
        logger.debug("Ha iniciado Trick Room...")
        battle.trick_room = True
        self.trick_roomTurn = 5
    else:
        # Otro Field
        logger.debug("Ha iniciado el campo {}...".format(field_name))
        battle.field = field_name
        self.fieldTurn = 5
    return battle


# Termina un Field
def fieldend(battle, split_msg):
    field_name = normalize_name(split_msg[2].split(':')[-1].strip())
    # ¿Qué Field es...?
    if field_name == constants.TRICK_ROOM:
        # Trick Room
        logger.debug("Ha terminado Trick Room...")
        battle.trick_room = False
        self.trick_roomTurn = 0
    else:
        # Otro Field
        logger.debug("Ha terminado el campo {}...".format(field_name))
        battle.field = None
        self.fieldTurn = 0
    return battle


# Side conditions are all effects that affect one side of the field. (For example: Tailwind, Stealth Rock, Reflect)
def sidestart(battle, split_msg):
    # split_msg = ["","-sidestart","p1: PopplioPeru98","move: Tailwind"]
    condition = split_msg[3].split(':')[-1].strip()
    condition = normalize_name(condition)
    # condition = "Tailwind"
    if is_opponent(battle, split_msg):
        logger.debug("Efecto {} ha sido iniciado por el opponente...".format(condition))
        battle.opponent.side_conditions[condition] += 5
    else:
        logger.debug("Efecto {} ha sido iniciado por el usuario...".format(condition))
        battle.user.side_conditions[condition] += 5
    return battle


# Indicates that the side condition CONDITION ended for the given SIDE.
def sideend(battle, split_msg):
    # |-sideend|p2: PopplioPeru98|move: Tailwind
    condition = split_msg[3].split(':')[-1].strip()
    condition = normalize_name(condition)
    # condition = "Tailwind"
    if is_opponent(battle, split_msg):
        logger.debug("Efecto {} del opponente ha terminado...".format(condition))
        battle.opponent.side_conditions[condition] = 0
    else:
        logger.debug("Efecto {} del usuario ha terminado...".format(condition))
        battle.user.side_conditions[condition] = 0
    return battle


# Intercambia efectos
def swapsideconditions(battle, _):
    # No se usa...
    return battle


# Establece Objeto
def set_item(battle, split_msg):
    # No se considera...
    return battle


# Remueve Objeto...
def remove_item(battle, split_msg):
    # No se considera...
    return battle


# Establece habilidad
def set_ability(battle, split_msg):
    # No se considera...
    return battle


# Establece habilidad
def set_opponent_ability_from_ability_tag(battle, split_msg):
    # No se considera...
    return battle


#
def form_change(battle, split_msg):
    # split_msg = ["","detailschange","p2a: Metagross","Metagross-Mega, L76"]
    if is_opponent(battle, split_msg):
        side = battle.opponent
    else:
        side = battle.user

    base_name = side.active.base_name
    hp_percent = float(side.active.hp) / side.active.max_hp
    previous_moves = side.active.moves
    previous_boosts = side.active.boosts
    previous_status = side.active.status
    previous_item = side.active.item

    new_pokemon = Pokemon.from_switch_string(split_msg[3])
    new_pokemon.moves = previous_moves
    if new_pokemon in side.reserve:
        side.reserve.remove(new_pokemon)

    side.active = new_pokemon
    side.active.hp = hp_percent * side.active.max_hp
    side.active.boosts = previous_boosts
    side.active.status = previous_status
    side.active.item = previous_item

    if side.active.name != "zoroark":
        side.active.base_name = base_name
    return battle


# Ataque Z
def zpower(battle, split_msg):
    # No VGC 2022
    return battle


# Reestablece estadísticas negativas
def clearnegativeboost(battle, split_msg):
    # No VGC 2022
    return battle


# Reestablece todas las estadísticas
def clearallboost(battle, _):
    # Usuario
    for stat, value in battle.user.active1.boosts.items():
        if value != 0:
            logger.debug("Configurando estadísticas de {} de {} a 0".format(battle.user.active1.name, stat))
            battle.user.active1.boosts[stat] = 0
    # Usuario
    for stat, value in battle.user.active2.boosts.items():
        if value != 0:
            logger.debug("Configurando estadísticas de {} de {} a 0".format(battle.user.active2.name, stat))
            battle.user.active2.boosts[stat] = 0
    # opponente
    for stat, value in battle.opponent.active1.boosts.items():
        if value != 0:
            logger.debug("Configurando estadísticas de {} de {} a 0".format(battle.opponent.active1.name, stat))
            battle.opponent.active1.boosts[stat] = 0
    # opponente
    for stat, value in battle.opponent.active2.boosts.items():
        if value != 0:
            logger.debug("Configurando estadísticas de {} de {} a 0".format(battle.opponent.active2.name, stat))
            battle.opponent.active2.boosts[stat] = 0
    return battle


# The Pokémon POKEMON used move MOVE which causes a temporary effect lasting the duration of the turn. (For example: Protect, Focus Punch, Roost).
def singleturn(battle, split_msg):
    # No se considera movimientos
    return battle


# Signals the upkeep phase of the turn where the number of turns left for field conditions are updated.
def upkeep(battle, _):
    # No se considera
    return battle


# Mega-evolución
def mega(battle, split_msg):
    # No VGC 2022
    return battle


# Ditto
def transform(battle, split_msg):
    if is_opponent(battle, split_msg):
        transformed_into_name = battle.user.active.name

        battle_copy = deepcopy(battle)
        battle.opponent.active.boosts = deepcopy(battle.user.active.boosts)

        battle_copy.user.from_json(battle_copy.request_json)

        if battle_copy.user.active.name == transformed_into_name or battle_copy.user.active.name.startswith(transformed_into_name):
            transformed_into = battle_copy.user.active
        else:
            transformed_into = find_pokemon_in_reserves(transformed_into_name, battle_copy.user.reserve)

        logger.debug("Opponent {} transformed into {}".format(battle.opponent.active.name, battle.user.active.name))
        battle.opponent.active.stats = deepcopy(transformed_into.stats)
        battle.opponent.active.ability = deepcopy(transformed_into.ability)
        battle.opponent.active.moves = deepcopy(transformed_into.moves)
        battle.opponent.active.types = deepcopy(transformed_into.types)

        if constants.TRANSFORM not in battle.opponent.active.volatile_statuses:
            battle.opponent.active.volatile_statuses.append(constants.TRANSFORM)
    return battle


# Actualiza el turno
def turn(battle, split_msg):
    # split_msg = ["","turn","1"]
    battle.turn = int(split_msg[2])
    if self.weatherTurn > 0:
        self.weatherTurn -= 1
    if self.fieldTurn > 0:
        self.fieldTurn -= 1
    if self.trick_roomTurn > 0:
        self.trick_roomTurn -= 1
    return battle


# ¿?
def noinit(battle, split_msg):
    if split_msg[2] == "rename":
        battle.battle_tag = split_msg[3]
        logger.debug("Renamed battle to {}".format(battle.battle_tag))
    return battle


# Actualiza estado
async def update_battle(battle, msg):
    msg_lines = msg.split('\n')

    action = None
    for i, line in enumerate(msg_lines):
        split_msg = line.split('|')
        if len(split_msg) < 2:
            continue

        action = split_msg[1].strip()

        battle_modifiers_lookup = {
            'request': request,
            'switch': switch_or_drag,
            'faint': faint,
            'drag': switch_or_drag,
            '-heal': heal_or_damage,
            '-damage': heal_or_damage,
            'move': move,
            '-boost': boost,
            '-unboost': unboost,
            '-status': status,
            '-activate': activate,
            '-prepare': prepare,
            '-start': start_volatile_status,
            '-end': end_volatile_status,
            '-curestatus': curestatus,
            '-cureteam': cureteam,
            '-weather': weather,
            '-fieldstart': fieldstart,
            '-fieldend': fieldend,
            '-sidestart': sidestart,
            '-sideend': sideend,
            '-swapsideconditions': swapsideconditions,
            '-item': set_item,
            '-enditem': remove_item,
            '-immune': set_ability,
            '-ability': set_opponent_ability_from_ability_tag,
            'detailschange': form_change,
            'replace': form_change,
            '-formechange': form_change,
            '-transform': transform,
            '-mega': mega,
            '-zpower': zpower,
            '-clearnegativeboost': clearnegativeboost,
            '-clearallboost': clearallboost,
            '-singleturn': singleturn,
            'upkeep': upkeep,
            'inactive': inactive,
            'inactiveoff': inactiveoff,
            'turn': turn,
            'noinit': noinit,
        }

        function_to_call = battle_modifiers_lookup.get(action)
        if function_to_call is not None:
            battle = function_to_call(battle, split_msg)
        if action == 'turn':
            return True, battle

    if action in ['inactive', 'updatesearch']:
        return False, battle

    if action != "request":
        return battle.force_switch, battle
    return None, battle