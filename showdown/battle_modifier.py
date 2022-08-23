import re
import json
import math
import logging
from copy import deepcopy
from collections import defaultdict
from teams.team_converter import normalize_name
import constants
from showdown.battle import Pokemon
from showdown.battle_bots.helpers import getATBoost
from showdown.battle_bots.helpers import getDEBoost

logger = logging.getLogger(__name__)

# ¿La modificación es para el opponente?
def is_opponent(battle,  split_msg):
    # split_msg = ["","switch","p1a: Incineroar","Incineroar, L50, M","190/202"]
    # split_msg[2] = "p1a: Incineroar"
    return not split_msg[2].startswith(battle.user.id)


# Obtiene el JSON y rqid... Además, avisa si es por force_switch o es solo un wait
def request(battle, split_msg):
    # split_msg = ["","request","{"active":[{"moves":[{"move":"Fake Out","id":"fakeout","pp":16,"maxpp":16,..."]
    if len(split_msg) >= 3:
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
        # logger.info("Tiempo restante: {} segundos...".format(time_left))
    return battle


# Ya no avanza el tiempo...
def inactiveoff(battle, _):
    #logger.info("Se ha detenido el tiempo...")
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
                        pkmnAux = deepcopy(battle.opponent.active1)
                        battle.opponent.active1 = deepcopy(pkmn)
                        # Inicio - Actualizacion de Stats
                        battle.opponent.active1.name = split_msg[3].partition(",")[0]
                        battle.opponent.active1.volatile_statuses = []
                        battle.opponent.active1.boosts = defaultdict(lambda: 0)
                        if battle.opponent.active1.opponent_confirmed != True:
                            battle.opponent.opponentPkmnConfirmed += 1
                        battle.opponent.active1.opponent_confirmed = True
                        # Fin - Actualizacion de Stats
                        # logger.debug("El oponente ha cambiado de Pokémon a " + str(battle.opponent.active1.name) + ": p2a")
                        # Intercambio en Reservados
                        battle.opponent.reserved[contador] = deepcopy(pkmnAux)
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
                    for pkmn in battle.opponent.reserved:
                        if pkmn.opponent_confirmed == False:
                            battle.opponent.reserved.remove(pkmn)
                    if len(battle.opponent.reserved) == 3:
                        logger.error("NO DEBE HABER 3")
            else:
                contador = 0
                for pkmn in battle.opponent.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        pkmnAux = deepcopy(battle.opponent.active2)
                        battle.opponent.active2 = deepcopy(pkmn)
                        # Inicio - Actualizacion de Stats
                        battle.opponent.active2.name = split_msg[3].partition(",")[0]
                        battle.opponent.active2.volatile_statuses = []
                        battle.opponent.active2.boosts = defaultdict(lambda: 0)
                        if battle.opponent.active2.opponent_confirmed != True:
                            battle.opponent.opponentPkmnConfirmed += 1
                        battle.opponent.active2.opponent_confirmed = True
                        # Fin - Actualizacion de Stats
                        # logger.debug("El oponente ha cambiado de Pokémon a " + str(battle.opponent.active2.name) + ": p2b")
                        # Intercambio en Reservados
                        battle.opponent.reserved[contador] = deepcopy(pkmnAux)
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
                    for pkmn in battle.opponent.reserved:
                        if pkmn.opponent_confirmed == False:
                            battle.opponent.reserved.remove(pkmn)
                    if len(battle.opponent.reserved) == 3:
                        logger.error("NO DEBE HABER 3")
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                contador = 0
                for pkmn in battle.opponent.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        pkmnAux = deepcopy(battle.opponent.active1)
                        battle.opponent.active1 = deepcopy(pkmn)
                        # Inicio - Actualizacion de Stats
                        battle.opponent.active1.name = split_msg[3].partition(",")[0]
                        battle.opponent.active1.volatile_statuses = []
                        battle.opponent.active1.boosts = defaultdict(lambda: 0)
                        if battle.opponent.active1.opponent_confirmed != True:
                            battle.opponent.opponentPkmnConfirmed += 1
                        battle.opponent.active1.opponent_confirmed = True
                        # Fin - Actualizacion de Stats
                        # logger.debug("El oponente ha cambiado de Pokémon a " + str(battle.opponent.active1.name) + ": p1a")
                        # Intercambio en Reservados
                        battle.opponent.reserved[contador] = deepcopy(pkmnAux)
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
                    for pkmn in battle.opponent.reserved:
                        if pkmn.opponent_confirmed == False:
                            battle.opponent.reserved.remove(pkmn)
                    if len(battle.opponent.reserved) == 3:
                        logger.error("NO DEBE HABER 3")
            else:
                contador = 0
                for pkmn in battle.opponent.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        pkmnAux = deepcopy(battle.opponent.active2)
                        battle.opponent.active2 = deepcopy(pkmn)
                        # Inicio - Actualizacion de Stats
                        battle.opponent.active2.name = split_msg[3].partition(",")[0]
                        battle.opponent.active2.volatile_statuses = []
                        battle.opponent.active2.boosts = defaultdict(lambda: 0)
                        if battle.opponent.active2.opponent_confirmed != True:
                            battle.opponent.opponentPkmnConfirmed += 1
                        battle.opponent.active2.opponent_confirmed = True
                        # Fin - Actualizacion de Stats
                        # logger.debug("El oponente ha cambiado de Pokémon a " + str(battle.opponent.active2.name) + ": p1b")
                        # Intercambio en Reservados
                        battle.opponent.reserved[contador] = deepcopy(pkmnAux)
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
                    for pkmn in battle.opponent.reserved:
                        if pkmn.opponent_confirmed == False:
                            battle.opponent.reserved.remove(pkmn)
                    if len(battle.opponent.reserved) == 3:
                        logger.error("NO DEBE HABER 3")
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                contador = 0
                for pkmn in battle.user.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        aux = deepcopy(battle.user.active1)
                        battle.user.active1 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.user.active1.volatile_statuses = []
                        battle.user.active1.boosts = defaultdict(lambda: 0)
                        # Fin - Limpieza de Stats
                        battle.user.active1.name = split_msg[3].partition(",")[0]
                        pkmn = deepcopy(aux)
                        # logger.debug("El usuario ha cambiado de Pokémon a " + str(battle.user.active1.name) + ": p1a")
                        # Intercambio en Reservados
                        battle.user.reserved[contador] = deepcopy(pkmn)
                    contador+=1
                for pkmn in battle.user.reserved:
                    if pkmn is None:
                        battle.user.reserved.remove(pkmn)
            else:
                contador = 0
                for pkmn in battle.user.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        aux = deepcopy(battle.user.active2)
                        battle.user.active2 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.user.active2.volatile_statuses = []
                        battle.user.active2.boosts = defaultdict(lambda: 0)
                        # Fin - Limpieza de Stats
                        battle.user.active2.name = split_msg[3].partition(",")[0]
                        pkmn = deepcopy(aux)
                        # logger.debug("El usuario ha cambiado de Pokémon a " + str(battle.user.active2.name) + ": p1b")
                        # Intercambio en Reservados
                        battle.user.reserved[contador] = deepcopy(pkmn)
                    contador+=1
                for pkmn in battle.user.reserved:
                    if pkmn is None:
                        battle.user.reserved.remove(pkmn)
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                contador = 0
                for pkmn in battle.user.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        aux = deepcopy(battle.user.active1)
                        battle.user.active1 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.user.active1.volatile_statuses = []
                        battle.user.active1.boosts = defaultdict(lambda: 0)
                        # Fin - Limpieza de Stats
                        battle.user.active1.name = split_msg[3].partition(",")[0]
                        pkmn = deepcopy(aux)
                        # logger.debug("El usuario ha cambiado de Pokémon a " + str(battle.user.active1.name) + ": p2a")
                        # Intercambio en Reservados
                        battle.user.reserved[contador] = deepcopy(pkmn)
                    contador+=1
                for pkmn in battle.user.reserved:
                    if pkmn is None:
                        battle.user.reserved.remove(pkmn)
            else:
                contador = 0
                for pkmn in battle.user.reserved:
                    if split_msg[3].partition(",")[0].startswith(pkmn.name[:-1]):
                        aux = deepcopy(battle.user.active2)
                        battle.user.active2 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.user.active2.volatile_statuses = []
                        battle.user.active2.boosts = defaultdict(lambda: 0)
                        # Fin - Limpieza de Stats
                        battle.user.active2.name = split_msg[3].partition(",")[0]
                        pkmn = deepcopy(aux)
                        # logger.debug("El usuario ha cambiado de Pokémon a " + str(battle.user.active2.name) + ": p2b") 
                        # Intercambio en Reservados
                        battle.user.reserved[contador] = deepcopy(pkmn)
                    contador+=1
                for pkmn in battle.user.reserved:
                    if pkmn is None:
                        battle.user.reserved.remove(pkmn)
    return battle

# Intercambio
def swap(battle, split_msg):
    # |swap|p1a: Gothitelle|1|[from] move: Ally Switch
    if is_opponent(battle, split_msg):
        # Opponent
        auxActive1 = deepcopy(battle.opponent.active1)
        battle.opponent.active1 = deepcopy(battle.opponent.active2)
        battle.opponent.active2 = deepcopy(auxActive1)
    else:
        # User
        auxActive1 = deepcopy(battle.user.active1)
        battle.user.active1 = deepcopy(battle.user.active2)
        battle.user.active2 = deepcopy(auxActive1)
    return battle


# Aumento o disminución de HP...
def heal_or_damage(battle, split_msg):
    # split_msg = ["","-damage","p1a: Bibarel","168/282","[from] item: Life Orb"]
    # split_msg = ["","-damage","p2a: Scizor","49/100"]
    # split_msg = ["","-heal","p1a: Aerodactyl","221/265"]
    # split_msg = ["","-damage","p2a: Indeedee","0 fnt"]
    # |-damage|p2a: Tornadus|91/100|[from] item: Life Orb
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                if constants.FNT in split_msg[3]:
                    # Si el Pokémon se ha debilitado debido a un movimiento
                    battle.opponent.active1.HPB = 0
                    #battle.opponent.active1.fainted = True
                else:
                    # Si el Pokémon aún tiene HP, se obtiene el nuevo %
                    new_hp_percentage = float(split_msg[3].split('/')[0]) / 100
                    # Se establece el nuevo %
                    battle.opponent.active1.HPB = int(battle.opponent.active1.max_hp * new_hp_percentage)
            else:
                if constants.FNT in split_msg[3]:
                    # Si el Pokémon se ha debilitado debido a un movimiento
                    battle.opponent.active2.HPB = 0
                    #battle.opponent.active2.fainted = True
                else:
                    # Si el Pokémon aún tiene HP, se obtiene el nuevo %
                    new_hp_percentage = float(split_msg[3].split('/')[0]) / 100
                    # Se establece el nuevo %
                    battle.opponent.active2.HPB = int(battle.opponent.active2.max_hp * new_hp_percentage)
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if constants.FNT in split_msg[3]:
                    # Si el Pokémon se ha debilitado debido a un movimiento
                    battle.opponent.active1.HPB = 0
                    #battle.opponent.active1.fainted = True
                else:
                    # Si el Pokémon aún tiene HP, se obtiene el nuevo %
                    new_hp_percentage = float(split_msg[3].split('/')[0]) / 100
                    # Se establece el nuevo %
                    battle.opponent.active1.HPB = int(battle.opponent.active1.max_hp * new_hp_percentage)
            else:
                if constants.FNT in split_msg[3]:
                    # Si el Pokémon se ha debilitado debido a un movimiento
                    battle.opponent.active2.HPB = 0
                    #battle.opponent.active2.fainted = True
                else:
                    # Si el Pokémon aún tiene HP, se obtiene el nuevo %
                    new_hp_percentage = float(split_msg[3].split('/')[0]) / 100
                    # Se establece el nuevo %
                    battle.opponent.active2.HPB = int(battle.opponent.active2.max_hp * new_hp_percentage)
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if constants.FNT in split_msg[3]:
                    battle.user.active1.HPB = 0
                    #battle.user.active1.fainted = True
                else:
                    battle.user.active1.HPB = int(math.ceil(float(split_msg[3].split('/')[0])))
                    battle.user.active1.max_hp = int(math.ceil(float(split_msg[3].split('/')[1].split(' ')[0])))
            else:
                if constants.FNT in split_msg[3]:
                    battle.user.active2.HPB = 0
                    #battle.user.active2.fainted = True
                else:
                    battle.user.active2.HPB = int(math.ceil(float(split_msg[3].split('/')[0])))
                    battle.user.active2.max_hp = int(math.ceil(float(split_msg[3].split('/')[1].split(' ')[0])))
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                if constants.FNT in split_msg[3]:
                    battle.user.active1.HPB = 0
                    #battle.user.active1.fainted = True
                else:
                    battle.user.active1.HPB = int(math.ceil(float(split_msg[3].split('/')[0])))
                    battle.user.active1.max_hp = int(math.ceil(float(split_msg[3].split('/')[1].split(' ')[0])))
            else:
                if constants.FNT in split_msg[3]:
                    battle.user.active2.HPB = 0
                    #battle.user.active2.fainted = True
                else:
                    battle.user.active2.HPB = int(math.ceil(float(split_msg[3].split('/')[0])))
                    battle.user.active2.max_hp = int(math.ceil(float(split_msg[3].split('/')[1].split(' ')[0])))
    return battle


# Pokémon debilitado...
def faint(battle, split_msg):
    # split_msg = ["","faint","p1b: Tornadus"]
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                battle.opponent.active1.HPB = 0
                #battle.opponent.active1.fainted = True
            else:
                battle.opponent.active2.HPB = 0
                #battle.opponent.active2.fainted = True
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                battle.opponent.active1.HPB = 0
                #battle.opponent.active1.fainted = True
            else:
                battle.opponent.active2.HPB = 0
                #battle.opponent.active2.fainted = True
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                battle.user.active1.HPB = 0
                #battle.user.active1.fainted = True
            else:
                battle.user.active2.HPB = 0
                #battle.user.active2.fainted = True
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                battle.user.active1.HPB = 0
                #battle.user.active1.fainted = True
            else:
                battle.user.active2.HPB = 0
                #battle.user.active2.fainted = True
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


# Estado volátil... Solo considero Dynamax... y Yawn
def start_volatile_status(battle, split_msg):
    # split_msg = ["","-start","p2a: Kyogre","Dynamax"]
    # split_msg = ["","-start","p2a: Incineroar","move: Yawn","[of] p1a: Gastrodon"]
    # split_msg = ["","-start","p1a: Charizard","Dynamax","Gmax"]
    volatile_status = normalize_name(split_msg[3])
    #logger.warning(volatile_status)
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                if volatile_status == constants.DYNAMAX:
                    if battle.opponent.active1.name != 'Shedinja':
                        battle.opponent.active1.HPB = int(battle.opponent.active1.HPB)*2
                        battle.opponent.active1.max_hp = int(battle.opponent.active1.max_hp)*2
                    battle.opponent.active1.volatile_statuses.append(constants.DYNAMAX)
                    battle.opponent.active1.isDynamax = True
                elif volatile_status == constants.DROWSY:
                    battle.opponent.active1.volatile_statuses.append(constants.DROWSY)
            else:
                if volatile_status == constants.DYNAMAX:
                    if battle.opponent.active2.name != 'Shedinja':
                        battle.opponent.active2.HPB = int(battle.opponent.active2.HPB)*2
                        battle.opponent.active2.max_hp = int(battle.opponent.active2.max_hp)*2
                    battle.opponent.active2.volatile_statuses.append(constants.DYNAMAX)
                    battle.opponent.active2.isDynamax = True
                elif volatile_status == constants.DROWSY:
                    battle.opponent.active2.volatile_statuses.append(constants.DROWSY)
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if volatile_status == constants.DYNAMAX:
                    if battle.opponent.active1.name != 'Shedinja':
                        battle.opponent.active1.HPB = int(battle.opponent.active1.HPB)*2
                        battle.opponent.active1.max_hp = int(battle.opponent.active1.max_hp)*2
                    battle.opponent.active1.volatile_statuses.append(constants.DYNAMAX)
                    battle.opponent.active1.isDynamax = True
                elif volatile_status == constants.DROWSY:
                    battle.opponent.active1.volatile_statuses.append(constants.DROWSY)
            else:
                if volatile_status == constants.DYNAMAX:
                    if battle.opponent.active2.name != 'Shedinja':
                        battle.opponent.active2.HPB = int(battle.opponent.active2.HPB)*2
                        battle.opponent.active2.max_hp = int(battle.opponent.active2.max_hp)*2
                    battle.opponent.active2.volatile_statuses.append(constants.DYNAMAX)
                    battle.opponent.active2.isDynamax = True
                elif volatile_status == constants.DROWSY:
                    battle.opponent.active2.volatile_statuses.append(constants.DROWSY)
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if volatile_status == constants.DYNAMAX:
                    if battle.user.active1.name != 'Shedinja':
                        battle.user.active1.HPB = int(battle.user.active1.HPB)*2
                        battle.user.active1.max_hp = int(battle.user.active1.max_hp)*2
                    battle.user.active1.volatile_statuses.append(constants.DYNAMAX)
                    battle.user.active1.isDynamax = True
                    # No more Dynamax
                    battle.user.active1.can_dynamax = False
                    battle.user.active2.can_dynamax = False
                    battle.user.reserved[0].can_dynamax = False
                    battle.user.reserved[1].can_dynamax = False
                elif volatile_status == constants.DROWSY:
                    battle.user.active1.volatile_statuses.append(constants.DROWSY)
            else:
                if volatile_status == constants.DYNAMAX:
                    if battle.user.active2.name != 'Shedinja':
                        battle.user.active2.HPB = int(battle.user.active2.HPB)*2
                        battle.user.active2.max_hp = int(battle.user.active2.max_hp)*2
                    battle.user.active2.volatile_statuses.append(constants.DYNAMAX)
                    battle.user.active2.isDynamax = True
                    # No more Dynamax
                    battle.user.active1.can_dynamax = False
                    battle.user.active2.can_dynamax = False
                    battle.user.reserved[0].can_dynamax = False
                    battle.user.reserved[1].can_dynamax = False
                elif volatile_status == constants.DROWSY:
                    battle.user.active2.volatile_statuses.append(constants.DROWSY)
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                if volatile_status == constants.DYNAMAX:
                    if battle.user.active1.name != 'Shedinja':
                        battle.user.active1.HPB = int(battle.user.active1.HPB)*2
                        battle.user.active1.max_hp = int(battle.user.active1.max_hp)*2
                    battle.user.active1.volatile_statuses.append(constants.DYNAMAX)
                    battle.user.active1.isDynamax = True
                    # No more Dynamax
                    battle.user.active1.can_dynamax = False
                    battle.user.active2.can_dynamax = False
                    battle.user.reserved[0].can_dynamax = False
                    battle.user.reserved[1].can_dynamax = False
                elif volatile_status == constants.DROWSY:
                    battle.user.active1.volatile_statuses.append(constants.DROWSY)
            else:
                if volatile_status == constants.DYNAMAX:
                    if battle.user.active2.name != 'Shedinja':
                        battle.user.active2.HPB = int(battle.user.active2.HPB)*2
                        battle.user.active2.max_hp = int(battle.user.active2.max_hp)*2
                    battle.user.active2.volatile_statuses.append(constants.DYNAMAX)
                    battle.user.active2.isDynamax = True
                    # No more Dynamax
                    battle.user.active1.can_dynamax = False
                    battle.user.active2.can_dynamax = False
                    battle.user.reserved[0].can_dynamax = False
                    battle.user.reserved[1].can_dynamax = False
                elif volatile_status == constants.DROWSY:
                    battle.user.active2.volatile_statuses.append(constants.DROWSY)
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
                    battle.opponent.active1.HPB = int(math.ceil(battle.opponent.active1.HPB/2))
                    battle.opponent.active1.max_hp = int(math.ceil(battle.opponent.active1.max_hp/2))
                    battle.opponent.active1.isDynamax = False
                    battle.opponent.active1.volatile_statuses.remove(constants.DYNAMAX)
            else:
                if volatile_status == constants.DYNAMAX:
                    battle.opponent.active2.HPB = int(math.ceil(battle.opponent.active2.HPB/2))
                    battle.opponent.active2.max_hp = int(math.ceil(battle.opponent.active2.max_hp/2))
                    battle.opponent.active2.isDynamax = False
                    battle.opponent.active2.volatile_statuses.remove(constants.DYNAMAX)
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if volatile_status == constants.DYNAMAX:
                    battle.opponent.active1.HPB = int(math.ceil(battle.opponent.active1.HPB/2))
                    battle.opponent.active1.max_hp = int(math.ceil(battle.opponent.active1.max_hp/2))
                    battle.opponent.active1.isDynamax = False
                    battle.opponent.active1.volatile_statuses.remove(constants.DYNAMAX)
            else:
                if volatile_status == constants.DYNAMAX:
                    battle.opponent.active2.HPB = int(math.ceil(battle.opponent.active2.HPB/2))
                    battle.opponent.active2.max_hp = int(math.ceil(battle.opponent.active2.max_hp/2))
                    battle.opponent.active2.isDynamax = False
                    battle.opponent.active2.volatile_statuses.remove(constants.DYNAMAX)
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if volatile_status == constants.DYNAMAX:
                    battle.user.active1.HPB = int(math.ceil(battle.user.active1.HPB/2))
                    battle.user.active1.max_hp = int(math.ceil(battle.user.active1.max_hp/2))
                    battle.user.active1.isDynamax = False
                    battle.user.active1.volatile_statuses.remove(constants.DYNAMAX)
            else:
                if volatile_status == constants.DYNAMAX:
                    battle.user.active2.HPB = int(math.ceil(battle.user.active2.HPB/2))
                    battle.user.active2.max_hp = int(math.ceil(battle.user.active2.max_hp/2))
                    battle.user.active2.isDynamax = False
                    battle.user.active2.volatile_statuses.remove(constants.DYNAMAX)
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                if volatile_status == constants.DYNAMAX:
                    battle.user.active1.HPB = int(math.ceil(battle.user.active1.HPB/2))
                    battle.user.active1.max_hp = int(math.ceil(battle.user.active1.max_hp/2))
                    battle.user.active1.isDynamax = False
                    battle.user.active1.volatile_statuses.remove(constants.DYNAMAX)
            else:
                if volatile_status == constants.DYNAMAX:
                    battle.user.active2.HPB = int(math.ceil(battle.user.active2.HPB/2))
                    battle.user.active2.max_hp = int(math.ceil(battle.user.active2.max_hp/2))
                    battle.user.active2.isDynamax = False
                    battle.user.active2.volatile_statuses.remove(constants.DYNAMAX)
    return battle


# Un estado se cura...
def curestatus(battle, split_msg):
    # split_msg = ["","-curestatus","p2a: Kyogre","slp","[msg]"]
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                battle.opponent.active1.status = None
                # logger.debug("{} se ha curado...".format(battle.opponent.active1.name))
            else:
                battle.opponent.active2.status = None
                # logger.debug("{} se ha curado...".format(battle.opponent.active2.name))
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                battle.opponent.active1.status = None
                # logger.debug("{} se ha curado...".format(battle.opponent.active1.name))
            else:
                battle.opponent.active2.status = None
                # logger.debug("{} se ha curado...".format(battle.opponent.active2.name))
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                battle.user.active1.status = None
                # logger.debug("{} se ha curado...".format(battle.user.active1.name))
            else:
                battle.user.active2.status = None
                # logger.debug("{} se ha curado...".format(battle.user.active2.name))
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                battle.user.active1.status = None
                # logger.debug("{} se ha curado...".format(battle.user.active1.name))
            else:
                battle.user.active2.status = None
                # logger.debug("{} se ha curado...".format(battle.user.active2.name))
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
        battle.weatherTurn = 0
    else:
        battle.weatherTurn = 5
    return battle


# Inicia un Field...
def fieldstart(battle, split_msg):
    # split_msg = ["","-fieldstart","move: Grassy Terrain","[from] ability: Grassy Surge","[of] p2a: Rillaboom"]
    # split_msg = ["","-fieldstart","move: Electric Terrain"]
    field_name = normalize_name(split_msg[2].split(':')[-1].strip())
    # ¿Qué Field es...?
    if field_name == constants.TRICK_ROOM:
        # Trick Room tiene una duración inicial de 5 turnos
        battle.trick_room = 5
    else:
        battle.terrain = field_name
        battle.terrainTurn = 5
    return battle


# Termina un Field
def fieldend(battle, split_msg):
    field_name = normalize_name(split_msg[2].split(':')[-1].strip())
    # ¿Qué Field es...?
    if field_name == constants.TRICK_ROOM:
        battle.trick_room = False
        battle.trick_roomTurn = 0
    else:
        battle.terrain = None
        battle.terrainTurn = 0
    return battle


# Side conditions are all effects that affect one side of the field. (For example: Tailwind, Stealth Rock, Reflect)
def sidestart(battle, split_msg):
    # split_msg = ["","-sidestart","p1: PopplioPeru98","move: Tailwind"]
    condition = split_msg[3].split(':')[-1].strip()
    condition = normalize_name(condition)
    # condition = "Tailwind"
    if is_opponent(battle, split_msg):
        battle.opponent.side_conditions[condition] = 5
    else:
        battle.user.side_conditions[condition] = 5
    return battle


# Indicates that the side condition CONDITION ended for the given SIDE.
def sideend(battle, split_msg):
    # |-sideend|p2: PopplioPeru98|move: Tailwind
    condition = split_msg[3].split(':')[-1].strip()
    condition = normalize_name(condition)
    # condition = "Tailwind"
    if is_opponent(battle, split_msg):
        # logger.debug("Efecto {} del opponente ha terminado...".format(condition))
        battle.opponent.side_conditions[condition] = 0
    else:
        # logger.debug("Efecto {} del usuario ha terminado...".format(condition))
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


# Cambio de forma permanente
def form_change(battle, split_msg):
    # split_msg = ["","detailschange","p2a: Metagross","Metagross-Mega, L76"]
    # split_msg = ["","detailschange","p1a: Zygarde","Zygarde-Complete, L50"]
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                auxDynamax = battle.opponent.active1.isDynamax
                volatile_statusAux = battle.opponent.active1.volatile_statuses
                battle.opponent.active1 = Pokemon(split_msg[3].partition(",")[0], 0, True)
                battle.opponent.active1.isDynamax = auxDynamax
                battle.opponent.active1.volatile_statuses = volatile_statusAux
            else:
                auxDynamax = battle.opponent.active2.isDynamax
                volatile_statusAux = battle.opponent.active2.volatile_statuses
                battle.opponent.active2 = Pokemon(split_msg[3].partition(",")[0], 0, True)
                battle.opponent.active2.isDynamax = auxDynamax
                battle.opponent.active2.volatile_statuses = volatile_statusAux
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                auxDynamax = battle.opponent.active1.isDynamax
                volatile_statusAux = battle.opponent.active1.volatile_statuses
                battle.opponent.active1 = Pokemon(split_msg[3].partition(",")[0], 0, True)
                battle.opponent.active1.isDynamax = auxDynamax
                battle.opponent.active1.volatile_statuses = volatile_statusAux
            else:
                auxDynamax = battle.opponent.active2.isDynamax
                volatile_statusAux = battle.opponent.active2.volatile_statuses
                battle.opponent.active2 = Pokemon(split_msg[3].partition(",")[0], 0, True)
                battle.opponent.active2.isDynamax = auxDynamax
                battle.opponent.active2.volatile_statuses = volatile_statusAux
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                auxDynamax = battle.user.active1.isDynamax
                volatile_statusAux = battle.user.active1.volatile_statuses
                battle.user.active1 = Pokemon(split_msg[3].partition(",")[0], 0)
                battle.user.active1.isDynamax = auxDynamax
                battle.user.active1.volatile_statuses = volatile_statusAux
            else:
                auxDynamax = battle.user.active2.isDynamax
                volatile_statusAux = battle.user.active2.volatile_statuses
                battle.user.active2 = Pokemon(split_msg[3].partition(",")[0], 0)
                battle.user.active2.isDynamax = auxDynamax
                battle.user.active2.volatile_statuses = volatile_statusAux
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                auxDynamax = battle.user.active1.isDynamax
                volatile_statusAux = battle.user.active1.volatile_statuses
                battle.user.active1 = Pokemon(split_msg[3].partition(",")[0], 0)
                battle.user.active1.isDynamax = auxDynamax
                battle.user.active1.volatile_statuses = volatile_statusAux
            else:
                auxDynamax = battle.user.active2.isDynamax
                volatile_statusAux = battle.user.active2.volatile_statuses
                battle.user.active2 = Pokemon(split_msg[3].partition(",")[0], 0)
                battle.user.active2.isDynamax = auxDynamax
                battle.user.active2.volatile_statuses = volatile_statusAux
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
            # logger.debug("Configurando estadísticas de {} de {} a 0".format(battle.user.active1.name, stat))
            battle.user.active1.boosts[stat] = 0
    # Usuario
    for stat, value in battle.user.active2.boosts.items():
        if value != 0:
            # logger.debug("Configurando estadísticas de {} de {} a 0".format(battle.user.active2.name, stat))
            battle.user.active2.boosts[stat] = 0
    # opponente
    for stat, value in battle.opponent.active1.boosts.items():
        if value != 0:
            # logger.debug("Configurando estadísticas de {} de {} a 0".format(battle.opponent.active1.name, stat))
            battle.opponent.active1.boosts[stat] = 0
    # opponente
    for stat, value in battle.opponent.active2.boosts.items():
        if value != 0:
            # logger.debug("Configurando estadísticas de {} de {} a 0".format(battle.opponent.active2.name, stat))
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
    # |-transform|p1b: Ditto|p2a: Incineroar|[from] ability: Imposter
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                if battle.opponent.active1.isDitto == 1:
                    auxHP = battle.opponent.active1.HPB
                    battle.opponent.active1 = deepcopy(battle.user.active2)
                    battle.opponent.active1.HPB = auxHP
                else:
                    battle.opponent.active1 = deepcopy(battle.user.active2)
                battle.opponent.active1.isDitto = 1
            else:
                if battle.opponent.active2.isDitto == 1:
                    auxHP = battle.opponent.active2.HPB
                    battle.opponent.active2 = deepcopy(battle.user.active1)
                    battle.opponent.active2.HPB = auxHP
                else:
                    battle.opponent.active2 = deepcopy(battle.user.active1)
                battle.opponent.active2.isDitto = 1
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if battle.opponent.active1.isDitto == 1:
                    auxHP = battle.opponent.active1.HPB
                    battle.opponent.active1 = deepcopy(battle.user.active2)
                    battle.opponent.active1.HPB = auxHP
                else:
                    battle.opponent.active1 = deepcopy(battle.user.active2)
                battle.opponent.active1.isDitto = 1
            else:
                if battle.opponent.active2.isDitto == 1:
                    auxHP = battle.opponent.active2.HPB
                    battle.opponent.active2 = deepcopy(battle.user.active1)
                    battle.opponent.active2.HPB = auxHP
                else:
                    battle.opponent.active2 = deepcopy(battle.user.active1)
                battle.opponent.active2.isDitto = 1
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if battle.user.active1.isDitto == 1:
                    auxHP = battle.user.active1.HPB
                    battle.user.active1 = deepcopy(battle.opponent.active2)
                    battle.user.active1.HPB = auxHP
                else:
                    battle.user.active1 = deepcopy(battle.opponent.active2)
                battle.user.active1.isDitto = 1
            else:
                if battle.user.active2.isDitto == 1:
                    auxHP = battle.user.active2.HPB
                    battle.user.active2 = deepcopy(battle.opponent.active1)
                    battle.user.active2.HPB = auxHP
                else:
                    battle.user.active1 = deepcopy(battle.opponent.active1)
                battle.user.active2.isDitto = 1
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                if battle.user.active1.isDitto == 1:
                    auxHP = battle.user.active1.HPB
                    battle.user.active1 = deepcopy(battle.opponent.active2)
                    battle.user.active1.HPB = auxHP
                else:
                    battle.user.active1 = deepcopy(battle.opponent.active2)
                battle.user.active1.isDitto = 1
            else:
                if battle.user.active2.isDitto == 1:
                    auxHP = battle.user.active2.HPB
                    battle.user.active2 = deepcopy(battle.opponent.active1)
                    battle.user.active2.HPB = auxHP
                else:
                    battle.user.active1 = deepcopy(battle.opponent.active1)
                battle.user.active2.isDitto = 1
    return battle


# Actualiza el turno
def turn(battle, split_msg):
    # split_msg = ["","turn","1"]
    battle.turn = int(split_msg[2])
    if battle.weatherTurn > 0:
        battle.weatherTurn -= 1
    if battle.terrainTurn > 0:
        battle.terrainTurn -= 1
    if battle.trick_room > 0:
        battle.trick_room -= 1
    if battle.user.side_conditions['tailwind'] > 0:
        battle.user.side_conditions['tailwind'] -= 1
    if battle.user.side_conditions['reflect'] > 0:
        battle.user.side_conditions['reflect'] -= 1
    if battle.user.side_conditions['lightscreen'] > 0:
        battle.user.side_conditions['lightscreen'] -= 1
    if battle.user.side_conditions['auroraveil'] > 0:
        battle.user.side_conditions['auroraveil'] -= 1
    if battle.opponent.side_conditions['tailwind'] > 0:
        battle.opponent.side_conditions['tailwind'] -= 1
    if battle.opponent.side_conditions['reflect'] > 0:
        battle.opponent.side_conditions['reflect'] -= 1
    if battle.opponent.side_conditions['lightscreen'] > 0:
        battle.opponent.side_conditions['lightscreen'] -= 1
    if battle.opponent.side_conditions['auroraveil'] > 0:
        battle.opponent.side_conditions['auroraveil'] -= 1
    # Yawn
    if constants.DROWSY in battle.user.active1.volatile_statuses:
        battle.user.active1.volatile_statuses.remove(constants.DROWSY)
    if constants.DROWSY in battle.user.active2.volatile_statuses:
        battle.user.active2.volatile_statuses.remove(constants.DROWSY)
    if constants.DROWSY in battle.opponent.active1.volatile_statuses:
        battle.opponent.active1.volatile_statuses.remove(constants.DROWSY)
    if constants.DROWSY in battle.opponent.active2.volatile_statuses:
        battle.opponent.active2.volatile_statuses.remove(constants.DROWSY)
    # Print Turn
    logger.info("Turno: {}".format(int(split_msg[2])))
    return battle


# ¿?
def noinit(battle, split_msg):
    if split_msg[2] == "rename":
        battle.battle_tag = split_msg[3]
        # logger.debug("Renamed battle to {}".format(battle.battle_tag))
    return battle

# Update Pkmn in Battle()
def updatePokemon(battle, pkmn, isUser, isReserved):
    # Reserved: boosts = 0
    if isReserved:
        pkmn.boosts = defaultdict(lambda: 0)
    # Inicializacion - No modificables los valores originales
    ATK_battle_base =  pkmn.ATK_battle_base
    DEF_battle_base =  pkmn.DEF_battle_base
    SPA_battle_base =  pkmn.SPA_battle_base
    SPD_battle_base =  pkmn.SPD_battle_base
    VEL_battle_base =  pkmn.VEL_battle_base
    # Stats Modifiers
    if 'attack' in pkmn.boosts:
        if pkmn.boosts['attack'] > 0:
            ATK_battle_base = math.floor(((2+int(pkmn.boosts['attack']))/2) * pkmn.ATK_battle_base)
        else:
            ATK_battle_base = math.floor((2/(2+abs(int(pkmn.boosts['attack'])))) * pkmn.ATK_battle_base)
    if 'defense' in pkmn.boosts:
        if pkmn.boosts['defense'] > 0:
            DEF_battle_base = math.floor(((2+int(pkmn.boosts['defense']))/2) * pkmn.DEF_battle_base)
        else:
            DEF_battle_base = math.floor((2/(2+abs(int(pkmn.boosts['defense'])))) * pkmn.DEF_battle_base)
    if 'special-attack' in pkmn.boosts:
        if pkmn.boosts['special-attack'] > 0:
            SPA_battle_base = math.floor(((2+int(pkmn.boosts['special-attack']))/2) * pkmn.SPA_battle_base)
        else:
            SPA_battle_base = math.floor((2/(2+abs(int(pkmn.boosts['special-attack'])))) * pkmn.SPA_battle_base)
    if 'special-defense' in pkmn.boosts:
        if pkmn.boosts['special-defense'] > 0:
            SPD_battle_base = math.floor(((2+int(pkmn.boosts['special-defense']))/2) * pkmn.SPD_battle_base)
        else:
            SPD_battle_base = math.floor((2/(2+abs(int(pkmn.boosts['special-defense'])))) * pkmn.SPD_battle_base)
    if 'speed' in pkmn.boosts:
        if pkmn.boosts['speed'] > 0:
            VEL_battle_base = math.floor(((2+int(pkmn.boosts['speed']))/2) * pkmn.VEL_battle_base)
        else:
            VEL_battle_base = math.floor((2/(2+abs(int(pkmn.boosts['speed'])))) * pkmn.VEL_battle_base)
    # Volatile Status:
    if isUser:
        # Reflect
        if "reflect" in battle.user.side_conditions and "auroraveil" not in battle.user.side_conditions:
            DEF_battle_base = int(DEF_battle_base * 4/3)
        # Light Screen
        if "lightscreen" in battle.user.side_conditions and "auroraveil" not in battle.user.side_conditions:
            SPD_battle_base = int(SPD_battle_base * 4/3)
        # Aurora Veil
        if "auroraveil" in battle.user.side_conditions:
            DEF_battle_base = int(DEF_battle_base * 4/3)
            SPD_battle_base = int(SPD_battle_base * 4/3)
    else:
        # Reflect
        if "reflect" in battle.opponent.side_conditions and "auroraveil" not in battle.opponent.side_conditions:
            DEF_battle_base = int(DEF_battle_base * 4/3)
        # Light Screen
        if "lightscreen" in battle.opponent.side_conditions and "auroraveil" not in battle.opponent.side_conditions:
            SPD_battle_base = int(SPD_battle_base * 4/3)
        # Aurora Veil
        if "auroraveil" in battle.opponent.side_conditions:
            DEF_battle_base = int(DEF_battle_base * 4/3)
            SPD_battle_base = int(SPD_battle_base * 4/3)
    # ATB
    if pkmn.status == constants.BURN:
        pkmn.ATB = max(ATK_battle_base/2, SPA_battle_base)
    elif pkmn.status == constants.SLEEP:
        pkmn.ASL = 1
        pkmn.ATB = math.floor(max(ATK_battle_base, SPA_battle_base)/3)
    elif pkmn.status == constants.FROZEN:
        pkmn.FRO = 1
        pkmn.ATB = math.floor(max(ATK_battle_base, SPA_battle_base)/10)
    else:
        if constants.DROWSY in pkmn.volatile_statuses:
            pkmn.DRO = 1
            pkmn.ATB = math.floor(max(ATK_battle_base, SPA_battle_base)/2)
        else:
            pkmn.ATB = max(ATK_battle_base, SPA_battle_base)
    # DEB - 'None'
    pkmn.DEB = math.floor((DEF_battle_base + SPD_battle_base)/2)
    # VEL - 'par'
    if pkmn.status == constants.PARALYZED:
        pkmn.PAR = 1
        pkmn.VEB = math.floor(VEL_battle_base/2)
    else:
        pkmn.VEB = VEL_battle_base
    # Tailwind
    if isUser:
        if battle.user.side_conditions['tailwind'] > 0:
            battle.user.active1.VEB *= 2
            battle.user.active2.VEB *= 2
    else:
        if battle.opponent.side_conditions['tailwind'] > 0:
            battle.opponent.active1.VEB *= 2
            battle.opponent.active2.VEB *= 2
    # Trick Room
    if battle.trick_room > 0:
        pkmn.VEB = constants.FASTEST - pkmn.VEB   # Pkmn fastest = 1
    # Dynamax - ATB *= 1.3
    if 'dynamax' in pkmn.volatile_statuses:
        pkmn.ATB = math.floor(pkmn.ATB * 1.3)
    # Actualización
    if isUser:
        if battle.user.active1.name == pkmn.name:
            battle.user.active1 = deepcopy(pkmn)
        elif battle.user.active2.name == pkmn.name:
            battle.user.active2 = deepcopy(pkmn)
    else:
        if battle.opponent.active1.name == pkmn.name:
            battle.opponent.active1 = deepcopy(pkmn)
        elif battle.opponent.active2.name == pkmn.name:
            battle.opponent.active2 = deepcopy(pkmn)
    # ATBoost & DEBoost
    if isUser:
        pkmn.ATBoost = getATBoost(pkmn.moveType, battle.opponent.active1.type1, battle.opponent.active1.type2, battle.opponent.active2.type1, battle.opponent.active2.type2, battle)
        pkmn.DEBoost = getDEBoost(battle.opponent.active1.moveType, battle.opponent.active2.moveType, pkmn.type1, pkmn.type2, battle)
    else:
        pkmn.ATBoost = getATBoost(pkmn.moveType, battle.user.active1.type1, battle.user.active1.type2, battle.user.active2.type1, battle.user.active2.type2, battle)
        pkmn.DEBoost = getDEBoost(battle.user.active1.moveType, battle.user.active2.moveType, pkmn.type1, pkmn.type2, battle)
    return battle

async def updateEnvironment(battle):
    # User
    battle = updatePokemon(battle, battle.user.active1, True, False)
    battle = updatePokemon(battle, battle.user.active2, True, False)
    for i in range(len(battle.user.reserved)):
        battle = updatePokemon(battle, battle.user.reserved[i], True, True)
    # Opponent
    battle = updatePokemon(battle, battle.opponent.active1, False, False)
    battle = updatePokemon(battle, battle.opponent.active2, False, False)
    for i in range(len(battle.opponent.reserved)):
        battle = updatePokemon(battle, battle.opponent.reserved[i], False, True)
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
            'swap': swap,
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