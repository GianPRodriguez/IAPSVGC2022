import json
import math
import constants

from copy import deepcopy
from collections import defaultdict
from teams.team_converter import normalize_name
from showdown.battle import Pokemon
from showdown.battle import Battle

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
        try:
            battle_json = json.loads(split_msg[2].strip('\''))
        except:
            return battle
        # Obtiene el rqid, que al parecer no se utilizará
        battle.rqid = battle_json[constants.RQID]
        # ¿battle_json tiene "forceSwitch"? Se activa cuando uso U-TURN
        if battle_json.get(constants.FORCE_SWITCH):
            if not battle.isTraining:
                print("[INFO]     Action: forceSwitch")
            battle.force_switch = True
        else:
            battle.force_switch = False
        # ¿battle_json tiene "wait"? ¿Se activa cuando el opponente usa U-TURN?
        if battle_json.get(constants.WAIT):
            if not battle.isTraining:
                print("[INFO]     Action: Wait")
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
    #regex_string = "(\d+) sec this turn"
    #if split_msg[2].startswith(constants.TIME_LEFT):
    #    capture = re.search(regex_string, split_msg[2])
    #    time_left = int(capture.group(1))
    #    # logger.info("Tiempo restante: {} segundos...".format(time_left))
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
    name = split_msg[3].partition(",")[0]
    if name.lower() == 'mimikyu':
        name = "mimikyu-disguised"
    if name.lower() in constants.standardName:
        namePkmn = constants.standardName[name.lower()]
    else:
        namePkmn = name.lower()
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                contador = 0
                for pkmn in battle.opponent.reserved:
                    if namePkmn == pkmn.name:
                        pkmnAux = deepcopy(battle.opponent.active1)
                        battle.opponent.active1 = deepcopy(pkmn)
                        # Inicio - Actualizacion de Stats
                        battle.opponent.active1.name = namePkmn
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
                        ...
                        #logger.error("NO DEBE HABER 3")
            else:
                contador = 0
                for pkmn in battle.opponent.reserved:
                    if namePkmn == pkmn.name:
                        pkmnAux = deepcopy(battle.opponent.active2)
                        battle.opponent.active2 = deepcopy(pkmn)
                        # Inicio - Actualizacion de Stats
                        battle.opponent.active2.name = namePkmn
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
                        ...
                        #logger.error("NO DEBE HABER 3")
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                contador = 0
                for pkmn in battle.opponent.reserved:
                    if namePkmn == pkmn.name:
                        pkmnAux = deepcopy(battle.opponent.active1)
                        battle.opponent.active1 = deepcopy(pkmn)
                        # Inicio - Actualizacion de Stats
                        battle.opponent.active1.name = namePkmn
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
                        ...
                        #logger.error("NO DEBE HABER 3")
            else:
                contador = 0
                for pkmn in battle.opponent.reserved:
                    if namePkmn == pkmn.name:
                        pkmnAux = deepcopy(battle.opponent.active2)
                        battle.opponent.active2 = deepcopy(pkmn)
                        # Inicio - Actualizacion de Stats
                        battle.opponent.active2.name = namePkmn
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
                        ...
                        #logger.error("NO DEBE HABER 3")
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                contador = 0
                for pkmn in battle.user.reserved:
                    if namePkmn == pkmn.name:
                        aux = deepcopy(battle.user.active1)
                        battle.user.active1 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.user.active1.volatile_statuses = []
                        battle.user.active1.boosts = defaultdict(lambda: 0)
                        # Fin - Limpieza de Stats
                        battle.user.active1.name = namePkmn
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
                    if namePkmn == pkmn.name:
                        aux = deepcopy(battle.user.active2)
                        battle.user.active2 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.user.active2.volatile_statuses = []
                        battle.user.active2.boosts = defaultdict(lambda: 0)
                        # Fin - Limpieza de Stats
                        battle.user.active2.name = namePkmn
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
                    if namePkmn == pkmn.name:
                        aux = deepcopy(battle.user.active1)
                        battle.user.active1 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.user.active1.volatile_statuses = []
                        battle.user.active1.boosts = defaultdict(lambda: 0)
                        # Fin - Limpieza de Stats
                        battle.user.active1.name = namePkmn
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
                    if namePkmn == pkmn.name:
                        aux = deepcopy(battle.user.active2)
                        battle.user.active2 = deepcopy(pkmn)
                        # Inicio - Limpieza de Stats
                        battle.user.active2.volatile_statuses = []
                        battle.user.active2.boosts = defaultdict(lambda: 0)
                        # Fin - Limpieza de Stats
                        battle.user.active2.name = namePkmn
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

#####################################################################################################################################
# Aumento de estadísticas
#####################################################################################################################################
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
    # split_msg = ["","detailschange","p2a: Metagross","Metagross-Mega", "L76"]
    # split_msg = ["","detailschange","p1a: Zygarde","Zygarde-Complete", "L50"]
    name = split_msg[3].partition(",")[0]
    if name.lower() in constants.standardName:
        namePkmn = constants.standardName[name.lower()]
    else:
        namePkmn = name.lower()
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                if namePkmn == 'mimikyu':
                    battle.opponent.active1.name = 'mimikyu-busted'
                else:
                    auxDynamax = battle.opponent.active1.isDynamax
                    volatile_statusAux = battle.opponent.active1.volatile_statuses
                    battle.opponent.active1 = Pokemon(namePkmn, 0, True)
                    battle.opponent.active1.isDynamax = auxDynamax
                    battle.opponent.active1.volatile_statuses = volatile_statusAux
            else:
                if namePkmn == 'mimikyu':
                    battle.opponent.active2.name = 'mimikyu-busted'
                else:
                    auxDynamax = battle.opponent.active2.isDynamax
                    volatile_statusAux = battle.opponent.active2.volatile_statuses
                    battle.opponent.active2 = Pokemon(namePkmn, 0, True)
                    battle.opponent.active2.isDynamax = auxDynamax
                    battle.opponent.active2.volatile_statuses = volatile_statusAux
        else:
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                if namePkmn == 'mimikyu':
                    battle.opponent.active1.name = 'mimikyu-busted'
                else:
                    auxDynamax = battle.opponent.active1.isDynamax
                    volatile_statusAux = battle.opponent.active1.volatile_statuses
                    battle.opponent.active1 = Pokemon(namePkmn, 0, True)
                    battle.opponent.active1.isDynamax = auxDynamax
                    battle.opponent.active1.volatile_statuses = volatile_statusAux
            else:
                if namePkmn == 'mimikyu':
                    battle.opponent.active2.name = 'mimikyu-busted'
                else:
                    auxDynamax = battle.opponent.active2.isDynamax
                    volatile_statusAux = battle.opponent.active2.volatile_statuses
                    battle.opponent.active2 = Pokemon(namePkmn, 0, True)
                    battle.opponent.active2.isDynamax = auxDynamax
                    battle.opponent.active2.volatile_statuses = volatile_statusAux
    else:
        if split_msg[2].startswith("p1"):
            # ¿p1a o p1b?
            if split_msg[2].startswith("p1a"):
                auxDynamax = battle.user.active1.isDynamax
                volatile_statusAux = battle.user.active1.volatile_statuses
                battle.user.active1 = Pokemon(namePkmn, 0)
                battle.user.active1.isDynamax = auxDynamax
                battle.user.active1.volatile_statuses = volatile_statusAux
            else:
                auxDynamax = battle.user.active2.isDynamax
                volatile_statusAux = battle.user.active2.volatile_statuses
                battle.user.active2 = Pokemon(namePkmn, 0)
                battle.user.active2.isDynamax = auxDynamax
                battle.user.active2.volatile_statuses = volatile_statusAux
        else:
            # ¿p2a o p2b?
            if split_msg[2].startswith("p2a"):
                auxDynamax = battle.user.active1.isDynamax
                volatile_statusAux = battle.user.active1.volatile_statuses
                battle.user.active1 = Pokemon(namePkmn, 0)
                battle.user.active1.isDynamax = auxDynamax
                battle.user.active1.volatile_statuses = volatile_statusAux
            else:
                auxDynamax = battle.user.active2.isDynamax
                volatile_statusAux = battle.user.active2.volatile_statuses
                battle.user.active2 = Pokemon(namePkmn, 0)
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

def error(battle, split_msg):
    battle.error = True
    #print("JUGADA INVALIDA")
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
    # Turn
    if not battle.isTraining:
        print("[INFO]     Turno: {}".format(int(split_msg[2])))
    return battle


# ¿?
def noinit(battle, split_msg):
    if split_msg[2] == "rename":
        battle.battle_tag = split_msg[3]
        # logger.debug("Renamed battle to {}".format(battle.battle_tag))
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
# ???
############################################################################################################################################################
BATTLE_MODIFIERS_LOOKUP = {
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
    'error': error,
}

async def update_battle_environment(battle: Battle, msg: str, msg_request: str) -> None:
    # Concatenamos los dos mensajes si existe msg_request para procesar todas las líneas en un solo flujo
    full_msg = f"{msg}\n{msg_request}" if msg_request else msg
    msg_lines = full_msg.split('\n')

    for line in msg_lines:
        split_msg = line.split('|')
        if len(split_msg) < 2:
            continue

        action = split_msg[1].strip()

        function_to_call = BATTLE_MODIFIERS_LOOKUP.get(action)
        if function_to_call is not None:
            battle = function_to_call(battle, split_msg)
            if action != 'error':
                battle.error = False