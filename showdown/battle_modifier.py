import re
import json
from copy import deepcopy
import logging

import constants
from data import all_move_json
from data import pokedex
from showdown.battle import Pokemon
from showdown.battle import LastUsedMove
from showdown.battle import DamageDealt
from showdown.battle import StatRange
from showdown.engine.helpers import normalize_name
from showdown.engine.helpers import get_pokemon_info_from_condition
from showdown.engine.helpers import calculate_stats
from showdown.engine.find_state_instructions import get_effective_speed
from showdown.engine.damage_calculator import calculate_damage
from showdown.engine.objects import boost_multiplier_lookup


logger = logging.getLogger(__name__)


MOVE_END_STRINGS = {'move', 'switch', 'upkeep', ''}

def find_pokemon_in_reserves(pkmn_name, reserves):
    for reserve_pkmn in reserves:
        if pkmn_name.startswith(reserve_pkmn.name) or reserve_pkmn.name.startswith(pkmn_name) or reserve_pkmn.base_name == pkmn_name:
            return reserve_pkmn
    return None


# ¿El cambio es hecho por el oponente?
def is_opponent(battle,  split_msg):
    # split_msg = ["","switch","p1a: Incineroar","Incineroar, L50, M","190/202"]
    # split_msg[2] = "p1a: Incineroar"
    return not split_msg[2].startswith(battle.user.name)


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
        # ¿battle_json tiene "wait"? ¿Se activa cuando el oponente usa U-TURN?
        if battle_json.get(constants.WAIT):
            logger.debug("Action: wait")
            battle.wait = True
        else:
            battle.wait = False
        # Se almacena el JSON para elegir una accióm
        if not battle.wait:
            battle.request_json = battle_json


# Avanza el tiempo...
def inactive(battle, split_msg):
    # split_msg = ["","inactive","Time left: 55 sec this turn "," 410 sec total"]
    regex_string = "(\d+) sec this turn"
    if split_msg[2].startswith(constants.TIME_LEFT):
        capture = re.search(regex_string, split_msg[2])
        time_left = int(capture.group(1))
        logger.debug("Tiempo restante: {} segundos...".format(time_left))


# Ya no avanza el tiempo...
def inactiveoff(battle, _):
    logger.debug("Se ha detenido el tiempo...")


# Ha habido un cambio de Pokémon...
def switch_or_drag(battle, split_msg):
    # |switch|POKEMON|DETAILS|HP STATUS or |drag|POKEMON|DETAILS|HP STATUS
    # |switch|p1a: Incineroar|Incineroar, L50, M|190/202
    # ¿El cambio es hecho por el oponente?
    if is_opponent(battle, split_msg):
        # Sí es hecho por el oponente...
        side = battle.opponent
        #logger.debug("Opponent has switched - clearing the last used move")
        logger.debug("El oponente ha cambiado de Pokémon...")
    else:
        side = battle.user
        side.side_conditions[constants.TOXIC_COUNT] = 0
        logger.debug("El usuario ha cambiado de Pokémon...")

    # Si el Pokémon cambiado tenía algún valor original...
    if side.active is not None:
        # set the pkmn's types back to their original value if the types were changed
        if constants.TYPECHANGE in side.active.volatile_statuses:
            original_types = pokedex[side.active.name][constants.TYPES]
            logger.debug("{} had it's type changed - changing its types back to {}".format(side.active.name, original_types))
            side.active.types = original_types

        # if the target was transformed, reset its transformed attributes
        if constants.TRANSFORM in side.active.volatile_statuses:
            logger.debug("{} was transformed. Resetting its transformed attributes".format(side.active.name))
            side.active.stats = calculate_stats(side.active.base_stats, side.active.level)
            side.active.ability = None
            side.active.moves = []
            side.active.types = pokedex[side.active.name][constants.TYPES]

        # reset the boost of the pokemon being replaced
        side.active.boosts.clear()

        # reset the volatile statuses of the pokemon being replaced
        side.active.volatile_statuses.clear()

        # reset toxic count for this side
        side.side_conditions[constants.TOXIC_COUNT] = 0

        # if the side is alive and has regenerator, give it back 1/3 of it's maxhp
        if side.active.hp > 0 and not side.active.fainted and side.active.ability == "regenerator":
            health_healed = int(side.active.max_hp / 3)
            side.active.hp = min(side.active.hp + health_healed, side.active.max_hp)
            logger.debug(
                "{} switched out with regenerator. Healing it to {}/{}".format(
                    side.active.name, side.active.hp, side.active.max_hp
                )
            )

    # check if the pokemon exists in the reserves
    # if it does not, then the newly-created pokemon is used (for formats without team preview)
    pkmn = Pokemon.from_switch_string(split_msg[3])
    pkmn = find_pokemon_in_reserves(pkmn.name, side.reserve)

    if pkmn is None:
        pkmn = Pokemon.from_switch_string(split_msg[3])
    else:
        side.reserve.remove(pkmn)

    side.last_used_move = LastUsedMove(
        pokemon_name=None,
        move='switch {}'.format(pkmn.name),
        turn=battle.turn
    )

    # pkmn != active is a special edge-case for Zoroark
    if side.active is not None and pkmn != side.active:
        side.reserve.append(side.active)

    side.active = pkmn
    if side.active.name in constants.UNKOWN_POKEMON_FORMES:
        side.active = Pokemon.from_switch_string(split_msg[3])


# Aumento o disminución de HP...
def heal_or_damage(battle, split_msg):
    # split_msg = ["","-damage","p1a: Bibarel","168/282","[from] item: Life Orb"]
    # split_msg = ["","-damage","p2a: Scizor","49/100"]
    # split_msg = ["","-heal","p1a: Aerodactyl","221/265"]
    # split_msg = ["","-damage","p2a: Indeedee","0 fnt"]
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            aux = "p2b"
        else:
            aux = "p2a"
        if split_msg[2].startswith(aux):
            # Se establece el aumento o disminución...
            if constants.FNT in split_msg[3]:
                # Si el Pokémon se ha debilitado debido a un movimiento
                battle.opponent.activo1.hp = 0
            else:
                # Si el Pokémon aún tiene HP, se obtiene el nuevo %
                new_hp_percentage = float(split_msg[3].split('/')[0]) / 100
                # Se establece el nuevo %
                battle.opponent.activo1.hp = battle.opponent.activo1.max_hp * new_hp_percentage
        else:
            # Se establece el aumento o disminución...
            if constants.FNT in split_msg[3]:
                # Si el Pokémon se ha debilitado debido a un movimiento
                battle.oponent.activo2.hp = 0
            else:
                # Si el Pokémon aún tiene HP, se obtiene el nuevo %
                new_hp_percentage = float(split_msg[3].split('/')[0]) / 100
                # Se establece el nuevo %
                battle.oponent.activo2.hp = battle.oponent.activo2.max_hp * new_hp_percentage
    else:
        if split_msg[2].startswith("p1"):
            aux = "p1a"
        else:
            aux = "p1b"
        if split_msg[2].startswith(aux):
            # Se establece el aumento o disminución...
            if constants.FNT in split_msg[3]:
                battle.user.activo1.hp = 0
            else:
                battle.user.activo1.hp = float(split_msg[3].split('/')[0])
                battle.user.activo1.max_hp = float(split_msg[3].split('/')[1].split()[0])
        else:
            # Se establece el aumento o disminución...
            if constants.FNT in split_msg[3]:
                battle.user.activo2.hp = 0
            else:
                battle.user.activo2.hp = float(split_msg[3].split('/')[0])
                battle.user.activo2.max_hp = float(split_msg[3].split('/')[1].split()[0])


# Pokémon debilitado...
def faint(battle, split_msg):
    # split_msg = ["","faint","p1b: Tornadus"]
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            aux = "p2b"
        else:
            aux = "p2a"
        if split_msg[2].startswith(aux):
            # Pokémon debilitado debe tener HP = 0
            battle.opponent.activo1.hp = 0
        else:
            # Pokémon debilitado debe tener HP = 0
            battle.oponent.activo2.hp = 0
    else:
        if split_msg[2].startswith("p1"):
            aux = "p1a"
        else:
            aux = "p1b"
        if split_msg[2].startswith(aux):
            # Pokémon debilitado debe tener HP = 0
            battle.user.activo1.hp = 0
        else:
            # Pokémon debilitado debe tener HP = 0
            battle.user.activo2.hp = 0

# Se realiza un movimiento...
def move(battle, split_msg):
    # split_msg = ["","move","p2a: Scizor","Bug Bite","p1a: Vileplume"]
    # split_msg = ["","move","p1b: Zacian","Behemoth Blade","p2b: Zekrom"]
    # No importa para el agente los movimientos registrados


# Aumento de estadísticas
def boost(battle, split_msg):
    # split_msg = ["","-boost","p2b: Celesteela","spe","1"]
    # Se obtiene el nombre de la estadística que ha incrementado... stat = "SPEED"
    stat = constants.STAT_ABBREVIATION_LOOKUPS[split_msg[3].strip()]
    # Se obtiene la cantidad en que ha aumentado... amount = 1
    amount = int(split_msg[4].strip())
    # Se define en qué Pokémon se realiza el incremento en la estadística...
    if is_opponent(battle, split_msg):
        # p2 (solo para este ejemplo)
        if split_msg[2].startswith("p2"):
            aux = "p2b"
        else:
            aux = "p2a"
        if split_msg[2].startswith(aux):
            # p2a (solo para este ejemplo)
            # Como la cantidad no puede superar los 6... hay un límite y se actualiza la cifra verdadera...
            battle.opponent.activo1.boosts[stat] = min(battle.opponent.activo1[stat] + amount, constants.MAX_BOOSTS)
        else:
            # p2b (solo para este ejemplo)
            # Como la cantidad no puede superar los 6... hay un límite y se actualiza la cifra verdadera...
            battle.oponent.activo2.boosts[stat] = min(battle.opponent.activo2[stat] + amount, constants.MAX_BOOSTS)
    else:
        if split_msg[2].startswith("p1"):
            aux = "p1a"
        else:
            aux = "p1b"
        # p1 (solo para este ejemplo)
        if split_msg[2].startswith(aux):
            # p1a (solo para este ejemplo)
            # Como la cantidad no puede superar los 6... hay un límite y se actualiza la cifra verdadera...
            battle.user.activo1.boosts[stat] = min(battle.user.activo1[stat] + amount, constants.MAX_BOOSTS)
        else:
            # p1b (solo para este ejemplo)
            # Como la cantidad no puede superar los 6... hay un límite y se actualiza la cifra verdadera...
            pkmn = battle.user.activo2.boosts[stat] = min(battle.user.activo2[stat] + amount, constants.MAX_BOOSTS)


# Disminución de estadísticas
def unboost(battle, split_msg):
    # split_msg = ["","-unboost","p2a: Whimsicott","spe","1"]
    # Se obtiene el nombre de la estadística que ha disminuido... stat = "SPEED"
    stat = constants.STAT_ABBREVIATION_LOOKUPS[split_msg[3].strip()]
    # Se obtiene la cantidad en que ha aumentado... amount = 1
    amount = int(split_msg[4].strip())
    # Se define en qué Pokémon se realiza el decremento en la estadística...
    if is_opponent(battle, split_msg):
        # p2 (solo para este ejemplo)
        if split_msg[2].startswith("p2"):
            aux = "p2b"
        else:
            aux = "p2a"
        if split_msg[2].startswith(aux):
            # p2a (solo para este ejemplo)
            # Como la cantidad no puede superar los -6... hay un límite y se actualiza la cifra verdadera...
            battle.opponent.activo1.boosts[stat] = max(battle.opponent.activo1[stat] - amount, -1*constants.MAX_BOOSTS)
        else:
            # p2b (solo para este ejemplo)
            # Como la cantidad no puede superar los -6... hay un límite y se actualiza la cifra verdadera...
            battle.oponent.activo2.boosts[stat] = max(battle.opponent.activo2[stat] - amount, -1*constants.MAX_BOOSTS)
    else:
        if split_msg[2].startswith("p1"):
            aux = "p1a"
        else:
            aux = "p1b"
        # p1 (solo para este ejemplo)
        if split_msg[2].startswith(aux):
            # p1a (solo para este ejemplo)
            # Como la cantidad no puede superar los -6... hay un límite y se actualiza la cifra verdadera...
            battle.user.activo1.boosts[stat] = max(battle.user.activo1[stat] - amount, -1*constants.MAX_BOOSTS)
        else:
            # p1b (solo para este ejemplo)
            # Como la cantidad no puede superar los -6... hay un límite y se actualiza la cifra verdadera...
            pkmn = battle.user.activo2.boosts[stat] = max(battle.user.activo2[stat] - amount, -1*constants.MAX_BOOSTS)


# Un Pokémon ha cambiado de status
def status(battle, split_msg):
    # split_msg = ["","-status","p2a: Slowbro","brn"]
    status_name = split_msg[3].strip()
    # status_name = "brn"
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            aux = "p2b"
        else:
            aux = "p2a"
        if split_msg[2].startswith(aux):
            battle.opponent.activo1.status = status_name
            logger.debug("{} ha obtenido el estado: {}".format(battle.opponent.activo1.name, status_name))
        else:
            battle.oponent.activo2.status = status_name
            logger.debug("{} ha obtenido el estado: {}".format(battle.oponent.activo2.name, status_name))
    else:
        if split_msg[2].startswith("p1"):
            aux = "p1a"
        else:
            aux = "p1b"
        if split_msg[2].startswith(aux):
            battle.user.activo1.status = status_name
            logger.debug("{} ha obtenido el estado: {}".format(battle.user.activo1.name, status_name))
        else:
            battle.user.activo2.status = status_name
            logger.debug("{} ha obtenido el estado: {}".format(battle.user.activo2.name, status_name))


# Se activa una habilidad...
def activate(battle, split_msg):
    # split_msg = ["","-activate","p2b: Indeedee","move: Psychic Terrain"]


# Se prepara para un movimiento...
def prepare(battle, split_msg):
    # ...


# Estado volátil... Solo considero Dynamax...
def start_volatile_status(battle, split_msg):
    # split_msg = ["","-start","p2a: Kyogre","Dynamax"]
    # split_msg = ["","-start","p2a: Incineroar","move: Yawn","[of] p1a: Gastrodon"]
    # split_msg = ["","-start","p1a: Charizard","Dynamax","Gmax"]
    volatile_status = normalize_name(split_msg[3].split(":")[-1])
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            aux = "p2b"
        else:
            aux = "p2a"
        if split_msg[2].startswith(aux):
            if volatile_status == constants.DYNAMAX:
                battle.opponent.activo1.hp *= 2
                battle.opponent.activo1.max_hp *= 2
                logger.debug("{} ha sido afectado por el fenómeno dinamax - se duplica su HP a {}/{}".format(battle.opponent.activo1.name, battle.opponent.activo1.hp, battle.opponent.activo1.max_hp))
        else:
            if volatile_status == constants.DYNAMAX:battle.opponent.activo1
                battle.opponent.activo2.hp *= 2
                battle.opponent.activo2.max_hp *= 2
                logger.debug("{} ha sido afectado por el fenómeno dinamax - se duplica su HP a {}/{}".format(battle.opponent.activo2.name, battle.opponent.activo2.hp, battle.opponent.activo2.max_hp))
    else:
        if split_msg[2].startswith("p1"):
            aux = "p1a"
        else:
            aux = "p1b"
        if split_msg[2].startswith(aux):
            if volatile_status == constants.DYNAMAX:
                battle.user.activo1.hp *= 2
                battle.user.activo1.max_hp *= 2
                logger.debug("{} ha sido afectado por el fenómeno dinamax - se duplica su HP a {}/{}".format(battle.user.activo1.name, battle.user.activo1.hp, battle.user.activo1.max_hp))
        else:
            if volatile_status == constants.DYNAMAX:
                battle.user.activo2.hp *= 2
                battle.user.activo2.max_hp *= 2
                logger.debug("{} ha sido afectado por el fenómeno dinamax - se duplica su HP a {}/{}".format(battle.user.activo2.name, battle.user.activo2.hp, battle.user.activo2.max_hp))



# Fin estado volátil...
def end_volatile_status(battle, split_msg):
    # split_msg = ["","-end","p2a: Kyogre","Dynamax"]
    volatile_status = normalize_name(split_msg[3].split(":")[-1])
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            aux = "p2b"
        else:
            aux = "p2a"
        if split_msg[2].startswith(aux):
            if volatile_status == constants.DYNAMAX:
                battle.opponent.activo1.hp /= 2
                battle.opponent.activo1.max_hp /= 2
                logger.debug("{} ha terminado el fenómeno dinamax - se reduce a la mitad su HP a {}/{}".format(battle.opponent.activo1.name, battle.opponent.activo1.hp, battle.opponent.activo1.max_hp))
        else:
            if volatile_status == constants.DYNAMAX:battle.opponent.activo1
                battle.opponent.activo2.hp /= 2
                battle.opponent.activo2.max_hp /= 2
                logger.debug("{} ha terminado el fenómeno dinamax - se reduce a la mitad su HP a {}/{}".format(battle.opponent.activo2.name, battle.opponent.activo2.hp, battle.opponent.activo2.max_hp))
    else:
        if split_msg[2].startswith("p1"):
            aux = "p1a"
        else:
            aux = "p1b"
        if split_msg[2].startswith(aux):
            if volatile_status == constants.DYNAMAX:
                battle.user.activo1.hp /= 2
                battle.user.activo1.max_hp /= 2
                logger.debug("{} ha terminado el fenómeno dinamax - se reduce a la mitad su HP a {}/{}".format(battle.user.activo1.name, battle.user.activo1.hp, battle.user.activo1.max_hp))
        else:
            if volatile_status == constants.DYNAMAX:
                battle.user.activo2.hp /= 2
                battle.user.activo2.max_hp /= 2
                logger.debug("{} ha terminado el fenómeno dinamax - se reduce a la mitad su HP a {}/{}".format(battle.user.activo2.name, battle.user.activo2.hp, battle.user.activo2.max_hp))


# Un estado se cura...
def curestatus(battle, split_msg):
    # split_msg = ["","-curestatus","p2a: Kyogre","slp","[msg]"]
    if is_opponent(battle, split_msg):
        if split_msg[2].startswith("p2"):
            aux = "p2b"
        else:
            aux = "p2a"
        if split_msg[2].startswith(aux):
            if volatile_status == constants.DYNAMAX:
                battle.opponent.activo1.status = None
                logger.debug("{} se ha curado...".format(battle.opponent.activo1.name))
        else:
            if volatile_status == constants.DYNAMAX:battle.opponent.activo1
                battle.opponent.activo2.status = None
                logger.debug("{} se ha curado...".format(battle.opponent.activo2.name))
    else:
        if split_msg[2].startswith("p1"):
            aux = "p1a"
        else:
            aux = "p1b"
        if split_msg[2].startswith(aux):
            if volatile_status == constants.DYNAMAX:
                battle.user.activo1.status = None
                logger.debug("{} se ha curado...".format(battle.user.activo1.name))
        else:
            if volatile_status == constants.DYNAMAX:
                battle.user.activo2.status = None
                logger.debug("{} se ha curado...".format(battle.user.activo2.name))


# The Pokémon POKEMON has used a move that cures its team of status effects, like Heal Bell.
def cureteam(battle, split_msg):
    # No se usa...


# Se establece un Clima... no me importa quién lo realiza
# split_msg = "|-weather|Sandstorm"
# split_msg = "|-weather|none"
def weather(battle, split_msg):
    # weather_name = "Sandstorm" || weather_name = "none"
    weather_name = normalize_name(split_msg[2].split(':')[-1].strip())
    # Mensaje en el CMD
    logger.debug("El clima {} ha empezado...".format(weather_name))
    # Se establece en class Battle:
    battle.weather = weather_name


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
    else:
        # Otro Field
        logger.debug("Ha iniciado el campo {}...".format(field_name))
        battle.field = field_name


# Termina un Field
def fieldend(battle, split_msg):
    field_name = normalize_name(split_msg[2].split(':')[-1].strip())
    # ¿Qué Field es...?
    if field_name == constants.TRICK_ROOM:
        # Trick Room
        logger.debug("Ha terminado Trick Room...")
        battle.trick_room = False
    else:
        # Otro Field
        logger.debug("Ha terminado el campo {}...".format(field_name))
        battle.field = None


# Side conditions are all effects that affect one side of the field. (For example: Tailwind, Stealth Rock, Reflect)
def sidestart(battle, split_msg):
    # split_msg = ["","-sidestart","p1: PopplioPeru98","move: Tailwind"]
    condition = split_msg[3].split(':')[-1].strip()
    condition = normalize_name(condition)
    # condition = "Tailwind"
    if is_opponent(battle, split_msg):
        logger.debug("Efecto {} ha sido iniciado por el oponente...".format(condition))
        battle.opponent.side_conditions[condition] += 1
    else:
        logger.debug("Efecto {} ha sido iniciado por el usuario...".format(condition))
        battle.user.side_conditions[condition] += 1


# Indicates that the side condition CONDITION ended for the given SIDE.
def sideend(battle, split_msg):
    # |-sideend|p2: PopplioPeru98|move: Tailwind
    condition = split_msg[3].split(':')[-1].strip()
    condition = normalize_name(condition)
    # condition = "Tailwind"
    if is_opponent(battle, split_msg):
        logger.debug("Efecto {} del oponente ha terminado...".format(condition))
        battle.opponent.side_conditions[condition] = 0
    else:
        logger.debug("Efecto {} del usuario ha terminado...".format(condition))
        battle.user.side_conditions[condition] = 0


# Intercambia efectos
def swapsideconditions(battle, _):
    # No se usa...


# Establece Objeto
def set_item(battle, split_msg):
    # No se considera...


# Remueve Objeto...
def remove_item(battle, split_msg):
    # No se considera...


# Establece habilidad
def set_ability(battle, split_msg):
    # No se considera...


# Establece habilidad
def set_opponent_ability_from_ability_tag(battle, split_msg):
    # No se considera...


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


# Ataque Z
def zpower(battle, split_msg):
    # No VGC 2022


# Reestablece estadísticas negativas
def clearnegativeboost(battle, split_msg):
    # No VGC 2022


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
    # Oponente
    for stat, value in battle.opponent.active1.boosts.items():
        if value != 0:
            logger.debug("Configurando estadísticas de {} de {} a 0".format(battle.opponent.active1.name, stat))
            battle.opponent.active1.boosts[stat] = 0
    # Oponente
    for stat, value in battle.opponent.active2.boosts.items():
        if value != 0:
            logger.debug("Configurando estadísticas de {} de {} a 0".format(battle.opponent.active2.name, stat))
            battle.opponent.active2.boosts[stat] = 0


# The Pokémon POKEMON used move MOVE which causes a temporary effect lasting the duration of the turn. (For example: Protect, Focus Punch, Roost).
def singleturn(battle, split_msg):
    # No se considera movimientos


# Signals the upkeep phase of the turn where the number of turns left for field conditions are updated.
def upkeep(battle, _):
    # No se considera


# Mega-evolución
def mega(battle, split_msg):
    # No VGC 2022


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


# Actualiza el turno
def turn(battle, split_msg):
    # split_msg = ["","turn","1"]
    battle.turn = int(split_msg[2])


# 
def noinit(battle, split_msg):
    if split_msg[2] == "rename":
        battle.battle_tag = split_msg[3]
        logger.debug("Renamed battle to {}".format(battle.battle_tag))


# Actualiza estado
def update_battle(battle, msg):
    msg_lines = msg.split('\n')

    action = None
    check_speed_ranges(battle, msg_lines)
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
            function_to_call(battle, split_msg)
        if action == 'turn':
            return True

    if action in ['inactive', 'updatesearch']:
        return False

    if action != "request":
        return battle.force_switch


async def async_update_battle(battle, msg):
    return update_battle(battle, msg)