import importlib
import json
import asyncio
import concurrent.futures
from copy import deepcopy
import logging
import ast

import data
from data.helpers import get_standard_battle_sets
import constants
import config
from showdown.engine.evaluate import Scoring
from showdown.battle import Pokemon
from showdown.battle import LastUsedMove
from showdown.battle_modifier import async_update_battle

from showdown.websocket_client import PSWebsocketClient

logger = logging.getLogger(__name__)


# Retorna un Boolean
def battle_is_finished(battle_tag, msg):
    #<< >battle-gen8vgc2022-1614957698-shk265rg7lhkyzn0tyjofy9lr7myd0gpw||win|PopplioPeru98
    return (
        msg.startswith(">{}".format(battle_tag)) and
        (constants.WIN_STRING in msg or constants.TIE_STRING in msg) and
        constants.CHAT_STRING not in msg
    )


# Retorna el Mensaje de Acción
async def accion_batalla(battle):
    # Copia del Estado Actual
    battle_copy = deepcopy(battle)
    # json
    if battle_copy.request_json:
        battle_copy.user.from_json(battle_copy.request_json)
    # 
    best_action = battle_copy.find_best_move
    #
    if constants.SWITCH_STRING in choice:
        # hola
        battle.user.last_used_move = LastUsedMove(battle.user.active.name, "switch {}".format(choice.split()[-1]), battle.turn)
    else:
        # 
        battle.user.last_used_move = LastUsedMove(battle.user.active.name, choice.split()[2], battle.turn)
    #
    logger.info(best_action)
    #
    return ast.literal_eval(best_action)


# Se envía la acción para TEAM PREVIEW
async def accion_team_preview(battle):
    # Configurado para INPUT por CMD
    print('Ingresa los 4 dígitos para el Team Preview... Ejemplo: "3142"')
    # eleccion es un entero: 3142
    eleccion = input()
    # Creo un array de enteros
    arrayEleccion = [int(a) for a in str(eleccion)]
    eleccion0 = arrayEleccion[0]
    eleccion1 = arrayEleccion[1]
    eleccion2 = arrayEleccion[2]
    eleccion3 = arrayEleccion[3]
    # Regreso los 4 enteros
    return int(eleccion0), int(eleccion1), int(eleccion2), int(eleccion3)


# Se define la etapa TEAM PREVIEW
async def etapa_team_preview(battle, ps_websocket_client):
    # DeepCopy del Estado Actual
    battle_copy = deepcopy(battle)
    # Se escoge los 4 Pokémon: eleccion(n) es un entero
    eleccion0, eleccion1, eleccion2, eleccion3 = await async_pick_move(battle_copy)
    # indices es una lista: [1,2,3,4,5,6]
    indices = list(range(1, 6))
    # Remuevo los 4 Pokémon seleccionados, solo quedan los no seleccionados
    indices.remove(eleccion0)
    indices.remove(eleccion1)
    indices.remove(eleccion2)
    indices.remove(eleccion3)
    # Se prepara el String que se va a enviar al WebSocket
    # >> battle-gen8vgc2022-1614957698-shk265rg7lhkyzn0tyjofy9lr7myd0gpw|/team 314256|2
    mensaje = ["/team {}{}{}{}{}|{}".format(eleccion0, eleccion1, eleccion2, eleccion3, "".join(str(x) for x in indices), battle.rqid)]
    # Se envía el String al WebSocket
    await ps_websocket_client.send_message(battle.battle_tag, mensaje)


#|request|
#|j|☆Nerbort
#|t:|1657546734
#|gametype|doubles
#|player|p1|PopplioPeru98|2|1171


async def get_battle_tag_and_opponent(ps_websocket_client: PSWebsocketClient):
    # Obtiene las etiquetas que aparecen al iniciar una batalla
    while True:
        # El string recibido lo guardo en msg
        msg = await ps_websocket_client.receive_message()
        #|init|battle
        #|title|PopplioPeru98 vs. Nerbort
        #|j|☆PopplioPeru98
        split_msg = msg.split('|')
        first_msg = split_msg[0]
        if 'battle' in first_msg:
            battle_tag = first_msg.replace('>', '').strip()
            user_name = split_msg[-1].replace('☆', '').strip()
            opponent_name = split_msg[4].replace(user_name, '').replace('vs.', '').strip()
            return battle_tag, opponent_name


async def initialize_battle_with_tag(ps_websocket_client: PSWebsocketClient, set_request_json=True):
    battle_module = importlib.import_module('showdown.battle_bots.{}.main'.format(config.battle_bot_module))

    battle_tag, opponent_name = await get_battle_tag_and_opponent(ps_websocket_client)
    while True:
        msg = await ps_websocket_client.receive_message()
        split_msg = msg.split('|')
        if split_msg[1].strip() == 'request' and split_msg[2].strip():
            user_json = json.loads(split_msg[2].strip('\''))
            user_id = user_json[constants.SIDE][constants.ID]
            opponent_id = constants.ID_LOOKUP[user_id]
            # Aquí se hace el cambio de bot y llamada de BattleBot...
            battle = battle_module.BattleBot(battle_tag)
            battle.opponent.name = opponent_id
            battle.opponent.account_name = opponent_name

            if set_request_json:
                battle.request_json = user_json

            return battle, opponent_id, user_json


# Inicia una Batalla VGC 2022
async def start_vgc2022_battle(ps_websocket_client: PSWebsocketClient, pokemon_battle_type):
    #
    battle, opponent_id, user_json = await initialize_battle_with_tag(ps_websocket_client, set_request_json=False)
    #
    battle.battle_type = constants.STANDARD_BATTLE
    #
    battle.generation = pokemon_battle_type[:4]

    # Inicializa mensaje
    msg = ''
    # Hasta que se cargue...
    while constants.START_TEAM_PREVIEW not in msg:
        msg = await ps_websocket_client.receive_message()

    preview_string_lines = msg.split(constants.START_TEAM_PREVIEW)[-1].split('\n')

    opponent_pokemon = []
    for line in preview_string_lines:
        if not line:
            continue

        split_line = line.split('|')
        if split_line[1] == constants.TEAM_PREVIEW_POKE and split_line[2].strip() == opponent_id:
            opponent_pokemon.append(split_line[3])

    # TEAM PREVIEW
    battle.initialize_team_preview(user_json, opponent_pokemon, pokemon_battle_type)

    # 
    await etapa_team_preview(battle, ps_websocket_client)

    return battle


async def start_battle(ps_websocket_client, pokemon_battle_type):
    # Inicia la Batalla
    battle = await start_vgc2022_battle(ps_websocket_client, pokemon_battle_type)
    # Envía el saludo
    await ps_websocket_client.send_message(battle.battle_tag, [config.greeting_message])
    # Inicia el tiempo
    await ps_websocket_client.send_message(battle.battle_tag, ['/timer on'])
    # Retorna la Clase Batalla
    return battle


# Nueva Batalla Pokémon completo
async def pokemon_battle(ps_websocket_client):
    # Solo inicia la batalla
    battle = await start_battle(ps_websocket_client)
    # Mientras...
    while True:
        # Recibo mensaje del WebSocket
        mensaje = await ps_websocket_client.receive_message()
        # ¿Ha finalizado la batalla?
        if battle_is_finished(battle.battle_tag, mensaje):
            # Sí ha finalizado la batalla... se obtiene información del ganador
            if constants.WIN_STRING in mensaje:
                # Se obtiene el NOMBRE del ganador
                winner = mensaje.split(constants.WIN_STRING)[-1].split('\n')[0].strip(" ")
            else:
                winner = None
            # Y lo imprimo
            logger.debug("Ganador {}".format(winner))
            # Se envía un mensaje de despedida al contricante
            await ps_websocket_client.send_message(battle.battle_tag, [config.battle_ending_message])
            # Es opcional guardar la batalla...
            await ps_websocket_client.leave_battle(battle.battle_tag, save_replay=config.save_replay)
            # Retorna al ganador
            return winner
        else:
            # No ha finalizado la batalla... se debe realizar una acción!!
            action_required = await async_update_battle(battle, mensaje)
            # action_required es BOOLEAN
            if action_required and not battle.wait:
                # Escoge el mejor movimiento y lo envía al websocket
                best_action = await accion_batalla(battle)
                await ps_websocket_client.send_message(battle.battle_tag, best_action)