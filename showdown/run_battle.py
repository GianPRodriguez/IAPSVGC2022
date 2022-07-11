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


def battle_is_finished(battle_tag, msg):
    logger.info("Tracking: run_battle.py / def battle_is_finished()")
    return (
        msg.startswith(">{}".format(battle_tag)) and
        (constants.WIN_STRING in msg or constants.TIE_STRING in msg) and
        constants.CHAT_STRING not in msg
    )


async def async_pick_move(battle):
    logger.info("Tracking: run_battle.py / sync def async_pick_move()")
    battle_copy = deepcopy(battle)
    if battle_copy.request_json:
        battle_copy.user.from_json(battle_copy.request_json)

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        best_move = await loop.run_in_executor(
            pool, battle_copy.find_best_move
        )
    choice = best_move[0]
    if constants.SWITCH_STRING in choice:
        battle.user.last_used_move = LastUsedMove(battle.user.active.name, "switch {}".format(choice.split()[-1]), battle.turn)
    else:
        battle.user.last_used_move = LastUsedMove(battle.user.active.name, choice.split()[2], battle.turn)
    logger.info(best_move)
    best_move = input()
    return ast.literal_eval(best_move)



async def handle_team_preview(battle, ps_websocket_client):
    battle_copy = deepcopy(battle)
    battle_copy.user.active = Pokemon.get_dummy()
    battle_copy.opponent.active = Pokemon.get_dummy()

    
    best_move = await async_pick_move(battle_copy)
    size_of_team = len(battle.user.reserve) + 1
    team_list_indexes = list(range(1, size_of_team))
    choice_digit = int(best_move[0].split()[-1])

    team_list_indexes.remove(choice_digit)
    message = ["/team {}{}|{}".format(choice_digit, "".join(str(x) for x in team_list_indexes), battle.rqid)]

    await ps_websocket_client.send_message(battle.battle_tag, message)



async def get_battle_tag_and_opponent(ps_websocket_client: PSWebsocketClient):
    logger.debug("run_battle.py / async def get_battle_tag_and_opponent")
    while True:
        msg = await ps_websocket_client.receive_message()
        split_msg = msg.split('|')
        first_msg = split_msg[0]
        if 'battle' in first_msg:
            battle_tag = first_msg.replace('>', '').strip()
            user_name = split_msg[-1].replace('☆', '').strip()
            opponent_name = split_msg[4].replace(user_name, '').replace('vs.', '').strip()
            return battle_tag, opponent_name


async def initialize_battle_with_tag(ps_websocket_client: PSWebsocketClient, set_request_json=True):
    logger.debug("run_battle.py / async def initialize_battle_with_tag")
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


async def read_messages_until_first_pokemon_is_seen(ps_websocket_client, battle, opponent_id, user_json):
    logger.info("run_battle.py / async def read_messages_until_first_pokemon_is_seen")
    # keep reading messages until the opponent's first pokemon is seen
    # this is run when starting non team-preview battles
    while True:
        msg = await ps_websocket_client.receive_message()
        if constants.START_STRING in msg:
            split_msg = msg.split(constants.START_STRING)[-1].split('\n')
            for line in split_msg:
                if opponent_id in line and constants.SWITCH_STRING in line:
                    battle.start_non_team_preview_battle(user_json, line)

                elif battle.started:
                    await async_update_battle(battle, line)

            # first move needs to be picked here
            best_move = await async_pick_move(battle)
            await ps_websocket_client.send_message(battle.battle_tag, best_move)

            return


async def start_random_battle(ps_websocket_client: PSWebsocketClient, pokemon_battle_type):
    logger.info("Tracking: run_battle.py / async def start_random_battle")
    battle, opponent_id, user_json = await initialize_battle_with_tag(ps_websocket_client)
    battle.battle_type = constants.RANDOM_BATTLE
    battle.generation = pokemon_battle_type[:4]

    await read_messages_until_first_pokemon_is_seen(ps_websocket_client, battle, opponent_id, user_json)

    return battle


# Inicia una Batalla Standard
async def start_standard_battle(ps_websocket_client: PSWebsocketClient, pokemon_battle_type):
    battle, opponent_id, user_json = await initialize_battle_with_tag(ps_websocket_client, set_request_json=False)
    battle.battle_type = constants.STANDARD_BATTLE
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
    await handle_team_preview(battle, ps_websocket_client)

    return battle

async def start_battle(ps_websocket_client, pokemon_battle_type):
    battle = await start_standard_battle(ps_websocket_client, pokemon_battle_type)

    await ps_websocket_client.send_message(battle.battle_tag, [config.greeting_message])
    await ps_websocket_client.send_message(battle.battle_tag, ['/timer on'])

    return battle


# Nueva Batalla Pokémon completo
async def pokemon_battle(ps_websocket_client, pokemon_battle_type):
    # Solo inicia la batalla
    battle = await start_battle(ps_websocket_client, pokemon_battle_type)
    while True:
        msg = await ps_websocket_client.receive_message()

        # ¿Ha finalizado la batalla?
        if battle_is_finished(battle.battle_tag, msg):
            # Sí ha finalizado la batalla... se obtiene información del ganador
            if constants.WIN_STRING in msg:
                # Se obtiene el NOMBRE del ganador
                winner = msg.split(constants.WIN_STRING)[-1].split('\n')[0].strip(" ")
            else:
                winner = None
            # Y lo imprimo
            logger.debug("Ganador {}".format(winner))
            # Se envía un mensaje al contrincante... es opcional
            await ps_websocket_client.send_message(battle.battle_tag, [config.battle_ending_message])
            # Es opcional guardar la batalla...
            await ps_websocket_client.leave_battle(battle.battle_tag, save_replay=config.save_replay)
            # Retorna al ganador
            return winner
        else:
            # No ha finalizado la batalla... se debe realizar una acción!!
            action_required = await async_update_battle(battle, msg)
            # action_required es BOOLEAN
            if action_required and not battle.wait:
                # Escoge el mejor movimiento y lo envía al websocket
                best_move = await async_pick_move(battle)
                await ps_websocket_client.send_message(battle.battle_tag, best_move)