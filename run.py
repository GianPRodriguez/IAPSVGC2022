from copy import deepcopy
from environs import Env
from config import init_logging
from threading import Thread
from teams import load_team
from multiprocessing import Process, Queue
from showdown.run_battle import pokemon_battle
from showdown.websocket_client import PSWebsocketClient
from data import all_move_json
from data import pokedex

import asyncio
import json
import constants
import config
import logging

logger = logging.getLogger(__name__)


def parse_configs():
    env = Env()
    env.read_env()

    config.battle_bot_module = env("BATTLE_BOT", 'safest')
    config.save_replay = env.bool("SAVE_REPLAY", config.save_replay)
    config.use_relative_weights = env.bool("USE_RELATIVE_WEIGHTS", config.use_relative_weights)
    config.gambit_exe_path = env("GAMBIT_PATH", config.gambit_exe_path)
    config.search_depth = int(env("MAX_SEARCH_DEPTH", config.search_depth))
    config.dynamic_search_depth = env.bool("DYNAMIC_SEARCH_DEPTH", config.dynamic_search_depth)
    config.greeting_message = env("GREETING_MESSAGE", config.greeting_message)
    config.battle_ending_message = env("BATTLE_OVER_MESSAGE", config.battle_ending_message)
    config.websocket_uri = env("WEBSOCKET_URI", "sim.smogon.com:8000")
    config.username = env("PS_USERNAME")
    config.password = env("PS_PASSWORD")
    config.bot_mode = env("BOT_MODE")
    config.team_name = env("TEAM_NAME", None)
    config.pokemon_mode = env("POKEMON_MODE", constants.DEFAULT_MODE)
    config.run_count = int(env("RUN_COUNT", 1))
    config.room_name = env("ROOM_NAME", config.room_name)

    if config.bot_mode == constants.CHALLENGE_USER:
        config.user_to_challenge = env("USER_TO_CHALLENGE")
    init_logging(env("LOG_LEVEL", "DEBUG"))


# MAIN
async def showdown(q):
    # Configuración de la Aplicación
    parse_configs()
    # Creo una Clase PSWebsocketClient
    ps_websocket_client = await PSWebsocketClient.create(config.username, config.password, config.websocket_uri)
    # Se lee "|challstr|4|a6e6e29652941bc1b872a2c34cb0b4e6fdd83355bd3b8787d1519685b29bfb5d267f479f4be9f9fe554605c6c9842160194586f55644f532d991560f817c5939bf7961319b23de0803bac5f6faa2935e395c0f28da55b4c052ece76f0f64e5acdfea8691ac2607012b0d742704920c9fe3c6bc13730b121d6459b84f03d399b8"
    await ps_websocket_client.login()
    # Contadores
    battles_run = 0
    wins = 0
    losses = 0
    # Hasta que no se alcance la cantidad de batallas definidas
    while True:
        # Se lee el EQUIPO
        team = load_team(config.team_name)
        # Opciones de juego
        if config.bot_mode == constants.CHALLENGE_USER:
            await ps_websocket_client.challenge_user(config.user_to_challenge, config.pokemon_mode, team)
        elif config.bot_mode == constants.ACCEPT_CHALLENGE:
            await ps_websocket_client.accept_challenge(config.pokemon_mode, team, config.room_name)
        elif config.bot_mode == constants.SEARCH_LADDER:
            # Validación ante Jugadores Humanos
            await ps_websocket_client.search_for_match(team)
        else:
            raise ValueError("Invalid Bot Mode")
        # Nueva Batalla Pokémon completo
        winner, q = await pokemon_battle(ps_websocket_client, q)
        q.put("HOLIII")
        if winner == config.username:
            wins += 1
        else:
            losses += 1

        logger.info("W: {}\tL: {}".format(wins, losses))

        battles_run += 1
        if battles_run >= config.run_count:
            break
    # Como terminaron las batallas pendientes...
    q.put([-1])

class AsyncDownload(Thread):
    def __init__(self, q):
        super().__init__()
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(showdown(q))