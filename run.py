from copy import deepcopy
from environs import Env
from config import init_logging
from threading import Thread
from teams import load_team
from showdown.run_battle import pokemon_battle
from asyncio.tasks import FIRST_COMPLETED
from showdown.websocket_client import PSWebsocketClient
import sys
import asyncio
import json
import constants
import config
import queue


def parse_configs():
    env = Env()
    env.read_env()

    config.battle_bot_module = env("BATTLE_BOT", 'smart')
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

# MAIN
async def showdown(q, q50, qm50, q150, username, password, teamname, botmode, opponentUsername, training, agentTraining, numBattles):
    # Configuración de la Aplicación
    parse_configs()
    # Parameters
    if not training:
        config.user_to_challenge = username
        config.bot_mode = botmode
        config.team_name = 'gen8/vgc2022/' + teamname
    else:
        team_name = 'gen8/vgc2022/' + teamname
    if config.bot_mode == constants.SEARCH_LADDER:
        config.username = username
        config.password = password
    if training:
        ps_websocket_client = await PSWebsocketClient.create(username, password, config.websocket_uri)
        isOk = await ps_websocket_client.login(training)
    else:
        # Creo una Clase PSWebsocketClient
        ps_websocket_client = await PSWebsocketClient.create(config.username, config.password, config.websocket_uri)
        isOk = await ps_websocket_client.login(training)
    # Se lee "|challstr|4|a6e6e29652941bc1b872a2c34cb0b4e6fdd83355bd3b8787d1519685b29bfb5d267f479f4be9f9fe554605c6c9842160194586f55644f532d991560f817c5939bf7961319b23de0803bac5f6faa2935e395c0f28da55b4c052ece76f0f64e5acdfea8691ac2607012b0d742704920c9fe3c6bc13730b121d6459b84f03d399b8"
    if isOk == -1:
        print("[INFO]     ¡Usuario o contraseña incorrecta!")
        q.put([-5])
        sys.exit()
    # Contadores
    battles_run = numBattles
    wins = 0
    losses = 0
    # Hasta que no se alcance la cantidad de batallas definidas
    if training:
        while True:
            # Se lee el EQUIPO dado un Nombre Específico...
            team = load_team(team_name)[0]
            # Opciones de juego
            if botmode == constants.CHALLENGE_USER:
                await ps_websocket_client.challenge_user(opponentUsername, config.pokemon_mode, team, training)
                #print("challenge_user")
                if agentTraining:
                    bucle = True
                    while bucle:
                        try:
                            if qm50.qsize() > 0:
                                cola = qm50.get(0)
                                if cola[0] == -1:
                                    return
                                else:
                                    bucle = False
                        except EOFError as e:
                            ...
            elif botmode == constants.ACCEPT_CHALLENGE:
                await ps_websocket_client.accept_challenge(config.pokemon_mode, team, config.room_name, training, q)
            elif botmode == constants.SEARCH_LADDER:
                # Validación ante Jugadores Humanos
                await ps_websocket_client.search_for_match(team)
            else:
                raise ValueError("Invalid Bot Mode")
            # Nueva Batalla Pokémon completo
            winner, q, battle_tag = await pokemon_battle(ps_websocket_client, q, q50, qm50, q150, botmode, training, agentTraining)
            if winner == "None":
                 message = ["/forfeit"]
                 await ps_websocket_client.send_message(battle_tag, message)
            if winner == username:
                wins += 1
            else:
                losses += 1
            battles_run += 1
            if battles_run >= config.run_count:
                break
    else:
        while True:
            # Se lee el EQUIPO dado un Nombre Específico...
            team = load_team(config.team_name)[0]
            # Opciones de juego
            if config.bot_mode == constants.CHALLENGE_USER:
                if training:
                    await ps_websocket_client.challenge_user(opponentUsername, config.pokemon_mode, team, False)
                else:
                    await ps_websocket_client.challenge_user(config.user_to_challenge, config.pokemon_mode, team, False)
            elif config.bot_mode == constants.ACCEPT_CHALLENGE:
                await ps_websocket_client.accept_challenge(config.pokemon_mode, team, config.room_name, False)
            elif config.bot_mode == constants.SEARCH_LADDER:
                # Validación ante Jugadores Humanos
                await ps_websocket_client.search_for_match(team)
            else:
                raise ValueError("Invalid Bot Mode")
            # Nueva Batalla Pokémon completo
            winner, q, battle_tag = await pokemon_battle(ps_websocket_client, q, q50, qm50, q150, botmode, training, False)
            if winner == config.username:
                wins += 1
            else:
                losses += 1
            if training:
                print("[INFO]     W: {}\tL: {}".format(wins, losses))
            battles_run += 1
            if battles_run >= config.run_count:
                break
    # Fin
    #raise Exception("FIN DE LA BATALLA")
    return


async def stop(q, q50, qm50, q150, username, password, teamname, botmode, opponentUsername, training, agentTraining, numBattles):
    poller = asyncio.ensure_future(showdown(q, q50, qm50, q150, username, password, teamname, botmode, opponentUsername, training, agentTraining, numBattles))
    while True:
        await asyncio.sleep(0.1)
        try:
            cola = q.get(0)
            if cola[0] == -1000:
                poller.cancel()
                return
        except queue.Empty:
            ...

class AsyncDownload(Thread):
    def __init__(self,q,q50,qm50,q150,username,password,teamname,botmode,opponentUsername,training,agentTraining,numBattles):
        super().__init__()
        if training:
            asyncio.run(showdown(q,q50,qm50,q150,username,password,teamname,botmode,opponentUsername,training,agentTraining,numBattles))
        else:
            asyncio.run(showdown(q,q50,qm50,q150,username,password,teamname,botmode,opponentUsername,training,agentTraining,numBattles))
        return