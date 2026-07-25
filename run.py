from environs import Env
from threading import Thread
from teams import load_team
from showdown.run_battle import pokemon_battle
from showdown.websocket_client import PSWebsocketClient
import asyncio
import constants
import config
import queue

#######################################################################################################################
# Se asignan los valores de configuración a las variables globales del módulo config.py
#######################################################################################################################
def parse_configs(username, password, teamname, botmode, numBattles):
    env = Env()
    env.read_env()

    try:
        config.battle_bot_module = env("BATTLE_BOT", 'smart')
        config.save_replay = env.bool("SAVE_REPLAY", config.save_replay)
        config.use_relative_weights = env.bool("USE_RELATIVE_WEIGHTS", config.use_relative_weights)
        config.gambit_exe_path = env("GAMBIT_PATH", config.gambit_exe_path)
        config.search_depth = int(env("MAX_SEARCH_DEPTH", config.search_depth))
        config.dynamic_search_depth = env.bool("DYNAMIC_SEARCH_DEPTH", config.dynamic_search_depth)
        config.greeting_message = env("GREETING_MESSAGE", config.greeting_message)
        config.battle_ending_message = env("BATTLE_OVER_MESSAGE", config.battle_ending_message)
        config.websocket_uri = env("WEBSOCKET_URI", "sim.smogon.com:8000")
        config.pokemon_mode = env("POKEMON_MODE", constants.DEFAULT_MODE)
        config.room_name = env("ROOM_NAME", config.room_name)
    except (ValueError, KeyError) as e:
        print(f"[ERROR] Fallo al cargar configuración: {e}")
        raise

    # Aplicar parámetros del usuario (sobrescriben los del .env)
    config.username = username
    config.password = password
    config.user_to_challenge = username
    config.bot_mode = botmode
    config.team_name = 'gen8/vgc2022/' + teamname
    config.run_count = numBattles

#######################################################################################################################
# Función principal que maneja la lógica de conexión, inicio de sesión y ejecución de batallas en Pokémon Showdown
#######################################################################################################################
async def showdown(q, username, password, teamname, botmode, numBattles):
    # Validación de parámetros
    if not username or not password or not teamname:
        print("[ERROR] ¡Usuario o contraseña incorrecta!")
        q.put([-5])
        return
    
    # Configuración de la Aplicación
    try:
        parse_configs(username, password, teamname, botmode, numBattles)
    except (ValueError, KeyError) as e:
        print(f"[ERROR] ¡Error del sistema!: {e}")
        q.put([-6])
        return
    
    # Inicio de Sesión con Usuario y Contraseña en Pokémon Showdown
    try:
        ps_websocket_client = await PSWebsocketClient.create(config.username, config.password, config.websocket_uri)
    except (asyncio.TimeoutError, OSError, ConnectionError) as e:
        print("[ERROR] ¡Usuario o contraseña incorrecta!")
        q.put([-5])
        return
    
    # Se inicia sesión en Pokémon Showdown, si es -1 es porque hubo un problema con el usuario o contraseña
    if await ps_websocket_client.login(False) == -1:
        print("[ERROR] ¡Usuario o contraseña incorrecta!")
        q.put([-5])
        await ps_websocket_client.close()
        return
    
    # Contadores
    stats = {'battles_run': 0, 'wins': 0, 'losses': 0}
    
    # Team almacena el equipo que se va a usar en la batalla, donde tendré el apoyo del bot
    try:
        team = load_team(config.team_name)[0]
    except Exception as e:
        print(f"[ERROR] ¡El equipo no cumple con las reglas del formato!: '{config.team_name}': {e}")
        q.put([-3])
        await ps_websocket_client.close()
        return

    # Opciones de juego
    try:
        if config.bot_mode == constants.CHALLENGE_USER:
            await ps_websocket_client.challenge_user(config.user_to_challenge, config.pokemon_mode, team, False)
        elif config.bot_mode == constants.ACCEPT_CHALLENGE:
            await ps_websocket_client.accept_challenge(config.pokemon_mode, team, config.room_name, False)
        elif config.bot_mode == constants.SEARCH_LADDER:
            # Módulo de Análisis
            await ps_websocket_client.search_for_match(team)
        else:
            raise ValueError("Invalid Bot Mode")
    except Exception as e:
        print(f"[ERROR] Fallo al iniciar las opciones de juego: {e}")
        q.put([-5])
        await ps_websocket_client.close()
        return
    
    # Módulo de Análisis
    try:
        while True:
            
            # Nueva Batalla Pokémon completo, pokemon_battle se ejecuta una vez por batalla completa (BO3)
            try:
                winner = await pokemon_battle(ps_websocket_client, q, config.bot_mode)
            except Exception as e:
                print(f"[ERROR] Fallo durante la batalla: {e}")
                q.put([-5])
                break
            
            if winner == config.username:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
            print(f"[INFO]     Batallas: {stats['battles_run'] + 1}/{config.run_count} | Victorias: {stats['wins']} | Derrotas: {stats['losses']}")
            
            stats['battles_run'] += 1

            # En el Módulo de Análisis debe ser 1, acepta de inmediato el siguiente juego
            if stats['battles_run'] >= config.run_count:
                break
    finally:
        await ps_websocket_client.close()
        print(f"[INFO]     Resumen Final - Batallas: {stats['battles_run']} | Victorias: {stats['wins']} | Derrotas: {stats['losses']}")

#######################################################################################################################
# Usado SOLO en el Módulo 3
#######################################################################################################################
async def stop(q, username, password, teamname, botmode, numBattles):
    poller = asyncio.create_task(showdown(q, username, password, teamname, botmode, numBattles))
    while True:
        await asyncio.sleep(0.1)
        try:
            cola = q.get(0)
            if cola[0] == -1000:
                poller.cancel()
                try:
                    await poller
                except asyncio.CancelledError:
                    pass
                return
        except queue.Empty:
            pass

#######################################################################################################################
# Método Principal - Módulo de Análisis
#######################################################################################################################
class AsyncDownload(Thread):
    def __init__(self, q, username, password, teamname, botmode, numBattles):
        super().__init__()
        asyncio.run(
            showdown(
                q, username, password, teamname, botmode, numBattles
            )
        )