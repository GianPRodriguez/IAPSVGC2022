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
# OK - Se asignan los valores de configuración a las variables globales del módulo config.py
#######################################################################################################################
def parse_configs(username, password, teamname, botmode, numBattles, pokemon_mode=None, battle_bot=None, only_from=None):
    env = Env()
    env.read_env()

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
    config.team_dir = env("TEAM_DIR", config.team_dir).strip().strip("/")
    config.battle_bot = env("BATTLE_BOT", config.battle_bot).strip().lower() or "random"

    # Overrides desde la UI (si vienen)
    if pokemon_mode:
        config.pokemon_mode = str(pokemon_mode).strip()
    if battle_bot:
        config.battle_bot = str(battle_bot).strip().lower() or config.battle_bot

    config.bot_mode = botmode
    config.team_name = f"{config.team_dir}/{teamname}"
    config.run_count = numBattles
    config.user_to_challenge = None

    if botmode == constants.CHALLENGE_USER:
        # Retar a un oponente: credenciales del bot desde .env; username UI = oponente
        config.user_to_challenge = username
        config.username = env("PS_USERNAME")
        config.password = env("PS_PASSWORD")
    elif botmode == constants.ACCEPT_CHALLENGE:
        # Esperar desafíos: credenciales del bot desde la UI (prefills .env)
        config.username = username
        config.password = password
        filter_user = (only_from or env("USER_TO_ACCEPT", "") or "").strip()
        config.user_to_challenge = filter_user or None
    else:
        # SEARCH_LADDER u otros: credenciales desde la UI
        config.username = username
        config.password = password

#######################################################################################################################
# OK - Función principal que maneja la lógica de conexión, inicio de sesión y ejecución de batallas en Pokémon Showdown
#######################################################################################################################
async def showdown(q, username, password, teamname, botmode, numBattles, pokemon_mode=None, battle_bot=None, only_from=None):
    # Validación de parámetros
    if botmode == constants.SEARCH_LADDER:
        if not username or not password or not teamname:
            print("[ERROR] ¡Usuario, contraseña y teamname son obligatorios para SEARCH_LADDER!")
            q.put([-5])
            return
    elif botmode == constants.CHALLENGE_USER:
        if not username or not teamname:
            print("[ERROR] ¡Usuario y teamname son obligatorios para CHALLENGE_USER!")
            q.put([-5])
            return
    elif botmode == constants.ACCEPT_CHALLENGE:
        if not username or not password or not teamname:
            print("[ERROR] ¡Usuario, contraseña y teamname son obligatorios para ACCEPT_CHALLENGE!")
            q.put([-5])
            return
    else:
        if not username or not teamname:
            print("[ERROR] ¡Configuración incompleta: se requiere al menos usuario y teamname!")
            q.put([-5])
            return

    parse_configs(
        username, password, teamname, botmode, numBattles,
        pokemon_mode=pokemon_mode,
        battle_bot=battle_bot,
        only_from=only_from,
    )

    ps_websocket_client = None
    try:
        ps_websocket_client = await PSWebsocketClient.create(config.username, config.password, config.websocket_uri)
        await ps_websocket_client.login(False)
    except Exception as e:
        print(f"[ERROR]  Fallo al Iniciar Sesión en Pokémon Showdown: {e}")
        q.put([-5])
        if ps_websocket_client is not None:
            await ps_websocket_client.close()
        return

    stats = {'battles_run': 0, 'wins': 0, 'losses': 0}

    try:
        team = load_team(config.team_name)[0]
    except Exception as e:
        print(f"[ERROR]  ¡Ha sucedido un error al momento de cargar el equipo!: '{config.team_name}': {e}")
        q.put([-6])
        await ps_websocket_client.close()
        return

    if not team:
        print(
            f"[ERROR]  Equipo vacío o no empaquetable: '{config.team_name}'. "
            "Revisa el export Showdown (6 Pokémon separados por línea en blanco)."
        )
        q.put([-3])
        await ps_websocket_client.close()
        return

    try:
        # Primer matchmaking (ACCEPT se hace dentro del bucle para poder re-esperar)
        if config.bot_mode == constants.CHALLENGE_USER:
            await ps_websocket_client.challenge_user(config.user_to_challenge, config.pokemon_mode, team, False)
        elif config.bot_mode == constants.SEARCH_LADDER:
            await ps_websocket_client.search_for_match(team)
        elif config.bot_mode == constants.ACCEPT_CHALLENGE:
            pass
        else:
            raise ValueError("Invalid Bot Mode")

        while True:
            if config.bot_mode == constants.ACCEPT_CHALLENGE:
                await ps_websocket_client.accept_challenge(
                    config.pokemon_mode,
                    team,
                    config.room_name,
                    False,
                    q,
                    only_from=config.user_to_challenge,
                )

            try:
                winner = await pokemon_battle(ps_websocket_client, q, config.bot_mode)
            except Exception as e:
                if config.bot_mode == constants.SEARCH_LADDER:
                    print(f"[ERROR]  ¡Ha sucedido un error durante el BO3 de la Batalla!: {e}")
                    q.put([-6])
                    break
                print(f"[ERROR]  ¡Ha sucedido un error durante la Batalla!: {e}")
                q.put([-6])
                break

            if winner is None:
                print("[INFO]  Batalla sin resultado válido")
            elif winner == config.username:
                stats['wins'] += 1
                stats['battles_run'] += 1
            else:
                stats['losses'] += 1
                stats['battles_run'] += 1

            print(f"[INFO]     Avance: {100 * stats['battles_run']/config.run_count:.2f} % | Victorias: {stats['wins']} | Derrotas: {stats['losses']}")

            if stats['battles_run'] >= config.run_count:
                break

            # Re-matchmaking entre batallas (si run_count > 1)
            if config.bot_mode == constants.CHALLENGE_USER:
                await ps_websocket_client.challenge_user(config.user_to_challenge, config.pokemon_mode, team, False)
            elif config.bot_mode == constants.SEARCH_LADDER:
                await ps_websocket_client.search_for_match(team)

    finally:
        await ps_websocket_client.close()
        if config.bot_mode == constants.ACCEPT_CHALLENGE:
            print(f"[INFO]  Resumen Final - Batallas: {stats['battles_run']} | Victorias: {stats['wins']} | Derrotas: {stats['losses']}")

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
# OK - Hilo que ejecuta showdown() sin bloquear el __init__ (usar .start())
#######################################################################################################################
class AsyncDownload(Thread):
    def __init__(
        self,
        q,
        username,
        password,
        teamname,
        botmode,
        numBattles,
        pokemon_mode=None,
        battle_bot=None,
        only_from=None,
    ):
        super().__init__(daemon=True)
        self.q = q
        self.username = username
        self.password = password
        self.teamname = teamname
        self.botmode = botmode
        self.numBattles = numBattles
        self.pokemon_mode = pokemon_mode
        self.battle_bot = battle_bot
        self.only_from = only_from

    def run(self):
        asyncio.run(
            showdown(
                self.q,
                self.username,
                self.password,
                self.teamname,
                self.botmode,
                self.numBattles,
                pokemon_mode=self.pokemon_mode,
                battle_bot=self.battle_bot,
                only_from=self.only_from,
            )
        )
