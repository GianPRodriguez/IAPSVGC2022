import multiprocessing
from multiprocessing.pool import INIT
import os
import io
import ast
import sys
import math
import json
import time
import queue
import config
import importlib
import constants
import traceback
import concurrent.futures

from os import path
from copy import deepcopy
from showdown.battle_modifier import update_battle_environment
from showdown.websocket_client import PSWebsocketClient
from showdown.battle import Battle
from showdown.battle_bots.helpers import format_decision
from showdown.showdown_log import mark_showdown_battle, clear_battle_debug_logs


def _apply_player_lines(battle: Battle, message: str) -> None:
    """Parsea |player|pN|nick|... en un mensaje multiparte (init / teampreview)."""
    if not message or '|player|' not in message:
        return
    for line in message.split('\n'):
        if '|player|' not in line:
            continue
        parts = line.split('|')
        try:
            idx = next(i for i, p in enumerate(parts) if p.strip() == 'player')
        except StopIteration:
            continue
        if idx + 2 >= len(parts):
            continue
        player_id = parts[idx + 1].strip()
        account = parts[idx + 2].strip()
        if not account:
            continue
        user_id = getattr(battle.user, 'id', None)
        opp_id = getattr(battle.opponent, 'id', None)
        if user_id and player_id == user_id:
            battle.user.account_name = account
        elif opp_id and player_id == opp_id:
            battle.opponent.account_name = account
        elif not user_id and player_id.endswith('1'):
            battle.user.account_name = account
        elif not opp_id and player_id.endswith('2'):
            battle.opponent.account_name = account


def _parse_title_names(parts: list):
    """Extrae nombres desde |title|A vs. B|... (formatos BO3 / Champions)."""
    title_idx = next((i for i, p in enumerate(parts) if p.strip() == 'title'), None)
    if title_idx is not None and title_idx + 1 < len(parts):
        title = parts[title_idx + 1]
        for sep in (' vs. ', ' vs ', ' VS. ', ' VS '):
            if sep in title:
                left, right = title.split(sep, 1)
                return left.strip(), right.strip()

    starred = parts[-1].replace('☆', '').strip() if parts else ''
    if starred and len(parts) > 4:
        other = parts[4].replace(starred, '').replace('vs.', '').replace('vs', '').strip()
        return starred, other
    return None, None


############################################################################################################################################################
# Por Implementar - Entrada Red Neuronal                                                                                                                 ###
############################################################################################################################################################
def NeuralNetworkInput(battle: Battle) -> list:
    listNeuralNetworkInput = []
    if battle:
        listNeuralNetworkInput.append(1.0)
    return listNeuralNetworkInput


############################################################################################################################################################
### OK - Obtiene el nombre del ganador de la batalla BO3                                                                                                 ###
############################################################################################################################################################
async def handle_finished_battleBO3(battle: Battle, msg: str, q: multiprocessing.Queue) -> str:
    winner = "Unknown"
    
    # Separamos el mensaje en líneas para buscar el comando '|win|'
    lines = msg.split('\n')
    for line in lines:
        if line.startswith('|win|'):
            parts = line.split('|')
            if len(parts) >= 3:
                winner = parts[2].strip()
                break

    print(f"[INFO]     Ganador de la partida BO3: {winner}")

    q.put([-1, winner == battle.user.account_name, battle.user.account_name, battle.opponent.account_name])
    return winner


############################################################################################################################################################
### OK - Confirma la preparación (ready) para la siguiente batalla en un encuentro al mejor de 3 (BO3)                                                   ###
############################################################################################################################################################
async def confirm_next_battle(ps_websocket_client, battle_tag: str, battleBO3_tag: str) -> None:
    # Preparamos la lista con el comando exacto requerido por Showdown para el chat de la sala BO3
    # Esto formará internamente: /msgroom <battleBO3_tag>,/confirmready
    message_list = [f"/msgroom {battleBO3_tag},/confirmready"]
    
    print(f"[SEND]     Iniciando siguiente batalla...")
    
    # Enviamos utilizando la firma correcta de send_message(room, message_list)
    # room = battle_tag
    # message_list = ["/msgroom .../confirmready"]
    await ps_websocket_client.send_message(battle_tag, message_list)


############################################################################################################################################################
### OK - Retorna True si la batalla BO3 ha terminado                                                                                                     ###
############################################################################################################################################################
async def battleBO3_is_finished(ps_websocket_client, battle_tag: str) -> bool:
    while True:
        msg = await ps_websocket_client.receive_message()
        if "Are you ready for game" in msg:
            # NEXT BATTLE
            return False
        elif msg.startswith(">{}".format(battle_tag)) and (constants.WIN_STRING in msg) and constants.CHAT_STRING not in msg:
            return True
        else:
            continue


############################################################################################################################################################
# Ok - Retorna la acción en texto, en base al INPUT de la Red Neuronal y el agente usado, que luego es enviado inmediatamente al WebSocket               ###
############################################################################################################################################################
async def createBattleAction(battle: Battle, msg_request: str) -> str:
    # 1. Actualiza el estado de los pkmn del usuario en base al json recibido
    first_turn = False
    battle.user.updatePkmnUser(json.loads(msg_request.split('|request|')[1]), first_turn, None)
    # 2. Actualiza SOLO las estadísticas de los PKMN Oponentes Activos
    # 3. Se genera el INPUT de la Red Neuronal
    listNeuralNetworkInput = NeuralNetworkInput(battle)
    # 4. Se define la mejor acción en base al listNeuralNetworkInput
    moveSwitch1, nMoveSwitch1, megaEvo1, toMove1, moveSwitch2, nMoveSwitch2, megaEvo2, toMove2, battle.reward, battle.score = await battle.action_battle(battle, listNeuralNetworkInput)
    # 5. Formateo la acción seleccionada para el Turno
    actionToWebSocket = await format_decision(moveSwitch1, nMoveSwitch1, megaEvo1, toMove1, moveSwitch2, nMoveSwitch2, megaEvo2, toMove2)
    # 6. Retorno el mensaje listo para enviarlo al WebSocket
    return actionToWebSocket


############################################################################################################################################################
### OK - Método que se encarga de manejar los mensajes de actualización de entorno y tipo de solicitud de acción, retorna la acción a realizar           ###
############################################################################################################################################################
async def handle_active_battle_turn(ps_websocket_client: PSWebsocketClient, battle: Battle, msg_t: str) -> None: 
    # 1. Recibir el mensaje del WebSocket que contiene el |request|
    msg_request = await ps_websocket_client.receive_message()
    # 2. Actualizar todas las variables del entorno (Pokémon activos, stats, estados, etc.) en base a msg_t
    try:
        await update_battle_environment(battle, msg_t, msg_request)
    except Exception as e:
        # Obtener el marco (frame) exacto del código donde ocurrió la excepción interna
        tb = e.__traceback__
        last_frame = traceback.extract_tb(tb)[-1]  # El marco más profundo (donde realmente falló)
        
        filename = last_frame.filename
        lineno = last_frame.lineno
        func_name = last_frame.name
        code_line = last_frame.line

        # Recorte limpio de variables para no saturar la consola
        msg_t_preview = str(msg_t)[:100] if msg_t else "N/A"
        msg_req_preview = str(msg_request)[:100] if msg_request else "N/A"

        print("\n" + "=" * 80)
        print(f"[ERROR DETALLADO EN EL ENTORNO DE BATALLA]")
        print(f"  > Módulo/Archivo : {filename}")
        print(f"  > Función        : {func_name}()")
        print(f"  > Número de Línea: {lineno}")
        print(f"  > Código Faltante: {code_line}")
        print(f"  > Mensaje Error  : {type(e).__name__}: {e}")
        print("-" * 80)
        print(f"  > ESTADO DE VARIABLES:")
        print(f"      Turno actual : {getattr(battle, 'turn', 'N/A')}")
        print(f"      rqid         : {getattr(battle, 'rqid', 'N/A')}")
        print(f"      msg_t        : {msg_t_preview}...")
        print(f"      msg_request  : {msg_req_preview}...")
        print("=" * 80 + "\n")

        # Re-lanzamos la excepción encadenada
        raise ValueError(f"Fallo en {filename} -> {func_name}() [Línea {lineno}]: {e}") from e

    # Spec C2: delta de material tras el update (antes de decidir / dump).
    try:
        from showdown.inference_sim.reward_telemetry import (
            observe_iap_spec_reward,
            push_iap_reward_to_ui,
        )
        c2_delta, _c2_ep = observe_iap_spec_reward(battle)
        # En |wait| no hay action_battle → empujar UI aquí si hubo delta.
        if battle.wait and abs(float(c2_delta)) > 1e-9:
            push_iap_reward_to_ui(
                battle, getattr(battle, 'q', None), label='(wait)',
            )
    except Exception as exc:
        print(f'[WARN]     observe_iap_spec_reward: {exc}')

    # 3. Si el servidor solicita esperar (p. ej. el oponente está realizando U-Turn / cambio)
    # Igual volcamos snap [WAIT] para paridad rqid vs logs_showdown.txt
    if battle.wait:
        battle.imprimir(to_console=True, accion='(wait)')
        return

    # 4. Si el turno requiere una acción activa o un cambio forzado (forceSwitch)
    if battle.force_switch or battle.active:
        battle.error = False
        try:
            battle_turn_action = await createBattleAction(battle, msg_request)
        except Exception as e:
            # Obtener el marco (frame) exacto del código donde ocurrió la excepción interna
            tb = e.__traceback__
            last_frame = traceback.extract_tb(tb)[-1]  # El marco más profundo donde realmente falló
            
            filename = last_frame.filename
            lineno = last_frame.lineno
            func_name = last_frame.name
            code_line = last_frame.line

            # Recorte limpio de variables para el log
            msg_req_preview = str(msg_request)[:100] if msg_request else "N/A"

            print("\n" + "=" * 80)
            print(f"[ERROR DETALLADO EN createBattleAction]")
            print(f"  > Módulo/Archivo : {filename}")
            print(f"  > Función        : {func_name}()")
            print(f"  > Número de Línea: {lineno}")
            print(f"  > Código Fallido : {code_line}")
            print(f"  > Mensaje Error  : {type(e).__name__}: {e}")
            print("-" * 80)
            print(f"  > ESTADO DE LA BATALLA:")
            print(f"      Turno actual : {getattr(battle, 'turn', 'N/A')}")
            print(f"      rqid         : {getattr(battle, 'rqid', 'N/A')}")
            print(f"      force_switch : {getattr(battle, 'force_switch', 'N/A')}")
            print(f"      msg_request  : {msg_req_preview}...")
            print("=" * 80 + "\n")

            # Re-lanzamos la excepción encadenada para conservar la causa raíz original
            raise ValueError(f"Fallo en {filename} -> {func_name}() [Línea {lineno}]: {e}") from e

        # Reporte CMD (fuente principal turno a turno; ACCION dentro del bloque)
        battle.imprimir(to_console=True, accion=battle_turn_action)

        await ps_websocket_client.send_message(battle.battle_tag, battle_turn_action)
        # El contador de turno / clima / side conditions lo actualiza únicamente |turn|
        # del protocolo. No adelantar battle.turn aquí: si no, un force-switch mid-turn
        # muestra el turno siguiente con el clima aún sin decrementar.
        return

    # 5. Si no es ni wait, ni active, ni forceSwitch, lanzamos una excepción específica
    raise ValueError(f"Se ha recibido un tipo de |request| inválido o desconocido: {msg_request}")


############################################################################################################################################################
#  aneja los errores de acción devueltos por el servidor de Pokémon Showdown (|error|).                                                                  ###
############################################################################################################################################################
async def handle_action_error(ps_websocket_client: PSWebsocketClient, battle: Battle, error_message: str) -> None:
    # 1. Documentar y registrar exhaustivamente el fallo de la acción anterior
    print("\n" + "!" * 80)
    print(f"[ERROR DE ACCION SHOWDOWN] {error_message}")
    battle.error = True
    battle.imprimir(to_console=True)
    print("!" * 80 + "\n")

    # 2. Reevaluar y recalcular una nueva acción utilizando la lógica del agente
    listNeuralNetworkInput = NeuralNetworkInput(battle)
    
    # 4. Invocamos la lógica de decisión del bot (similar a handle_active_battle_turn)
    moveSwitch1, nMoveSwitch1, megaEvo1, toMove1, moveSwitch2, nMoveSwitch2, megaEvo2, toMove2, battle.reward, battle.score = await battle.action_battle(battle, listNeuralNetworkInput)

    # 5. Formatear la nueva decisión corregida para enviarla al servidor
    battle_turn_action = await format_decision(moveSwitch1, nMoveSwitch1, megaEvo1, toMove1, moveSwitch2, nMoveSwitch2, megaEvo2, toMove2)

    # 6. Enviar la nueva acción correctiva a través del WebSocket
    battle.imprimir(to_console=True, accion=f'(retry error) {battle_turn_action}')
    await ps_websocket_client.send_message(battle.battle_tag, battle_turn_action)


############################################################################################################################################################
# OK - Dado el mensaje, se imprime el ganador de la Batalla Individual                                                                                   ###
############################################################################################################################################################
async def handle_finished_battle(ps_websocket_client: PSWebsocketClient, battle: Battle, q: multiprocessing.Queue, mensaje: str) -> None:

    if constants.WIN_STRING in mensaje:  # Si hay un ganador en la Batalla Individual...
        winner = mensaje.split(constants.WIN_STRING)[-1].split('\n')[0].strip(" ")  # Se almacena el nombre del ganador
    else:
        raise ValueError("¡La batalla terminó sin un ganador válido") # Sino, se lanza excepción

    print("[INFO]     Batalla Individual: {} - Ganador: {}".format(battle.battleBO3_count, winner))
    # Spec C2/C1 terminal + telemetría UI (Streamlit / start.py)
    try:
        from showdown.inference_sim.reward_telemetry import finalize_iap_game_reward
        finalize_iap_game_reward(battle, winner, q)
    except Exception as exc:
        print(f'[WARN]     finalize_iap_game_reward: {exc}')
    battle.append_game_banner('END', winner=winner, to_console=True)
    battle.battleBO3_count = battle.battleBO3_count + 1

    await ps_websocket_client.send_message(battle.battle_tag, [config.battle_ending_message])  # Es opcional enviar un mensaje de despedida al oponente...
    #await ps_websocket_client.leave_battle(battle.battle_tag, save_replay=config.save_replay)  # Es opcional guardar la batalla...


############################################################################################################################################################
### OK - Retorna True si la batalla individual ha terminado completamente                                                                                ###
############################################################################################################################################################
def battle_is_finished(battle_tag: str, msg: str) -> bool:
    return (
        msg.startswith(">{}".format(battle_tag)) and
        (constants.WIN_STRING in msg) and constants.CHAT_STRING not in msg and "-message" not in msg
    )


############################################################################################################################################################
# OK - Escoge y Envía la acción para la vista previa del equipo                                                                                          ###
############################################################################################################################################################
async def send_action_team_preview(ps_websocket_client: PSWebsocketClient, battle: Battle) -> None:

    print("[INFO]     Team Preview")

    # Se escoge los 4 Pokémon: action_team_preview = tupla (5, 2, 1, 4)
    action_team_preview = await battle.action_team_preview(battle)
    # Se formatea en String; action_team_preview_str = "5214"
    action_team_preview_str = "".join(map(str, action_team_preview))
    # Se prepara el String que se va a enviar al WebSocket: mensaje = "/team 314256|2"
    mensaje = ["/team {}|{}".format(action_team_preview_str, battle.rqid)]

    battle.imprimir(to_console=True, accion=mensaje)

    await ps_websocket_client.send_message(battle.battle_tag, mensaje)


############################################################################################################################################################
# OK - Obtiene el equipo del Oponente en base a la vista previa del equipo                                                                               ###
############################################################################################################################################################
def get_opponent_team(preview_string_lines: str, opponent_id: str) -> list:
    equipo = []
    # El prefijo que estamos buscando en las líneas, por ejemplo: "|showteam|p2|"
    target_prefix = f"|showteam|{opponent_id}|"
    
    # Recorremos cada línea del string
    for linea in preview_string_lines.strip().split('\n'):
        if linea.startswith(target_prefix):
            # Removemos el prefijo para quedarnos solo con la cadena de los Pokémon
            team_data = linea[len(target_prefix):]
            
            # Cada Pokémon está separado por el carácter ']'
            pokemon_raw_list = team_data.split(']')
            
            for pkmn_str in pokemon_raw_list:
                if not pkmn_str.strip():
                    continue
                
                # Los atributos de cada Pokémon están separados por '|'
                atributos = pkmn_str.split('|')
                
                # Extraemos según el índice correspondiente en la cadena de Showdown
                nombre = atributos[0]
                objeto = atributos[2]
                habilidad = atributos[3]
                movimientos = atributos[4].split(',')
                naturaleza = atributos[5]
                
                # Formateamos el Pokémon y lo agregamos a la lista final
                pkmn_formateado = [nombre, objeto, habilidad, movimientos, naturaleza]
                equipo.append(pkmn_formateado)
            
            # Una vez que encontramos y procesamos la línea del equipo, podemos detener la búsqueda
            break
            
    return equipo


############################################################################################################################################################
# OK - Obtiene el equipo del Oponente en base a la vista previa del equipo                                                                               ###
############################################################################################################################################################
def get_natures(preview_string_lines: str, opponent_id: str) -> list:
    natures = []
    # El prefijo que estamos buscando en las líneas, por ejemplo: "|showteam|p2|"
    target_prefix = f"|showteam|{opponent_id}|"
    
    # Recorremos cada línea del string
    for linea in preview_string_lines.strip().split('\n'):
        if linea.startswith(target_prefix):
            # Removemos el prefijo para quedarnos solo con la cadena de los Pokémon
            team_data = linea[len(target_prefix):]

            # Cada Pokémon está separado por el carácter ']'
            pokemon_raw_list = team_data.split(']')
            
            for pkmn_str in pokemon_raw_list:
                if not pkmn_str.strip():
                    continue
                
                # Los atributos de cada Pokémon están separados por '|'
                atributos = pkmn_str.split('|')
                
                # Extraemos según el índice correspondiente en la cadena de Showdown
                nature = atributos[5]

                natures.append(nature)
            
            # Una vez que encontramos y procesamos la línea del equipo, podemos detener la búsqueda
            break

    return natures


############################################################################################################################################################
# OK - Obtiene el battle_tag de cada una de las Batallas Individuales                                                                                    ###
############################################################################################################################################################
async def get_battle_tag(ps_websocket_client: PSWebsocketClient) -> str:
    while True:
        # Voy recibiendo mensajes del WebSocket hasta que aparezca el mensaje que contiene init
        msg = await ps_websocket_client.receive_message()
        # msg = ">battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw|init|battle|title|PopplioPeru98 vs. Braindeadmeta|j|☆PopplioPeru98"

        if not isinstance(msg, str) or not msg.strip():
            raise ValueError("Se recibió un mensaje vacío desde el WebSocket")

        if constants.BO3 in msg:
            parts = msg.split('|')
            first_msg = parts[0].strip()  # Ej: ">battle-gen8vgc2022-1614956049-..."
            
            if first_msg.startswith('>battle'):
                battle_tag = first_msg.replace('>', '').strip()
                return battle_tag


############################################################################################################################################################
# OK - Inicia una Nueva Batalla Individual Pokémon                                                                                                       ###
############################################################################################################################################################
async def start_battle(ps_websocket_client: PSWebsocketClient, battle: Battle) -> None:
    # LIMPIAR
    battle.toClean()
    # Obtengo el battle_tag de la Batalla Individual
    battle.battle_tag = await get_battle_tag(ps_websocket_client)
    print("[TAG1]     {}".format(battle.battle_tag))

    team_preview_message = ""

    # Entra en el bucle hasta que reciba el mensaje que contiene la vista previa del equipo del oponente
    while constants.START_TEAM_PREVIEW not in team_preview_message:
        team_preview_message = await ps_websocket_client.receive_message()

        if not isinstance(team_preview_message, str) or not team_preview_message.strip():
            raise ValueError("Se recibió un mensaje vacío desde el WebSocket")
        _apply_player_lines(battle, team_preview_message)

    mark_showdown_battle(battle.battle_tag)
    while True:
        # Recibo un mensaje del WebSocket
        request_message = await ps_websocket_client.receive_message()

        if not isinstance(request_message, str) or not request_message.strip():
            raise ValueError("Se recibió un mensaje vacío desde el WebSocket")

        split_msg = request_message.split('|')

        # Si la cadena de arriba...
        if len(split_msg) > 2 and split_msg[1].strip() == 'request' and split_msg[2].strip():
            # Se obtiene el JSON TEAM PREVIEW de la batalla
            request_payload = split_msg[2].strip().strip("'")
            user_json = json.loads(request_payload)
            
            user_id = user_json[constants.SIDE][constants.ID]  # user_id = "p1"
            opponent_id = constants.ID_LOOKUP[user_id]  # opponent_id = "p2"

            # Actualización de variables de la Clase Batalla
            battle.opponent.id = opponent_id
            battle.request_json = user_json
            battle.user.id = user_id
            # Reasignar cuentas con IDs ya conocidos (corrige p1/p2 provisional)
            _apply_player_lines(battle, team_preview_message)
            if user_json.get(constants.SIDE, {}).get('name'):
                battle.user.account_name = user_json[constants.SIDE]['name']
            break

    opponent_team = get_opponent_team(team_preview_message, battle.opponent.id) # ["Swampert","Swampertite","Torrent",["WaveCrash", "Surf","BulkUp","FlipTurn"],"Brave"] x6
    natures = get_natures(team_preview_message, battle.user.id)

    # Se completa las clases Battler con los datos de los 6 PKMN usuario y oponente, se ejecuta inicialmente 1 vez por Batalla Individual
    await battle.initialize_team_preview(user_json, opponent_team, natures)

    # Envía el saludo para la Batalla Individual
    await ps_websocket_client.send_message(battle.battle_tag, [config.greeting_message])

    # Inicia el contador de tiempo para la Batalla Individual
    #await ps_websocket_client.send_message(battle.battle_tag, ['/timer on'])


############################################################################################################################################################
# OK - Inicia la Batalla BO3, deja todo listo para que inicie la Batalla Individual, retorno la Clase Batalla con el bo3_tag                             ###
############################################################################################################################################################
def _queue_battle_start_error(q: multiprocessing.Queue, error: Exception) -> None:
    """Mapea excepciones de inicio de BO3 a códigos de la UI (-2/-3/-4/-6)."""
    text = str(error).lower()
    # "Your team was rejected" / equipo vacío debe ir a -3, no a -2 (rechazo humano).
    if "equipo" in text or "team" in text or "formato" in text or "utm" in text:
        q.put([-3])
    elif "rechazado" in text or ("rejected" in text and "team" not in text):
        q.put([-2])
    elif "no ha sido encontrado" in text or "user not found" in text:
        q.put([-4])
    else:
        q.put([-6])


async def start_battleBO3(ps_websocket_client: PSWebsocketClient, botMode: str) -> Battle:
    # Nuevo BO3: vaciar logs del set anterior (no al finalizar, para poder revisarlos)
    try:
        clear_battle_debug_logs()
    except Exception as e:
        print(f"[WARN]     No se pudieron limpiar los logs al inicio del BO3: {e}")

    # Creo la Clase Batalla, que estará con vida durante el BO3, y que contiene toda la información de la Batalla, tanto del usuario como del oponente
    bot_name = (config.battle_bot or "random").strip().lower()
    try:
        module_bot = importlib.import_module('showdown.battle_bots.{}.main'.format(bot_name))
    except ModuleNotFoundError:
        print(f"[WARN]     BATTLE_BOT '{bot_name}' no existe; usando 'random'")
        bot_name = "random"
        module_bot = importlib.import_module('showdown.battle_bots.random.main')
    print(f"[INFO]     Battle bot activo: {bot_name}")

    # Instanciamos la clase BattleBot que vive dentro de ese módulo
    battle = module_bot.BattleBot() 

    battle.botMode = botMode

    # Inicia la Batalla VGC
    while True:
        # Voy recibiendo mensajes del WebSocket hasta que aparezca el mensaje que contiene init
        message = await ps_websocket_client.receive_message()
        # message = ">battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw|init|battle|title|PopplioPeru98 vs. Braindeadmeta|j|☆PopplioPeru98"

        if not isinstance(message, str) or not message.strip():
            raise ValueError("Se recibió un mensaje vacío desde el WebSocket")

        parts = message.split('|')
        first_part = parts[0]
        # first_part = ">battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw"

        # Rechazo de la solicitud de batalla
        if len(parts) >= 5 and parts[4].startswith("/nonotify"):
            raise Exception("¡El oponente ha rechazado la solicitud de batalla!")
        if message.startswith("|popup|") and ("rejected" in message.lower() or "rechaz" in message.lower()):
            raise Exception("¡El oponente ha rechazado la solicitud de batalla!")
        
        # Nombre de usuario no encontrado
        if message.startswith("|popup|The user") or (
            message.startswith("|popup|") and "could not be found" in message.lower()
        ):
            raise Exception("¡El nombre de usuario no ha sido encontrado!")

        # Equipo inválido / vacío (Showdown: "requires you to use your own team")
        if message.startswith("|popup|Your team") or (
            message.startswith("|popup|") and "own team" in message.lower()
        ):
            raise Exception("¡El equipo del agente no cumple con las reglas del formato!")

        if 'game-bestof3' in first_part: # Happy Path
            battle_tag = first_part.replace('>', '').strip()  # battle_tag = "battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw"
            print("[TAG0]     {}".format(battle_tag))
            # Asigno el battle_tag de la Batalla BO3
            battle.battleBO3_tag = battle_tag
            mark_showdown_battle(battle_tag)
            _apply_player_lines(battle, message)
            user_name, opponent_name = _parse_title_names(parts)
            # Solo rellenar huecos: |player| tiene prioridad sobre el título
            if not battle.user.account_name and user_name:
                battle.user.account_name = user_name
            if not battle.opponent.account_name and opponent_name:
                battle.opponent.account_name = opponent_name
            break

    # Retorna la Clase Batalla, que va a ser usada durante toda la Batalla BO3
    return battle


############################################################################################################################################################
# OK - Flujo Principal Batalla BO3, devuelve el ganador o genera excepción si hubo un error                                                              ###
############################################################################################################################################################
async def pokemon_battle(ps_websocket_client: PSWebsocketClient, q: multiprocessing.Queue, botmode: str) -> str:
    try:
        battle = await start_battleBO3(ps_websocket_client, botmode) # Crea la Clase Batalla BO3 | init (1)
    except Exception as e:
        print(f"[ERROR]  ¡Ha sucedido un error al momento de Comenzar la Batalla BO3!: {e}")
        _queue_battle_start_error(q, e)
        raise

    q.put([0])  # UI: batalla en curso
    battle.q = q  # Misma cola para Team Preview / turnos (Módulo de Análisis)

    while True: # Para cada uno de las Batallas Individuales dentro del BO3
        try:
            await start_battle(ps_websocket_client, battle) # Inicialización de la Batalla Individual | init (x), clearpoke, |request|{"teamPreview"...
            await send_action_team_preview(ps_websocket_client, battle) # Envío la acción para el Team Preview
        except Exception as e:
            print(f"[ERROR]  ¡Ha sucedido un error al momento de Iniciar la Batalla Individual!: {e}")
            q.put([-6])
            raise

        while True: # Para cada uno de los turnos de la Batalla Individual
            msg = await ps_websocket_client.receive_message() # Recibo Mensajes por cada turno de la Batalla Individual

            if battle_is_finished(battle.battle_tag, msg): # ¿Ha finalizado Completamente la Batalla Individual? |win|
                try: # Si ha finalizado...
                    await handle_finished_battle(ps_websocket_client, battle, q, msg) # Entonces se imprime el ganador de la Batalla Individual...
                except Exception as e:
                    print(f"[ERROR]  ¡Ha sucedido un error al momento de Finalizar la Batalla Individual!: {e}")
                    q.put([-6])
                    raise
                break # Y se sale del bucle
            else:
                if "|t:|" in msg:
                    await handle_active_battle_turn(ps_websocket_client, battle, msg) # Si no lo tiene, se maneja el nuevo turno confirmado
                elif "|error|" in msg:
                    await handle_action_error(ps_websocket_client, battle, msg) # Se enviará una nueva acción según el agente, pero se documentará totalmente por qué falló
                elif msg.startswith(">{}".format(battle.battleBO3_tag)) and (constants.WIN_STRING in msg) and constants.CHAT_STRING not in msg:
                    winner = await handle_finished_battleBO3(battle, msg, q) # Sí ha finalizado la Batalla BO3, entonces se obtiene el ganador y se sale del bucle
                    return winner
                else:
                    continue
        
        if not (await battleBO3_is_finished(ps_websocket_client, battle.battleBO3_tag)): # ¿Ha finalizado la Batalla BO3?
            await confirm_next_battle(ps_websocket_client, battle.battle_tag, battle.battleBO3_tag) # Si no ha finalizado, entonces se espera la siguiente Batalla Individual | confirmReady
        else:
            winner = await handle_finished_battleBO3(battle, msg, q) # Sí ha finalizado la Batalla BO3, entonces se obtiene el ganador y se sale del bucle
            return winner