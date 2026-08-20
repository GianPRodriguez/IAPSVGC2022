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
from showdown.battle_modifier import updateStatsOpponents
from showdown.websocket_client import PSWebsocketClient
from showdown.battle import Battle
from showdown.battle_bots.helpers import format_decision

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
async def handle_finished_battleBO3(battle: Battle, msg: str) -> str:
    print("\n" + "=" * 80)
    print("[INFO] ¡Batalla BO3 finalizada detectada!")
    print("=" * 80)

    winner = "Unknown"
    
    # Separamos el mensaje en líneas para buscar el comando '|win|'
    lines = msg.split('\n')
    for line in lines:
        if line.startswith('|win|'):
            parts = line.split('|')
            if len(parts) >= 3:
                winner = parts[2].strip()
                break

    print(f"[INFO] Ganador de la partida: {winner}")

    # Opcionalmente, registramos el resultado en el archivo de reporte si está activo
    try:
        with open('reporte.txt', 'a', encoding='utf-8') as f:
            print(f"\n[FIN DE PARTIDA BO3] Ganador: {winner} (Turno: {battle.turn})", file=f)
    except Exception as e:
        print(f"No se pudo escribir el resultado en el reporte.txt: {e}")

    print("=" * 80 + "\n")
    
    return winner


############################################################################################################################################################
### OK - Confirma la preparación (ready) para la siguiente batalla en un encuentro al mejor de 3 (BO3)                                                   ###
############################################################################################################################################################
async def confirm_next_battle(ps_websocket_client, battle_tag: str, battleBO3_tag: str) -> None:
    # Construimos el mensaje de confirmación siguiendo el formato de las salas de Best of 3 de Showdown
    # Formato esperado: <battle_tag>|/msgroom <battleBO3_tag>,/confirmready
    message_to_send = f"{battle_tag}|/msgroom {battleBO3_tag},/confirmready"
    
    print(f"[SEND] Confirmando siguiente batalla BO3: {message_to_send}")
    
    # Enviamos el mensaje a través del cliente WebSocket
    await ps_websocket_client.send_message(message_to_send)


############################################################################################################################################################
### OK - Retorna True si la batalla BO3 ha terminado                                                                                                     ###
############################################################################################################################################################
def battleBO3_is_finished(battle_tag: str, msg: str) -> bool:
    return (
        msg.startswith(">{}".format(battle_tag)) and
        (constants.WIN_STRING in msg) and constants.CHAT_STRING not in msg
    )


############################################################################################################################################################
# Ok - Retorna la acción en texto, en base al INPUT de la Red Neuronal y el agente usado, que luego es enviado inmediatamente al WebSocket               ###
############################################################################################################################################################
async def createBattleAction(battle: Battle, msg_request: str) -> str:
    # 1. Actualiza el estado de los pkmn del usuario en base al json recibido
    battle.user.updatePkmnUser(json.loads(msg_request.split('|request|')[1]), first_turn=False)

    # 2. Actualiza SOLO las estadísticas de los PKMN Oponentes Activos
    await updateStatsOpponents(battle)

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
    await update_battle_environment(battle, msg_t, msg_request)
    print("[handle_active_battle_turn] update_battle_environment: FINALIZADO")

    # 3. Si el servidor solicita esperar (p. ej. el oponente está realizando U-Turn / cambio)
    if battle.wait:
        print(f"Turno Tipo: wait")
        return
    
    # 4. Si el turno requiere una acción activa o un cambio forzado (forceSwitch)
    if battle.force_switch or battle.active:
        print("force_switch: {}".format(battle.force_switch))
        print("active: {}".format(battle.active))
        battle.error = False
        battle_turn_action = await createBattleAction(battle, msg_request)
        print("[INFO]  Turno: {}".format(battle.turn))
        await ps_websocket_client.send_message(battle.battle_tag, battle_turn_action)
        print("[SEND]  Acción Turno: {}".format(battle_turn_action))
        battle.turn = battle.turn + 1
        return

    # 5. Si no es ni wait, ni active, ni forceSwitch, lanzamos una excepción específica
    raise ValueError(f"Se ha recibido un tipo de |request| inválido o desconocido: {msg_request}")


############################################################################################################################################################
#  aneja los errores de acción devueltos por el servidor de Pokémon Showdown (|error|).                                                                  ###
############################################################################################################################################################
async def handle_action_error(ps_websocket_client: PSWebsocketClient, battle: Battle, error_message: str) -> None:
    # 1. Documentar y registrar exhaustivamente el fallo de la acción anterior
    print("\n" + "!" * 80)
    print(f"[ERROR DE ACCIÓN DETECTADO EN SHOWDOWN]")
    print(f"  > Mensaje del servidor: {error_message}")
    print(f"  > Entorno Batalla: {battle.imprimir()}")
    print("!" * 80 + "\n")
    battle.error = True

    # 2. Reevaluar y recalcular una nueva acción utilizando la lógica del agente
    print("[handle_action_error] Solicitando una nueva decisión de emergencia al agente...")

    # 3. Se genera el INPUT de la Red Neuronal
    listNeuralNetworkInput = NeuralNetworkInput(battle)
    
    # 4. Invocamos la lógica de decisión del bot (similar a handle_active_battle_turn)
    moveSwitch1, nMoveSwitch1, megaEvo1, toMove1, moveSwitch2, nMoveSwitch2, megaEvo2, toMove2, battle.reward, battle.score = await battle.action_battle(battle, listNeuralNetworkInput)

    # 5. Formatear la nueva decisión corregida para enviarla al servidor
    battle_turn_action = await format_decision(moveSwitch1, nMoveSwitch1, megaEvo1, toMove1, moveSwitch2, nMoveSwitch2, megaEvo2, toMove2)

    # 6. Enviar la nueva acción correctiva a través del WebSocket
    print("[INFO]  Turno ERROR: {}".format(battle.turn))
    await ps_websocket_client.send_message(battle.battle_tag, battle_turn_action)
    print("[SEND]  Acción Turno ERROR: {}".format(battle_turn_action))


############################################################################################################################################################
# OK - Dado el mensaje, se imprime el ganador de la Batalla Individual                                                                                   ###
############################################################################################################################################################
async def handle_finished_battle(ps_websocket_client: PSWebsocketClient, battle: Battle, q: multiprocessing.Queue, mensaje: str) -> None:

    if constants.WIN_STRING in mensaje:  # Si hay un ganador en la Batalla Individual...
        winner = mensaje.split(constants.WIN_STRING)[-1].split('\n')[0].strip(" ")  # Se almacena el nombre del ganador
    else:
        raise ValueError("¡La batalla terminó sin un ganador válido") # Sino, se lanza excepción

    q.put([-1, winner == battle.opponent.account_name, battle.user.account_name, battle.opponent.account_name])  # Envío el ganador a la Interfaz e indico que ha finalizado la Batalla Individual
    print("[INFO]  Batalla Individual: {} - Ganador: {}".format(battle.battleBO3_count, winner))
    battle.battleBO3_count = battle.battleBO3_count + 1

    await ps_websocket_client.send_message(battle.battle_tag, [config.battle_ending_message])  # Es opcional enviar un mensaje de despedida al oponente...
    await ps_websocket_client.leave_battle(battle.battle_tag, save_replay=config.save_replay)  # Es opcional guardar la batalla...


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

    print("[INFO]  Acción Vista Previa del Equipo") # Se imprime en CMD la info de la Fase

    # Se escoge los 4 Pokémon: action_team_preview = tupla (5, 2, 1, 4)
    action_team_preview = await battle.action_team_preview(battle)
    # Se formatea en String; action_team_preview_str = "5214"
    action_team_preview_str = "".join(map(str, action_team_preview))
    # Se prepara el String que se va a enviar al WebSocket: mensaje = "/team 314256|2"
    mensaje = ["/team {}|{}".format(action_team_preview_str, battle.rqid)]
    
    # Se envía la acción al WebSocket y regreso al método que llamó a esta función, que es start_team_preview_phase
    await ps_websocket_client.send_message(battle.battle_tag, mensaje)
    print("[SEND]  Acción Vista Previa del Equipo: {}".format(mensaje))


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
# OK - Obtiene el battle_tag de cada una de las Batallas Individuales                                                                                    ###
############################################################################################################################################################
async def get_battle_tag(ps_websocket_client: PSWebsocketClient) -> str:
    while True:
        # Voy recibiendo mensajes del WebSocket hasta que aparezca el mensaje que contiene init
        msg = await ps_websocket_client.receive_message()
        # msg = ">battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw|init|battle|title|PopplioPeru98 vs. Braindeadmeta|j|☆PopplioPeru98"

        if not isinstance(msg, str) or not msg.strip():
            raise ValueError("Se recibió un mensaje vacío desde el WebSocket")

        if INIT in msg:
            parts = msg.split('|')
            first_msg = parts[0].strip()  # Ej: ">battle-gen8vgc2022-1614956049-..."
            
            if first_msg.startswith('>battle'):
                battle_tag = first_msg.replace('>', '').strip()
                return battle_tag


############################################################################################################################################################
# OK - Inicia una Nueva Batalla Individual Pokémon                                                                                                       ###
############################################################################################################################################################
async def start_battle(ps_websocket_client: PSWebsocketClient, battle: Battle) -> None:
    # Obtengo el battle_tag de la Batalla Individual
    battle.battle_tag = await get_battle_tag(ps_websocket_client)
    print("[start_battle] battle_tag: {}".format(battle.battle_tag))

    team_preview_message = ""

    # Entra en el bucle hasta que reciba el mensaje que contiene la vista previa del equipo del oponente
    while constants.START_TEAM_PREVIEW not in team_preview_message:
        team_preview_message = await ps_websocket_client.receive_message()

        if not isinstance(team_preview_message, str) or not team_preview_message.strip():
            raise ValueError("Se recibió un mensaje vacío desde el WebSocket")

    # Ahora toca los datos de mi equipo para la fase de vista previa del equipo
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
            break

    opponent_team = get_opponent_team(team_preview_message, battle.opponent.id) # ["Swampert","Swampertite","Torrent",["WaveCrash", "Surf","BulkUp","FlipTurn"],"Brave"] x6

    # Se completa las clases Battler con los datos de los 6 PKMN usuario y oponente, se ejecuta inicialmente 1 vez por Batalla Individual
    await battle.initialize_team_preview(user_json, opponent_team)

    # Envía el saludo para la Batalla Individual
    await ps_websocket_client.send_message(battle.battle_tag, [config.greeting_message])

    # Inicia el contador de tiempo para la Batalla Individual
    await ps_websocket_client.send_message(battle.battle_tag, ['/timer on'])


############################################################################################################################################################
# OK - Inicia la Batalla BO3, deja todo listo para que inicie la Batalla Individual, retorno la Clase Batalla con el bo3_tag                             ###
############################################################################################################################################################
async def start_battleBO3(ps_websocket_client: PSWebsocketClient, botMode: str) -> Battle:
    # Creo la Clase Batalla, que estará con vida durante el BO3, y que contiene toda la información de la Batalla, tanto del usuario como del oponente
    module_bot = importlib.import_module('showdown.battle_bots.{}.main'.format("random"))

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
        # Nombre de usuario no encontrado
        if message.startswith("|popup|The user"):
            raise Exception("¡El nombre de usuario no ha sido encontrado!")
        
        if 'battle' in first_part: # Happy Path
            battle_tag = first_part.replace('>', '').strip()  # battle_tag = "battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw"
            print("[start_battleBO3] Battle Tag BO3: {}".format(battle_tag))
            # Asigno el battle_tag de la Batalla BO3
            battle.battleBO3_tag = battle_tag
            user_name = parts[-1].replace('☆', '').strip()  # user_name = "PopplioPeru98"
            opponent_name = parts[4].replace(user_name, '').replace('vs.', '').strip()  # opponent_name = "Braindeadmeta"
            battle.opponent.account_name = opponent_name
            battle.user.account_name = user_name
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
        q.put([-6])
        raise
    
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
                elif "|-message|" in msg and "|win|" in msg:
                    print(f"[MSG]  ¡La Batalla no ha finalizado completamente!")
                    break
                else:
                    continue
        
        if not (await battleBO3_is_finished(battle.battleBO3_tag, msg)): # ¿Ha finalizado la Batalla BO3?
            await confirm_next_battle(ps_websocket_client, battle.battle_tag, battle.battleBO3_tag) # Si no ha finalizado, entonces se espera la siguiente Batalla Individual | confirmReady
        else:
            winner = await handle_finished_battleBO3(battle, msg) # Sí ha finalizado la Batalla BO3, entonces se obtiene el ganador y se sale del bucle
            print("[INFO] Ganador de la Batalla BO3: {}".format(winner))
            return winner