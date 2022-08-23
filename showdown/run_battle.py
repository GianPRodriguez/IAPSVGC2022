from multiprocessing import Process, Queue
import importlib
import os
import json
import io
import concurrent.futures
from copy import deepcopy
import ast
import sys
import constants
import config
from showdown.battle_bots.helpers import format_decision
from showdown.battle_modifier import update_battle
from showdown.battle_modifier import updateEnvironment
from showdown.websocket_client import PSWebsocketClient

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
    # ¿Hay solicitud de JSON pendiente?
    if battle.request_json:
        battle.user.updatePkmnUser(battle.request_json, first_turn=False)
        # Actualizo las variables del entorno...
        battle = await updateEnvironment(battle)
        # Los imprimo...
        battle.imprimir()
    # Actualiza las estadísticas de los Pkmn del usuario y oponente
    #await battle.updateStatistics()
    # Se define la mejor acción
    battle, moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2 = await battle.action_battle(battle)
    mensaje = await format_decision(battle, moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2)
    # Imprimo en CMD la acción
    print("[WARNING]     {}".format(mensaje))
    # Retorno el mensaje listo para enviarlo al WebSocket
    return battle, mensaje


async def escoge_accion_team_preview(battle, ps_websocket_client):
    # Se escoge los 4 Pokémon: eleccion(n) es un entero
    print("[INFO]     Vista Previa del Equipo")
    eleccion0, eleccion1, eleccion2, eleccion3 = await battle.action_team_preview(battle)    
    # Se prepara el String que se va a enviar al WebSocket
    # >> battle-gen8vgc2022-1614957698-shk265rg7lhkyzn0tyjofy9lr7myd0gpw|/team 314256|2
    mensaje = ["/team {}{}{}{}|{}".format(eleccion0, eleccion1, eleccion2, eleccion3, battle.rqid)]
    # Imprimo en CMD la acción hecha por el agente
    print("[WARNING]     {}".format(mensaje))
    # Se envía la acción al WebSocket
    if battle.botMode == constants.CHALLENGE_USER or battle.botMode == constants.ACCEPT_CHALLENGE:
        # Remuevo los 4 Pokémon seleccionados, solo quedan los no seleccionados
        battle.user.active1 = battle.user.reserved[eleccion0-1]
        battle.user.active2 = battle.user.reserved[eleccion1-1]
        aux = []
        aux.append(battle.user.reserved[eleccion2-1])
        aux.append(battle.user.reserved[eleccion3-1])
        battle.user.reserved = deepcopy(aux)
        await ps_websocket_client.send_message(battle.battle_tag, mensaje)
    return battle

async def get_battle_tag_and_opponent(ps_websocket_client: PSWebsocketClient, q):
    # Obtiene las etiquetas que aparecen al iniciar una batalla
    while True:
        # El string recibido lo guardo en msg
        msg = await ps_websocket_client.receive_message()
        # msg = ">battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw|init|battle|title|PopplioPeru98 vs. Braindeadmeta|j|☆PopplioPeru98"
        split_msg = msg.split('|')
        first_msg = split_msg[0]
        if len(split_msg) == 5:
            if split_msg[4].startswith("/nonotify"):    # Rechazo
                q.put([-2])
                print("[ERROR]     El oponente ha rechazado la solicitud de batalla")
                sys.exit()
        if msg.startswith("|popup|The user"):
            q.put([-4])
            print("[ERROR]     ¡El nombre de usuario no ha sido encontrado!")
            sys.exit()
        if msg.startswith("|popup|Your team"):
            q.put([-3])
            print("[ERROR]     ¡El equipo del agente inteligente no cumple con las reglas del formato!")
            sys.exit()
        # first_msg = ">battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw"
        if 'battle' in first_msg:
            battle_tag = first_msg.replace('>', '').strip()
            # battle_tag = "battle-gen8vgc2022-1614956049-sefnavzupbfhf6qq7ge3ckmkpoqyilkpw"
            user_name = split_msg[-1].replace('☆', '').strip()
            # user_name = "PopplioPeru98"
            opponent_name = split_msg[4].replace(user_name, '').replace('vs.', '').strip()
            # opponent_name = "Braindeadmeta"
            return battle_tag, opponent_name


async def initialize_battle_with_tag(ps_websocket_client: PSWebsocketClient, q, set_request_json=True):
    battle_module = importlib.import_module('showdown.battle_bots.{}.main'.format(config.battle_bot_module))
    # Obtengo el id de la batalla y el nombre del oponente
    battle_tag, opponent_name = await get_battle_tag_and_opponent(ps_websocket_client, q)
    # Mientras...
    while True:
        # Recibo un mensaje del WebSocket
        msg = await ps_websocket_client.receive_message()
        split_msg = msg.split('|')
        # ">battle-gen8vgc2022-1615425859-wyhg27ady04ol40g13v0qhy5qf6031rpw|request|{"teamPreview":true,"maxChosenTeamSize":4,"side":{"name":"PopplioPeru98","id":"p1","pokemon":[{"ident":"p1: Incineroar","details":"Incineroar, L50, F","con"
        # Si la cadena de arriba...
        if split_msg[1].strip() == 'request' and split_msg[2].strip():
            # Se obtiene el JSON 
            user_json = json.loads(split_msg[2].strip('\''))
            # user_json es un diccionario derivado del JSON
            user_id = user_json[constants.SIDE][constants.ID]
            # user_id = "p1"
            opponent_id = constants.ID_LOOKUP[user_id]
            # opponent_id = "p2"
            # Aquí se hace el cambio de bot y llamada de BattleBot...
            battle = battle_module.BattleBot(battle_tag)
            battle.q = q
            battle.q.put([0])
            # Actualización
            battle.opponent.id = opponent_id
            # battle.opponent.id = "p2"
            battle.opponent.account_name = opponent_name
            # battle.opponent.account_name = "Braindeadmeta"
            if set_request_json:
                # El JSON se guarda en battle.request_json...
                battle.request_json = user_json
            # Retorna 
            return battle, opponent_id, user_json


# Inicia una Batalla VGC 2022
async def inicia_vgc2022(ps_websocket_client: PSWebsocketClient, q, botMode):
    # Se obtiene las variables de abajo
    battle, opponent_id, user_json = await initialize_battle_with_tag(ps_websocket_client, q, set_request_json=True)
    battle.botMode = botMode
    # Inicializa mensaje donde se cargará el TEAM PREVIEW
    msg = ''
    # Mientras no aparezca al mensaje de TEAM PREVIEW
    while constants.START_TEAM_PREVIEW not in msg:
        msg = await ps_websocket_client.receive_message()
    # ">battle-gen8vgc2022-1615320442-rkf1xxub9e146xgon0ekxyzr2hhlfxfpw|player|p2|Jisss|lucas|1112|teamsize|p1|6|teamsize|p2|6|gen|8|tier|[Gen 8] VGC 2022|rated||rule|Species Clause: Limit one of each Pokémon|rule|Item Clause: Limit one of each item|clearpoke|poke|p1|Incineroar, L50, M||poke|p1|Kyogre, L50||poke|p1|Tornadus, L50, M||poke|p1|Zacian-*, L50||poke|p1|Indeedee-F, L50, F||poke|p1|Alcremie, L50, F||poke|p2|Lapras, L50, M||poke|p2|Kyogre, L50||poke|p2|Zacian-*, L50||poke|p2|Tapu Fini, L50||poke|p2|Zapdos, L50||poke|p2|Mimikyu, L50, F||teampreview|4"
    preview_string_lines = msg.split(constants.START_TEAM_PREVIEW)[-1].split('\n')
    # preview_string_lines = "|poke|p1|Incineroar, L50, M||poke|p1|Kyogre, L50||poke|p1|Tornadus, L50, M||poke|p1|Zacian-*, L50||poke|p1|Indeedee-F, L50, F||poke|p1|Alcremie, L50, F||poke|p2|Lapras, L50, M||poke|p2|Kyogre, L50||poke|p2|Zacian-*, L50||poke|p2|Tapu Fini, L50||poke|p2|Zapdos, L50||poke|p2|Mimikyu, L50, F||teampreview|4"
    opponent_pokemon = []
    # Para cada linea...
    for line in preview_string_lines:
        if not line:
            # Pasa a ETAPA TEAM PREVIEW
            continue
        # line = "|poke|p1|Incineroar, L50, M|"
        split_line = line.split('|')
        # split_line[1] = "poke"
        # split_line[2] = "p1"
        # split_line[3] = "Incineroar, L50, M"
        if split_line[1] == constants.TEAM_PREVIEW_POKE and split_line[2].strip() == opponent_id:
            # Voy agrupando: ["Lapras, L50, M","Kyogre, L50","Zacian-*, L50","Tapu Fini, L50","Zapdos, L50","Mimikyu, L50, F"]
            opponent_pokemon.append(split_line[3])
    # ETAPA TEAM PREVIEW
    await battle.initialize_team_preview(user_json, opponent_pokemon)
    # Se escoge la mejor acción y lo envía al WebSocket
    battle = await escoge_accion_team_preview(battle, ps_websocket_client)
    # Retorno el estado actual de la batalla
    return battle


async def iniciar_batalla(ps_websocket_client, q, botMode):
    # Inicia la Batalla VGC 2022
    battle = await inicia_vgc2022(ps_websocket_client, q, botMode)
    # Envía el saludo
    await ps_websocket_client.send_message(battle.battle_tag, [config.greeting_message])
    # Inicia el tiempo
    if battle.botMode == constants.CHALLENGE_USER or battle.botMode == constants.ACCEPT_CHALLENGE:
        await ps_websocket_client.send_message(battle.battle_tag, ['/timer on'])
    # Retorna la Clase Batalla
    return battle


# Nueva Batalla Pokémon completo
async def pokemon_battle(ps_websocket_client, q, botMode):
    # Solo inicia la batalla
    battle = await iniciar_batalla(ps_websocket_client, q, botMode)
    battle.imprimir()
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
                q.put([-1,winner==battle.opponent.account_name,battle.user.account_name,battle.opponent.account_name])
            else:
                winner = None
            # Y lo imprimo
            print("[INFO]     Ganador {}".format(winner))
            # Se envía un mensaje de despedida al contricante
            await ps_websocket_client.send_message(battle.battle_tag, [config.battle_ending_message])
            # Es opcional guardar la batalla...
            await ps_websocket_client.leave_battle(battle.battle_tag, save_replay=config.save_replay)
            q = battle.q
            # Retorna al ganador
            return winner, q
        else:
            # No ha finalizado la batalla... actualizo las variables del entorno
            action_required, battle = await update_battle(battle, mensaje)
            # action_required es BOOLEAN
            if action_required and not battle.wait:
                # Escoge el mejor movimiento y lo envía al websocket
                battle, mensaje = await accion_batalla(battle)
                q = battle.q
                if battle.botMode == constants.CHALLENGE_USER or battle.botMode == constants.ACCEPT_CHALLENGE:
                    await ps_websocket_client.send_message(battle.battle_tag, mensaje)