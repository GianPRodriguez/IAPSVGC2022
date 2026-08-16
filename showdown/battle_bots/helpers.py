import constants
import requests

from showdown.battle import Battle

############################################################################################################################################################
# OK - Helper para formatear una acción individual y eliminar espacios vacíos                                                                            ###
############################################################################################################################################################
def _helper_format_action(action_type: str, slot: int, mega: int, target: int) -> str:
    parts = [
        action_type,
        str(slot) if slot != 0 else "",
        "mega" if mega != 0 else "",
        str(target) if target != 0 else ""
    ]
    # Filtra los elementos vacíos "" y los une con un único espacio
    return " ".join(p for p in parts if p)


############################################################################################################################################################
# OK - Se formatea la acción para cada turno de la Batalla                                                                                               ###
############################################################################################################################################################
async def format_decision(move_switch_1: str, n_move_switch_1: int, mega_1: int, to_move_1: int, move_switch_2: str, n_move_switch_2: int, mega_2: int, to_move_2: int) -> list[str]:
    
    # Formateamos la acción para cada uno de los dos Pokémon activos
    action_1 = _helper_format_action(move_switch_1, n_move_switch_1, mega_1, to_move_1)
    action_2 = _helper_format_action(move_switch_2, n_move_switch_2, mega_2, to_move_2)
    
    # Creamos el mensaje final
    message = f"/choose {action_1}, {action_2}" # message = "/choose move 2 mega 1,move 1 2"
    return [message]


############################################################################################################################################################
# OK - Define Nombre Oficial del Movimiento                                                                                                              ###
############################################################################################################################################################
async def getMoveName(battle, position: int, nMoveSwitch: int) -> str:
    return battle.request_json.get('active', False)[position].get('moves', False)[nMoveSwitch].get('move', '')


############################################################################################################################################################
# OK - Define Color del Movimiento                                                                                                                       ###
############################################################################################################################################################
async def getColorMove(move: str) -> str:
    url = 'https://pokeapi.co/api/v2/move/'

    urlMove = url + move
    response = requests.get(urlMove)
    if response.status_code == 200:
        payload = response.json()
        move_type = payload.get('type', {}).get('name', '')
        return constants.COLOR_TYPE.get(move_type, '#FFFFFF')
    else:
        return '#FFFFFF'


############################################################################################################################################################
# OK - ¿A quién va dirigido el movimiento?                                                                                                               ###
############################################################################################################################################################
async def getToMovePokemon(battle: Battle, toMove: str) -> str:
    if toMove == 1:
        return battle.opponent.active1.name
    elif toMove == 2:
        return battle.opponent.active2.name
    elif toMove == -1:
        return battle.user.active2.name
    elif toMove == -2:
        return battle.user.active1.name
    else:
        # toMove == 0
        return None