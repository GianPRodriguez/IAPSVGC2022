from showdown.battle import Battle

############################################################################################################################################################
# OK - Helper para formatear una acción individual y eliminar espacios vacíos                                                                            ###
############################################################################################################################################################
def _helper_format_action(action_type: str, slot: int, mega: int, target: int) -> str:
    # mega: 0=none, 1=mega, 2=terastallize (legacy code path; Reg M-C disables 2)
    from showdown.format_rules import TERASTALLIZATION_ENABLED
    special = ""
    if mega == 1:
        special = "mega"
    elif mega == 2 and TERASTALLIZATION_ENABLED:
        special = "terastallize"
    parts = [
        action_type,
        str(slot) if slot != 0 else "",
        special,
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


def enforce_single_mega(first, second, score_first: float = 0.0, score_second: float = 0.0):
    """Como mucho UNA mega-evolución por orden doble.

    Showdown solo admite una por combate y rechaza ``move X mega, move Y mega``
    con ``[Invalid choice] Can't move: You can only mega-evolve once per battle``.
    poke-env reintenta entonces la misma orden (el bot es determinista) hasta que
    ``random.random() < 1/1000`` acepta la orden por defecto: miles de reintentos
    por turno. Se conserva la mega del hueco con la acción de más valor (empate:
    el izquierdo) y se quita del otro.
    """
    from dataclasses import replace

    if not (getattr(first, 'mega', False) and getattr(second, 'mega', False)):
        return first, second
    try:
        if float(score_second) > float(score_first):
            return replace(first, mega=False), second
        return first, replace(second, mega=False)
    except TypeError:  # orden que no es un dataclass: no tocar
        return first, second
