import logging

import config
import constants

from showdown.engine.objects import StateMutator
from showdown.engine.select_best_move import pick_safest
from showdown.engine.select_best_move import get_payoff_matrix


logger = logging.getLogger(__name__)


# Formato de Acción
def format_decision(battle, moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2):
    # Formats a decision for communication with Pokemon-Showdown
    if nMoveSwitch1 == 0:
        nMoveSwitch1 = ""
    if toMove1 == 0:
        toMove1 = ""
    if nMoveSwitch2 == 0:
        nMoveSwitch2 = ""
    if toMove2 == 0:
        toMove2 = ""
    message = "/choose {} {} {} {}, {} {} {} {}".format(moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2)
    return [message, str(battle.rqid)]


# Acción
def pick_safest_move_from_battles(battles):
    moveSwitch1 = input("¿move/switch/pass?")
    nMoveSwitch1 = input("¿0/1/2/3/4?")
    dynamax1 = input("¿/dynamax?")
    toMove1 = input("¿0/1/2?")
    moveSwitch2 = input("¿move/switch/pass?")
    nMoveSwitch2 = input("¿0/1/2/3/4?")
    dynamax2 = input("¿/dynamax?")
    toMove2 = input("¿0/1/2?")
    return moveSwitch1, (int)nMoveSwitch1, dynamax1, (int)toMove1, moveSwitch2, (int)nMoveSwitch2, dynamax2, (int)toMove2