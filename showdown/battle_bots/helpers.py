import logging

import config
import constants

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
    if dynamax1 == "0":
        dynamax1 = ""
    else:
        dynamax1 = "dynamax"
    if dynamax2 == "0":
        dynamax2 = ""
    else:
        dynamax1 = "dynamax"
    message = "/choose {} {} {} {}, {} {} {} {}".format(moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2)
    return [message, str(battle.rqid)]