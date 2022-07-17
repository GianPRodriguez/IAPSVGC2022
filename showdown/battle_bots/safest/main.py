import logging

from showdown.battle import Battle

from ..helpers import format_decision
from ..helpers import pick_safest_move_from_battles

logger = logging.getLogger(__name__)

class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)

    # Busca el mejor movimiento - origen
    def find_best_move(self, battle):
        print("ENTRO AL BOT")    
        # safest_move es la decisión del bot en bruto
        moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2 = pick_safest_move_from_battles(battle)
        # Retorna el mensaje formateado listo para ser enviado al WebSocket
        return format_decision(self, moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2)