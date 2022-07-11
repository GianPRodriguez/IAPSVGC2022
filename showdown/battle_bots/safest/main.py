import logging

from showdown.battle import Battle

from ..helpers import format_decision
from ..helpers import pick_safest_move_from_battles

logger = logging.getLogger(__name__)

class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)

    # Busca el mejor movimiento - origen
    def find_best_move(self):
        battles = self.prepare_battles(join_moves_together=True)        
        # safest_move es la decisión del bot en bruto
        safest_move = pick_safest_move_from_battles(battles)
        return format_decision(self, safest_move)