import logging

from showdown.battle import Battle

logger = logging.getLogger(__name__)

class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)

    async def action_team_preview(self, battle):
        # Configurado para INPUT por CMD
        print('Ingresa los 4 dígitos para el Team Preview... Ejemplo: "3142"')
        # eleccion es un entero: 3142
        eleccion = input()
        # Creo un array de enteros
        arrayEleccion = [int(a) for a in str(eleccion)]
        eleccion0 = arrayEleccion[0]
        eleccion1 = arrayEleccion[1]
        eleccion2 = arrayEleccion[2]
        eleccion3 = arrayEleccion[3]
        # Regreso los 4 enteros
        return int(eleccion0), int(eleccion1), int(eleccion2), int(eleccion3)

    async def action_battle(self, battle):
        # Configurado para INPUT por CMD
        print('Por favor, responda las siguientes preguntas...')
        print()
        moveSwitch1 = input("¿move/switch/pass?: ")
        nMoveSwitch1 = input("¿0/1/2/3/4?: ")
        dynamax1 = input("dynamax... ¿0/1?: ")
        toMove1 = input("¿0/1/2?: ")
        moveSwitch2 = input("¿move/switch/pass?: ")
        nMoveSwitch2 = input("¿0/1/2/3/4?: ")
        dynamax2 = input("dynamax... ¿0/1?: ")
        toMove2 = input("¿0/1/2?: ")
        return battle, moveSwitch1, int(nMoveSwitch1), dynamax1, int(toMove1), moveSwitch2, int(nMoveSwitch2), dynamax2, int(toMove2)