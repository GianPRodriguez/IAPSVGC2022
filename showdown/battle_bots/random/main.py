import random
import json
import logging
import constants
from showdown.battle import Battle
from showdown.battle_bots.helpers import getMoveName
from showdown.battle_bots.helpers import getColorMove
from showdown.battle_bots.helpers import getToMovePokemon


logger = logging.getLogger(__name__)

class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)

    def _choose_switch_slot(self, battle):
        switch3F = battle.user.reserved[0].fainted
        switch4F = battle.user.reserved[1].fainted
        if switch3F is False and switch4F is True:
            return 3
        if switch3F is True and switch4F is False:
            return 4
        return random.choice([3, 4])

    def _get_available_moves(self, battle, active_index):
        moves = battle.request_json.get('active', 'False')[active_index].get('moves', 'False')
        return [index + 1 for index, move in enumerate(moves) if move.get('disabled', 'False') is False]

#######################################################################################################################
# OK - Escoge los 4 Pokémon de los 6 disponibles de manera aleatoria para el inicio de la batalla
#######################################################################################################################
    async def action_team_preview(self, battle: Battle) -> tuple[int, ...]:
        # 1. Obtener 4 índices únicos aleatorios entre 1 y 6
        pkmn_indices = random.sample(range(1, 7), 4)

        # 2. Acceso seguro a la lista de Pokémon en el JSON
        pokemon_list = battle.request_json.get('side', {}).get('pokemon', [])

        # 3. Extraer nombres de forma segura usando list comprehension
        names = []
        for idx in pkmn_indices:
            if 0 <= idx - 1 < len(pokemon_list):
                name = pokemon_list[idx - 1].get('details', '').partition(",")[0]
            else:
                name = "Unknown"
            names.append(name)

        # 4. Enviar a la cola para la interfaz gráfica
        battle.q.put([1, names[0], names[1], names[2], names[3], 0, 0])

        # 5. Retornar los 4 enteros
        return tuple(pkmn_indices) # tupla (5, 2, 1, 4)

#######################################################################################################################
# Escoge los 4 Pokémon de los 6 disponibles de manera aleatoria para el inicio de la batalla
#######################################################################################################################
    async def action_battle(self, battle, state):
        moveSwitch1 = None
        nMoveSwitch1 = 0
        dynamax1 = 0
        toMove1 = 0
        moveSwitch2 = None
        nMoveSwitch2 = 0
        dynamax2 = 0
        toMove2 = 0
        # Inicio del Árbol...
        if battle.force_switch:
            # Si hay al menos un cambio obligatorio...
            izquierdaSwitch = (battle.request_json)["forceSwitch"][0]
            derechaSwitch = (battle.request_json)["forceSwitch"][1]
            # Si alguno es falso... hay que hacer pass
            if izquierdaSwitch == True and derechaSwitch == False:
                # derecha
                moveSwitch2 = 'pass'
                nMoveSwitch2 = 0
                dynamax2 = 0
                toMove2 = 0
                # izquierda... ¿con quién cambio?
                moveSwitch1 = 'switch'
                dynamax1 = 0
                toMove1 = 0
                # Se toma la decisión...
                nMoveSwitch1 = self._choose_switch_slot(battle)
            elif izquierdaSwitch == False and derechaSwitch == True:
                # izquierda
                moveSwitch1 = 'pass'
                nMoveSwitch1 = 0
                dynamax1 = 0
                toMove1 = 0
                # derecha... ¿con quien cambio?
                moveSwitch2 = 'switch'
                dynamax2 = 0
                toMove2 =  0
                # Se toma la decisión...
                nMoveSwitch2 = self._choose_switch_slot(battle)
            else: # Los 2 Pokémon hacen cambio... los 2 suplentes deben estar disponibles... o solo hay 1 Pokémon disponible
                # izquierda... ¿con quién cambio?
                if battle.user.reserved[0].fainted or battle.user.reserved[1].fainted:
                    moveSwitch1 = 'switch'
                    dynamax1 = 0
                    toMove1 = 0
                    # Se toma la decisión...
                    if battle.user.reserved[0].fainted:
                        nMoveSwitch1 = 4
                    else:
                        nMoveSwitch1 = 3
                else:
                    moveSwitch1 = 'switch'
                    dynamax1 = 0
                    toMove1 = 0
                    nMoveSwitch1 = random.choice([3, 4])
                # derecha... ¿con quien cambio?
                if battle.user.reserved[0].fainted or battle.user.reserved[1].fainted:
                    moveSwitch2 = 'pass'
                    dynamax2 = 0
                    toMove2 = 0
                    nMoveSwitch2 = 0
                else:
                    moveSwitch2 = 'switch'
                    dynamax2 = 0
                    toMove2 =  0
                    # Se toma la decisión...
                    if nMoveSwitch1 == 3:
                        nMoveSwitch2 = 4
                    else:
                        nMoveSwitch2 = 3
        else:
            # Para el Pokémon de la izquierda... ¿no está debilitado?
            if battle.user.active1.fainted == False:
                # Movimiento especial que bloquea el movimiento del Pokémon en el sgte turno
                if (battle.request_json).get('active','False')[0].get('moves','False')[0].get('pp','False') == 'False':
                    print("The opposing Shedinja used Phantom Force!")
                    # Está atrapado y tiene que hacer ese movimiento...
                    moveSwitch1 = 'move'
                    nMoveSwitch1 = 1
                    dynamax1 = 0
                    toMove1 = 0
                else:
                    # ¿move o switch?
                    moveSwitch1 = random.choice(['move', 'switch'])
                    if moveSwitch1 == 'move':
                        # ¿Qué movimientos están disponibles?
                        pokemon1move = self._get_available_moves(battle, 0)
                        # ¿Hay al menos algún movimiento para escoger?
                        if len(pokemon1move) > 0:
                            # Sí, hay al menos uno...
                            nMoveSwitch1 = random.choice(pokemon1move)
                            # Ahora que se ha escogido al menos 1 movimiento... ¿es posible hacer dynamax?
                            if battle.user.active1.can_dynamax:
                                logger.warning(str(battle.user.active1.can_dynamax))
                                logger.warning(str(type(battle.user.active1.can_dynamax)))
                                if ((battle.request_json).get('active','False')[0].get('canDynamax','False')):
                                    # Sí es posible...
                                    dynamax1 = random.choice([0, 1])
                                    if dynamax1:
                                        battle.user.active1.can_dynamax = False
                                        battle.user.active2.can_dynamax = False
                                        battle.user.reserved[0].can_dynamax = False
                                        battle.user.reserved[1].can_dynamax = False
                                        # Como va a realizar dynamax...
                                        if (battle.request_json)['active'][0]['maxMoves']['maxMoves'][nMoveSwitch1-1]['target'] == 'adjacentFoe':
                                            toMove1 = random.choice([1, 2])
                                        else: # 'self'
                                            toMove1 = 0
                                    else:
                                        # Ahora, ¿tiene un solo objetivo?
                                        if (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allySide' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'self' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'all':
                                            toMove1 = 0
                                        elif (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'normal' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'any' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'selected-pokemon':
                                            if battle.user.active2.fainted == False:
                                                toMove1 = random.choice([-2, 1, 2])
                                            else:
                                                toMove1 = random.choice([1, 2])
                                        else:
                                            logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target']))
                            else:
                                # No es posible...
                                dynamax1 = 0
                                # ¿Actualmente es Dynamax?
                                if constants.DYNAMAX in battle.user.active1.volatile_statuses:
                                    if (battle.request_json)['active'][0]['maxMoves']['maxMoves'][nMoveSwitch1-1]['target'] == 'adjacentFoe':
                                        toMove1 = random.choice([1, 2])
                                    else: # 'self'
                                        toMove1 = 0
                                else:
                                    # Ahora, ¿tiene un solo objetivo?
                                    if (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allySide' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'self' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'all':
                                        toMove1 = 0
                                    elif (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'normal' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'any' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'selected-pokemon':
                                        if battle.user.active2.fainted == False:
                                            toMove1 = random.choice([-2, 1, 2])
                                        else:
                                            toMove1 = random.choice([1, 2])
                                    else:
                                        logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target']))
                        else:
                            # No hay ningún movimiento disponible... algo muy poco probable... Struggle se considera como uno... así que solo queda cambiar, sí o sí debe haber al menos 1
                            moveSwitch1 = 'switch'
                            nMoveSwitch1 = self._choose_switch_slot(battle)
                    else:
                        # El Random ha elegido Switch
                        if ((battle.request_json).get('active','False')[0].get('trapped','False') == True or (battle.request_json).get('active','False')[0].get('maybeTrapped','False') == True) or (battle.user.reserved[0].fainted and battle.user.reserved[1].fainted):
                            # Si está atrapado... o los 2 reservados están debilitados... solo queda hacer un movimiento
                            print("Para 0 ... Si está atrapado... o los 2 reservados están debilitados... solo queda hacer un movimiento")
                            moveSwitch1 = 'move'
                            pokemon1move = []
                            if (battle.request_json).get('active','False')[0].get('moves','False')[0].get('disabled','False') == False:
                                pokemon1move.append(1)
                            if (battle.request_json).get('active','False')[0].get('moves','False')[1].get('disabled','False') == False:
                                pokemon1move.append(2)
                            if (battle.request_json).get('active','False')[0].get('moves','False')[2].get('disabled','False') == False:
                                pokemon1move.append(3)
                            if (battle.request_json).get('active','False')[0].get('moves','False')[3].get('disabled','False') == False:
                                pokemon1move.append(4)
                            # Como hay al menos uno...
                            nMoveSwitch1 = random.choice(pokemon1move)
                            # Ahora que se ha escogido al menos 1 movimiento... ¿es posible hacer dynamax?
                            if battle.user.active1.can_dynamax:
                                if ((battle.request_json)['active'][0]['canDynamax']):
                                    # Sí es posible...
                                    dynamax1 = random.choice([0, 1])
                                    if dynamax1:
                                        battle.user.active1.can_dynamax = False
                                        battle.user.active2.can_dynamax = False
                                        battle.user.reserved[0].can_dynamax = False
                                        battle.user.reserved[1].can_dynamax = False
                                        # Como va a realizar dynamax...
                                        if (battle.request_json)['active'][0]['maxMoves']['maxMoves'][nMoveSwitch1-1]['target'] == 'adjacentFoe':
                                            toMove1 = random.choice([1, 2])
                                        else: # 'self'
                                            toMove1 = 0
                                    else:
                                        # Ahora, ¿tiene un solo objetivo?
                                        if (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allySide' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'self' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'all':
                                            toMove1 = 0
                                        elif (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'normal' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'any' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'selected-pokemon':
                                            if battle.user.active2.fainted == False:
                                                toMove1 = random.choice([-2, 1, 2])
                                            else:
                                                toMove1 = random.choice([1, 2])
                                        else:
                                            logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target']))
                            else:
                                # No es posible...
                                dynamax1 = 0
                                # ¿Actualmente es Dynamax?
                                if constants.DYNAMAX in battle.user.active1.volatile_statuses:
                                    if (battle.request_json)['active'][0]['maxMoves']['maxMoves'][nMoveSwitch1-1]['target'] == 'adjacentFoe':
                                        toMove1 = random.choice([1, 2])
                                    else: # 'self'
                                        toMove1 = 0
                                else:
                                    # Ahora, ¿tiene un solo objetivo?
                                    if (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'allySide' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'self' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'all':
                                        toMove1 = 0
                                    elif (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'normal' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'any' or (battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target'] == 'selected-pokemon':
                                        if battle.user.active2.fainted == False:
                                            toMove1 = random.choice([-2, 1, 2])
                                        else:
                                            toMove1 = random.choice([1, 2])
                                    else:
                                        logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][0]['moves'][nMoveSwitch1-1]['target']))
                        else:
                            # Si no... sí puede hacer un cambio normal
                            nMoveSwitch1 = self._choose_switch_slot(battle)
                            print("CAMBIO ALEATORIO NORMAL PARA 1")
            else:
                moveSwitch1 = 'pass'
                nMoveSwitch1 = 0
                dynamax1 = 0
                toMove1 = 0
            # Para el Pokémon de la derecha... ¿no está debilitado?
            if battle.user.active2.fainted == False:
                # Movimiento especial...
                if (battle.request_json).get('active','False')[1].get('moves','False')[0].get('pp','False') == 'False':
                    print("The opposing Shedinja used Phantom Force!")
                    # Está atrapado y tiene que hacer ese movimiento...
                    moveSwitch2 = 'move'
                    nMoveSwitch2 = 1
                    dynamax2 = 0
                    toMove2 = 0
                else:
                    # ¿move o switch?
                    moveSwitch2 = random.choice(['move', 'switch'])
                    if moveSwitch2 == 'move':
                        # ¿Qué movimientos están disponibles?
                        pokemon2move = self._get_available_moves(battle, 1)
                        # ¿Hay al menos algún movimiento para escoger?
                        if len(pokemon2move) > 0:
                            # Sí, hay al menos uno...
                            nMoveSwitch2 = random.choice(pokemon2move)
                            # Ahora que se ha escogido al menos 1 movimiento... ¿es posible hacer dynamax?
                            if battle.user.active2.can_dynamax:
                                if ((battle.request_json).get('active','False')[1].get('canDynamax','False')):
                                    # Sí es posible...
                                    dynamax2 = random.choice([0, 1])
                                    if dynamax2:
                                        battle.user.active1.can_dynamax = False
                                        battle.user.active2.can_dynamax = False
                                        battle.user.reserved[0].can_dynamax = False
                                        battle.user.reserved[1].can_dynamax = False
                                        # Como va a realizar dynamax...
                                        if (battle.request_json)['active'][1]['maxMoves']['maxMoves'][nMoveSwitch2-1]['target'] == 'adjacentFoe':
                                            toMove2 = random.choice([1, 2])
                                        else: # 'self'
                                            toMove2 = 0
                                    else:
                                        # Ahora, ¿tiene un solo objetivo?
                                        if (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allySide' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'self' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'all':
                                            toMove2 = 0
                                        elif (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'normal' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'any' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'selected-pokemon':
                                            if battle.user.active1.fainted == False:
                                                toMove2 = random.choice([-1, 1, 2])
                                            else:
                                                toMove2 = random.choice([1, 2])
                                        else:
                                            logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target']))
                            else:
                                # No es posible...
                                dynamax2 = 0
                                # ¿Actualmente es Dynamax?
                                if constants.DYNAMAX in battle.user.active2.volatile_statuses:
                                    if (battle.request_json)['active'][1]['maxMoves']['maxMoves'][nMoveSwitch2-1]['target'] == 'adjacentFoe':
                                        toMove2 = random.choice([1, 2])
                                    else: # 'self'
                                        toMove2 = 0
                                else:
                                    # Ahora, ¿tiene un solo objetivo?
                                    if (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allySide' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'self' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'all':
                                        toMove2 = 0
                                    elif (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'normal' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'any' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'selected-pokemon':
                                        if battle.user.active1.fainted == False:
                                            toMove2 = random.choice([-1, 1, 2])
                                        else:
                                            toMove2 = random.choice([1, 2])
                                    else:
                                        logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target']))
                        else:
                            # No hay ningún movimiento disponible... algo muy poco probable... Struggle se considera como uno... así que solo queda cambiar, sí o sí debe haber al menos 1
                            moveSwitch2 = 'switch'
                            nMoveSwitch2 = self._choose_switch_slot(battle)
                            print("CAMBIO ALEATORIO NORMAL PARA 2")
                    else:
                        # El Random ha elegido switch
                        if ((battle.request_json).get('active','False')[1].get('trapped','False') == True or (battle.request_json).get('active','False')[1].get('maybeTrapped','False') == True) or ((battle.user.reserved[0].fainted or battle.user.reserved[1].fainted) and (nMoveSwitch1 == 3 or nMoveSwitch1 == 4)) or (battle.user.reserved[0].fainted and battle.user.reserved[1].fainted):
                            # Si está atrapado... o los 2 reservados están debilitados... solo queda hacer un movimiento
                            print("Para 1 ... Si está atrapado... o los 2 reservados están debilitados... o solo un reservado está habilitado y ya fue escogido por pkmn 0... solo queda hacer un movimiento")
                            moveSwitch2 = 'move'
                            pokemon2move = []
                            if (battle.request_json).get('active','False')[1].get('moves','False')[0].get('disabled') == False:
                                pokemon2move.append(1)
                            if (battle.request_json).get('active','False')[1].get('moves','False')[1].get('disabled') == False:
                                pokemon2move.append(2)
                            if (battle.request_json).get('active','False')[1].get('moves','False')[2].get('disabled') == False:
                                pokemon2move.append(3)
                            if (battle.request_json).get('active','False')[1].get('moves','False')[3].get('disabled') == False:
                                pokemon2move.append(4)
                            # ¿Hay al menos algún movimiento para escoger?
                            if len(pokemon2move) > 0:
                                # Sí, hay al menos uno...
                                nMoveSwitch2 = random.choice(pokemon2move)
                                # Ahora que se ha escogido al menos 1 movimiento... ¿es posible hacer dynamax?
                                if battle.user.active2.can_dynamax:
                                    if ((battle.request_json)['active'][1]['canDynamax']):
                                        # Sí es posible...
                                        dynamax2 = random.choice([0, 1])
                                        if dynamax2:
                                            battle.user.active1.can_dynamax = False
                                            battle.user.active2.can_dynamax = False
                                            battle.user.reserved[0].can_dynamax = False
                                            battle.user.reserved[1].can_dynamax = False
                                            # Como va a realizar dynamax...
                                            if (battle.request_json)['active'][1]['maxMoves']['maxMoves'][nMoveSwitch2-1]['target'] == 'adjacentFoe':
                                                toMove2 = random.choice([1, 2])
                                            else: # 'self'
                                                toMove2 = 0
                                        else:
                                            # Ahora, ¿tiene un solo objetivo?
                                            if (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allySide' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'self' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'all':
                                                toMove2 = 0
                                            elif (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'normal' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'any' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'selected-pokemon':
                                                if battle.user.active1.fainted == False:
                                                    toMove2 = random.choice([-1, 1, 2])
                                                else:
                                                    toMove2 = random.choice([1, 2])
                                            else:
                                                logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target']))
                                else:
                                    # No es posible...
                                    dynamax2 = 0
                                    # ¿Actualmente es Dynamax?
                                    if constants.DYNAMAX in battle.user.active2.volatile_statuses:
                                        if (battle.request_json)['active'][1]['maxMoves']['maxMoves'][nMoveSwitch2-1]['target'] == 'adjacentFoe':
                                            toMove2 = random.choice([1, 2])
                                        else: # 'self'
                                            toMove2 = 0
                                    else:
                                        # Ahora, ¿tiene un solo objetivo?
                                        if (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allAdjacentFoes' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'allySide' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'self' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'all':
                                            toMove2 = 0
                                        elif (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'normal' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'any' or (battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target'] == 'selected-pokemon':
                                            if battle.user.active1.fainted == False:
                                                toMove2 = random.choice([-1, 1, 2])
                                            else:
                                                toMove2 = random.choice([1, 2])
                                        else:
                                            logger.error("ACCION NO CONTEMPLADA" + str((battle.request_json)['active'][1]['moves'][nMoveSwitch2-1]['target']))
                        else:
                            if nMoveSwitch1 == 0:
                                nMoveSwitch2 = self._choose_switch_slot(battle)
                            else:
                                nMoveSwitch2 = 3 if nMoveSwitch1 == 4 else 4
            else:
                moveSwitch2 = 'pass'
                nMoveSwitch2 = 0
                dynamax2 = 0
                toMove2 = 0


        # Se define los Text Pkmn 1
        if moveSwitch1 == 'move':
            text1 = await getMoveName(battle, 0, nMoveSwitch1-1, dynamax1)
            text1Aux = text1.lower().replace(' ', '-')
            color1 = await getColorMove(text1Aux)
            text3 = await getToMovePokemon(battle, text1Aux, toMove1)
            color3 = '#FFFFFF'
        elif moveSwitch1 == 'pass':
            text1 = 'Pass'
            color1 = '#FFFFFF'
            text3 = 'Pass'
            color3 = '#FFFFFF'
        elif moveSwitch1 == 'switch':
            text1 = 'Switch to'
            color1 = '#FFFFFF'
            text3 = battle.user.reserved[nMoveSwitch1-3].name
            color3 = '#FFFFFF'
        else:
            logger.error("NO DEBE ENTRAR ACÁ - moveSwitch1 == 'switch'")

        # Se define los Text Pkmn 2
        if moveSwitch2 == 'move':
            text2 = await getMoveName(battle, 1, nMoveSwitch2-1, dynamax2)
            text2Aux = text2.lower().replace(' ', '-')
            color2 = await getColorMove(text2Aux)
            text4 = await getToMovePokemon(battle, text2Aux, toMove2)
            color4 = '#FFFFFF'
        elif moveSwitch2 == 'pass':
            text2 = 'Pass'
            color2 = '#FFFFFF'
            text4 = 'Pass'
            color4 = '#FFFFFF'
        elif moveSwitch2 == 'switch':
            text2 = 'Switch to'
            color2 = '#FFFFFF'
            text4 = battle.user.reserved[nMoveSwitch2-3].name
            color4 = '#FFFFFF'
        else:
            logger.error("NO DEBE ENTRAR ACÁ - moveSwitch2 == 'switch'")

        
        # Se envía al Queue la acción
        lastReward, reward = self.calculateRewards()
        logger.warning([2, text1, color1, text2, color2, text3, color3, text4, color4, lastReward, reward])
        battle.q.put([2, text1, color1, text2, color2, text3, color3, text4, color4, lastReward, reward])
        return moveSwitch1, int(nMoveSwitch1), dynamax1, int(toMove1), moveSwitch2, int(nMoveSwitch2), dynamax2, int(toMove2)