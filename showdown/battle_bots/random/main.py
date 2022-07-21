import random
import logging

from showdown.battle import Battle

logger = logging.getLogger(__name__)

class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)

    async def action_team_preview(self, battle):
        eleccion = ''.join(random.sample("123456", 4))
        eleccion0 = eleccion[0]
        eleccion1 = eleccion[1]
        eleccion2 = eleccion[2]
        eleccion3 = eleccion[3]
        # Regreso los 4 enteros
        return int(eleccion0), int(eleccion1), int(eleccion2), int(eleccion3)

    async def action_battle(self, battle):
        # Inicio del Árbol...
        if battle.force_switch:
            # Si hay al menos un cambio obligatorio...
            izquierdaSwitch = json.loads(battle.request_json)["forceSwitch"][0]
            derechaSwitch = json.loads(battle.request_json)["forceSwitch"][1]
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
                switch3F = battle.user.reserved[0].fainted
                switch4F = battle.user.reserved[1].fainted
                if switch3F == False and switch4F == True:
                    # Solo Pokémon 3 puede reemplazar 
                    nMoveSwitch1 = 1
                elif switch3F == True and switch4F == False:
                    # Solo Pokémon 4 puede reemplazar
                    nMoveSwitch1 = 2
                else:
                    # Pokémon 3 y 4 pueden reemplazar
                    nMoveSwitch1 = random.choice([1, 2])
            elif derechaSwitch == False and derechaSwitch == True:
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
                switch3F = battle.user.reserved[0].fainted
                switch4F = battle.user.reserved[1].fainted
                if switch3F == False and switch4F == True:
                    # Solo Pokémon 3 puede reemplazar 
                    nMoveSwitch2 = 1
                elif switch3F == True and switch4F == False:
                    # Solo Pokémon 4 puede reemplazar
                    nMoveSwitch2 = 2
                else:
                    # Pokémon 3 y 4 pueden reemplazar
                    nMoveSwitch2 = random.choice([1, 2])
            else: # Los 2 Pokémon hacen cambio... los 2 suplentes deben estar disponibles
                # izquierda... ¿con quién cambio?
                moveSwitch1 = 'switch'
                dynamax1 = 0
                toMove1 = 0
                # Se toma la decisión...
                nMoveSwitch1 = random.choice([1, 2])
                # derecha... ¿con quien cambio?
                moveSwitch2 = 'switch'
                dynamax2 = 0
                toMove2 =  0
                # Se toma la decisión...
                if nMoveSwitch1 == 1:
                    nMoveSwitch2 = 2
                else:
                    nMoveSwitch2 = 1
        else:
            # Para el Pokémon de la izquierda... ¿no está debilitado?
            if !battle.user.active1.fainted:
                # ¿move o switch?
                moveSwitch1 = random.choice(['move', 'switch'])
                if moveSwitch1 == 'move':
                    # ¿Qué movimientos están disponibles?
                    pokemon1move = []
                    if !json.loads(battle.request_json)['active'][0]['moves'][0]['disabled']:
                        pokemon1move.append(1)
                    if !json.loads(battle.request_json)['active'][0]['moves'][1]['disabled']:
                        pokemon1move.append(2)
                    if !json.loads(battle.request_json)['active'][0]['moves'][2]['disabled']:
                        pokemon1move.append(3)
                    if !json.loads(battle.request_json)['active'][0]['moves'][3]['disabled']:
                        pokemon1move.append(4)
                    # ¿Hay al menos algún movimiento para escoger?
                    if len(pokemon1move) > 0:
                        # Sí, hay al menos uno...
                        nMoveSwitch1 = random.choice(pokemon1move)
                        # Ahora que se ha escogido al menos 1 movimiento... ¿es posible hacer dynamax?
                        if battle.user.active1.can_dynamax:
                            if (json.loads(battle.request_json)['active'][0]['canDynamax']):
                                # Sí es posible...
                                dynamax1 = random.choice([0, 1])
                                if dynamax1:
                                    battle.user.active1.can_dynamax = False
                                    battle.user.active2.can_dynamax = False
                                    battle.user.reserved[0].can_dynamax = False
                                    battle.user.reserved[1].can_dynamax = False
                                    # Como va a realizar dynamax...
                                    if json.loads(battle.request_json)['active'][0]['maxMoves']['maxMoves'][nMoveSwitch1]['target'] == 'adjacentFoe':
                                        toMove1 = random.choice([1, 2])
                                    else: # 'self'
                                        toMove1 = 0
                        else:
                            # No es posible...
                            dynamax1 = 0
                        # Ahora, ¿tiene un solo objetivo?
                            if json.loads(battle.request_json)['active'][0]['moves'][nMoveSwitch1]['target'] == 'allAdjacentFoes' or json.loads(battle.request_json)['active'][0]['moves'][nMoveSwitch1]['target'] == 'allySide' or or json.loads(battle.request_json)['active'][0]['moves'][nMoveSwitch1]['target'] == 'self':
                                toMove1 = 0
                            elif json.loads(battle.request_json)['active'][0]['moves'][nMoveSwitch1]['target'] == 'normal' or json.loads(battle.request_json)['active'][0]['moves'][nMoveSwitch1]['target'] == 'any':
                                toMove1 = random.choice([-2, 1, 2])
                            else:
                                logger.error("ACCION NO CONTEMPLADA" + str(json.loads(battle.request_json)['active'][1]['moves'][nMoveSwitch1]['target']))
                    else:
                        # No hay ningún movimiento disponible... algo muy poco probable... Struggle se considera como uno... así que solo queda cambiar, sí o sí debe haber al menos 1
                        switch3F = battle.user.reserved[0].fainted
                        switch4F = battle.user.reserved[1].fainted
                        if switch3F == False and switch4F == True:
                            # Solo Pokémon 3 puede reemplazar 
                            nMoveSwitch1 = 1
                        elif switch3F == True and switch4F == False:
                            # Solo Pokémon 4 puede reemplazar
                            nMoveSwitch1 = 2
                        else:
                            # Pokémon 3 y 4 pueden reemplazar
                            nMoveSwitch1 = random.choice([1, 2])
                else:
                    try:
                        if json.loads(battle.request_json)['active'][0]['trapped']:
                            # Si está atrapado... solo queda hacer un movimiento
                            pokemon1move = []
                            if !json.loads(battle.request_json)['active'][0]['moves'][0]['disabled']:
                                pokemon1move.append(1)
                            if !json.loads(battle.request_json)['active'][0]['moves'][1]['disabled']:
                                pokemon1move.append(2)
                            if !json.loads(battle.request_json)['active'][0]['moves'][2]['disabled']:
                                pokemon1move.append(3)
                            if !json.loads(battle.request_json)['active'][0]['moves'][3]['disabled']:
                                pokemon1move.append(4)
                            # Como hay al menos uno...
                            nMoveSwitch1 = random.choice(pokemon1move)
                            # Ahora que se ha escogido al menos 1 movimiento... ¿es posible hacer dynamax?
                            if battle.user.active1.can_dynamax:
                                if (json.loads(battle.request_json)['active'][0]['canDynamax']):
                                    # Sí es posible...
                                    dynamax1 = random.choice([0, 1])
                                    if dynamax1:
                                        battle.user.active1.can_dynamax = False
                                        battle.user.active2.can_dynamax = False
                                        battle.user.reserved[0].can_dynamax = False
                                        battle.user.reserved[1].can_dynamax = False
                                        # Como va a realizar dynamax...
                                        if json.loads(battle.request_json)['active'][0]['maxMoves']['maxMoves'][nMoveSwitch1]['target'] == 'adjacentFoe':
                                            toMove1 = random.choice([1, 2])
                                        else: # 'self'
                                            toMove1 = 0
                        else:
                            # No es posible...
                            dynamax1 = 0
                        # Ahora, ¿tiene un solo objetivo?
                            if json.loads(battle.request_json)['active'][0]['moves'][nMoveSwitch1]['target'] == 'allAdjacentFoes' or json.loads(battle.request_json)['active'][0]['moves'][nMoveSwitch1]['target'] == 'allySide' or or json.loads(battle.request_json)['active'][0]['moves'][nMoveSwitch1]['target'] == 'self':
                                toMove1 = 0
                            elif json.loads(battle.request_json)['active'][0]['moves'][nMoveSwitch1]['target'] == 'normal' or json.loads(battle.request_json)['active'][0]['moves'][nMoveSwitch1]['target'] == 'any':
                                toMove1 = random.choice([-2, 1, 2])
                            else:
                                logger.error("ACCION NO CONTEMPLADA" + str(json.loads(battle.request_json)['active'][1]['moves'][nMoveSwitch1]['target']))
                        else:
                            switch3F = battle.user.reserved[0].fainted
                            switch4F = battle.user.reserved[1].fainted
                            if switch3F == False and switch4F == True:
                                # Solo Pokémon 3 puede reemplazar 
                                nMoveSwitch1 = 1
                            elif switch3F == True and switch4F == False:
                                # Solo Pokémon 4 puede reemplazar
                                nMoveSwitch1 = 2
                            else:
                                # Pokémon 3 y 4 pueden reemplazar
                                nMoveSwitch1 = random.choice([1, 2])
                    except KeyError:
                        ...
            else:
                moveSwitch1 = 'pass'
                nMoveSwitch1 = 0
                dynamax1 = 0
                toMove1 = 0
            # Para el Pokémon de la derecha... ¿no está debilitado?
            if !battle.user.active2.fainted:
                # ¿move o switch?
                moveSwitch2 = random.choice(['move', 'switch'])
                if moveSwitch2 == 'move':
                    # ¿Qué movimientos están disponibles?
                    pokemon2move = []
                    if !json.loads(battle.request_json)['active'][1]['moves'][0]['disabled']:
                        pokemon2move.append(1)
                    if !json.loads(battle.request_json)['active'][1]['moves'][1]['disabled']:
                        pokemon2move.append(2)
                    if !json.loads(battle.request_json)['active'][1]['moves'][2]['disabled']:
                        pokemon2move.append(3)
                    if !json.loads(battle.request_json)['active'][1]['moves'][3]['disabled']:
                        pokemon2move.append(4)
                    # ¿Hay al menos algún movimiento para escoger?
                    if len(pokemon2move) > 0:
                        # Sí, hay al menos uno...
                        nMoveSwitch2 = random.choice(pokemon2move)
                        # Ahora que se ha escogido al menos 1 movimiento... ¿es posible hacer dynamax?
                        if battle.user.active2.can_dynamax:
                            if (json.loads(battle.request_json)['active'][1]['canDynamax']):
                                # Sí es posible...
                                dynamax2 = random.choice([0, 1])
                                if dynamax1:
                                    battle.user.active1.can_dynamax = False
                                    battle.user.active2.can_dynamax = False
                                    battle.user.reserved[0].can_dynamax = False
                                    battle.user.reserved[1].can_dynamax = False
                                    # Como va a realizar dynamax...
                                    if json.loads(battle.request_json)['active'][1]['maxMoves']['maxMoves'][nMoveSwitch1]['target'] == 'adjacentFoe':
                                        toMove2 = random.choice([1, 2])
                                    else: # 'self'
                                        toMove2 = 0
                        else:
                            # No es posible...
                            dynamax2 = 0
                        # Ahora, ¿tiene un solo objetivo?
                            if json.loads(battle.request_json)['active'][1]['moves'][nMoveSwitch1]['target'] == 'allAdjacentFoes' or json.loads(battle.request_json)['active'][1]['moves'][nMoveSwitch1]['target'] == 'allySide' or or json.loads(battle.request_json)['active'][1]['moves'][nMoveSwitch1]['target'] == 'self':
                                toMove2 = 0
                            elif json.loads(battle.request_json)['active'][1]['moves'][nMoveSwitch1]['target'] == 'normal' or json.loads(battle.request_json)['active'][1]['moves'][nMoveSwitch1]['target'] == 'any':
                                toMove2 = random.choice([-2, 1, 2])
                            else:
                                logger.error("ACCION NO CONTEMPLADA" + str(json.loads(battle.request_json)['active'][1]['moves'][nMoveSwitch1]['target']))
                    else:
                        # No hay ningún movimiento disponible... algo muy poco probable... Struggle se considera como uno... así que solo queda cambiar, sí o sí debe haber al menos 1
                        switch3F = battle.user.reserved[0].fainted
                        switch4F = battle.user.reserved[1].fainted
                        if switch3F == False and switch4F == True:
                            # Solo Pokémon 3 puede reemplazar 
                            nMoveSwitch2 = 1
                        elif switch3F == True and switch4F == False:
                            # Solo Pokémon 4 puede reemplazar
                            nMoveSwitch2 = 2
                        else:
                            # Pokémon 3 y 4 pueden reemplazar
                            nMoveSwitch2 = random.choice([1, 2])
                else:
                    try:
                        if json.loads(battle.request_json)['active'][1]['trapped']:
                            # Si está atrapado... solo queda hacer un movimiento
                            pokemon2move = []
                            if !json.loads(battle.request_json)['active'][1]['moves'][0]['disabled']:
                                pokemon2move.append(1)
                            if !json.loads(battle.request_json)['active'][1]['moves'][1]['disabled']:
                                pokemon2move.append(2)
                            if !json.loads(battle.request_json)['active'][1]['moves'][2]['disabled']:
                                pokemon2move.append(3)
                            if !json.loads(battle.request_json)['active'][1]['moves'][3]['disabled']:
                                pokemon2move.append(4)
                            # ¿Hay al menos algún movimiento para escoger?
                            if len(pokemon2move) > 0:
                                # Sí, hay al menos uno...
                                nMoveSwitch2 = random.choice(pokemon2move)
                                # Ahora que se ha escogido al menos 1 movimiento... ¿es posible hacer dynamax?
                                if battle.user.active2.can_dynamax:
                                    if (json.loads(battle.request_json)['active'][1]['canDynamax']):
                                        # Sí es posible...
                                        dynamax2 = random.choice([0, 1])
                                        if dynamax1:
                                            battle.user.active1.can_dynamax = False
                                            battle.user.active2.can_dynamax = False
                                            battle.user.reserved[0].can_dynamax = False
                                            battle.user.reserved[1].can_dynamax = False
                                            # Como va a realizar dynamax...
                                            if json.loads(battle.request_json)['active'][1]['maxMoves']['maxMoves'][nMoveSwitch1]['target'] == 'adjacentFoe':
                                                toMove2 = random.choice([1, 2])
                                            else: # 'self'
                                                toMove2 = 0
                                else:
                                    # No es posible...
                                    dynamax2 = 0
                                # Ahora, ¿tiene un solo objetivo?
                                    if json.loads(battle.request_json)['active'][1]['moves'][nMoveSwitch1]['target'] == 'allAdjacentFoes' or json.loads(battle.request_json)['active'][1]['moves'][nMoveSwitch1]['target'] == 'allySide' or or json.loads(battle.request_json)['active'][1]['moves'][nMoveSwitch1]['target'] == 'self':
                                        toMove2 = 0
                                    elif json.loads(battle.request_json)['active'][1]['moves'][nMoveSwitch1]['target'] == 'normal' or json.loads(battle.request_json)['active'][1]['moves'][nMoveSwitch1]['target'] == 'any':
                                        toMove2 = random.choice([-2, 1, 2])
                                    else:
                                        logger.error("ACCION NO CONTEMPLADA" + str(json.loads(battle.request_json)['active'][1]['moves'][nMoveSwitch1]['target']))
                        else:
                            switch3F = battle.user.reserved[0].fainted
                            switch4F = battle.user.reserved[1].fainted
                            if switch3F == False and switch4F == True:
                                # Solo Pokémon 3 puede reemplazar 
                                nMoveSwitch2 = 1
                            elif switch3F == True and switch4F == False:
                                # Solo Pokémon 4 puede reemplazar
                                nMoveSwitch2 = 2
                            else:
                                # Pokémon 3 y 4 pueden reemplazar
                                nMoveSwitch2 = random.choice([1, 2])
                    except KeyError:
                        ...
            else:
                moveSwitch2 = 'pass'
                nMoveSwitch2 = 0
                dynamax2 = 0
                toMove2 = 0
        return battle, moveSwitch1, int(nMoveSwitch1), dynamax1, int(toMove1), moveSwitch2, int(nMoveSwitch2), dynamax2, int(toMove2)