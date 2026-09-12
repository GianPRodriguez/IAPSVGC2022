import math
import json
import random
import logging
import constants
import json
import neat
import pickle

from numpy import sqrt
from showdown.battle import Battle
from API.MoveData import getColorMove
from API.PokeAPI import getMoveTarget
from showdown.battle_bots.helpers import Box
from showdown.battle_bots.helpers import getMoveName
from showdown.battle_bots.helpers import findRewardSingles
from showdown.battle_bots.helpers import formula
from showdown.battle_bots.helpers import findGroupReward
from showdown.battle_bots.helpers import findTeam
from showdown.battle_bots.helpers import getATBoost
from showdown.battle_bots.helpers import getDEBoost


diccionario = json.loads(open("API/JSONs/movesANN.json","r").read())

class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)

    async def action_team_preview(self, battle):
        bestRewardIJ = []
        for i in range(180):
            rewardx1OpponentTeam = -1000000
            bestJ = -1
            o1, o2, or1, or2 = findRewardSingles(i, 1, battle)
            for j in range(180):
                u1, u2, ur1, ur2 = findRewardSingles(j, 0, battle)
                aux = findGroupReward(o1, o2, or1, or2, u1, u2, ur1, ur2, battle)
                #if i == 177:
                #    print("j: {} aux: {} bestJ: {} rewardx1OpponentTeam: {}".format(j, aux, bestJ, rewardx1OpponentTeam))
                if aux > rewardx1OpponentTeam:
                    rewardx1OpponentTeam = aux
                    bestJ = j
            bestRewardIJ.append(Box(rewardx1OpponentTeam, i, bestJ))
            #print(bestRewardIJ[i].get())
        bestRewardIJ.sort()
        # Obtener Suma de Reward
        sum = 0
        for i in range(32):
            sum = sum + (5000-bestRewardIJ[i].get()[2])
        listWeights = []
        listJ = []
        for i in range(32):
            listWeights.append((5000-bestRewardIJ[i].get()[2])/sum)
            listJ.append(bestRewardIJ[i].get()[1])
        bestJChoose = random.choices(listJ, weights=listWeights, k=1)[0]
        # bestJChoose -> team choosed
        eleccion0, eleccion1, eleccion2, eleccion3 = findTeam(bestJChoose, battle)
        # get names of Pkmn
        namePokemon0 = battle.request_json.get('side','False').get('pokemon', 'False')[eleccion0-1].get('details', 'False').partition(",")[0]
        namePokemon1 = battle.request_json.get('side','False').get('pokemon', 'False')[eleccion1-1].get('details', 'False').partition(",")[0]
        namePokemon2 = battle.request_json.get('side','False').get('pokemon', 'False')[eleccion2-1].get('details', 'False').partition(",")[0]
        namePokemon3 = battle.request_json.get('side','False').get('pokemon', 'False')[eleccion3-1].get('details', 'False').partition(",")[0]
        # Send to Queue
        if not battle.isTraining:
            print("[ACTION]   {}".format([namePokemon0, namePokemon1, namePokemon2, namePokemon3]))
            battle.q.put([1,namePokemon0, namePokemon1, namePokemon2, namePokemon3, 0, 0])
        # Return 4 integers
        return eleccion0, eleccion1, eleccion2, eleccion3

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
                switch3F = battle.user.reserved[0].isFainted()
                switch4F = battle.user.reserved[1].isFainted()
                if switch3F == False and switch4F == True:
                    # Solo Pokémon 3 puede reemplazar 
                    nMoveSwitch1 = 3
                elif switch3F == True and switch4F == False:
                    # Solo Pokémon 4 puede reemplazar
                    nMoveSwitch1 = 4
                else:
                    # Pokémon 3 y 4 pueden reemplazar
                    nMoveSwitch1 = random.choice([3, 4])
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
                switch3F = battle.user.reserved[0].isFainted()
                switch4F = battle.user.reserved[1].isFainted()
                if switch3F == False and switch4F == True:
                    # Solo Pokémon 3 puede reemplazar 
                    nMoveSwitch2 = 3
                elif switch3F == True and switch4F == False:
                    # Solo Pokémon 4 puede reemplazar
                    nMoveSwitch2 = 4
                else:
                    # Pokémon 3 y 4 pueden reemplazar
                    nMoveSwitch2 = random.choice([3, 4])
            else: # Los 2 Pokémon hacen cambio... los 2 suplentes deben estar disponibles... o solo hay 1 Pokémon disponible
                # izquierda... ¿con quién cambio?
                if battle.user.reserved[0].isFainted() or battle.user.reserved[1].isFainted():
                    moveSwitch1 = 'switch'
                    dynamax1 = 0
                    toMove1 = 0
                    # Se toma la decisión...
                    if battle.user.reserved[0].isFainted():
                        nMoveSwitch1 = 4
                    else:
                        nMoveSwitch1 = 3
                else:
                    moveSwitch1 = 'switch'
                    dynamax1 = 0
                    toMove1 = 0
                    nMoveSwitch1 = random.choice([3, 4])
                # derecha... ¿con quien cambio?
                if battle.user.reserved[0].isFainted() or battle.user.reserved[1].isFainted():
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
            # Lee el Genoma
            with open("./old/neuroevolution/modelos/winner.pkl", "rb") as f:
                genome = pickle.load(f)
            # Archivo de Configuración
            config_path = "./old/neuroevolution/config-feedforward"
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
            # Crea la Red Neuronal
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # Dado un 'Estado', devuelve el array 'action_values'
            action_values = net.activate(state)
            # Devuelve el mayor valor del 'action_values'
            action_value = int(action_values.index(max(action_values)))
            # Mensaje de la acción
            action = '/choose ' + diccionario[str(action_value)]
            # INICIO CORTE PASS
            if int(state[10]) == 0:
                # Pkmn de la izquierda fainted
                arreglo = action.split(',')
                arreglo[0] = 'pass'
                action = '/choose ' + arreglo[0] + ',' + arreglo[1]
            elif state[33] == 0:
                # Pkmn de la derecha fainted
                arreglo = action.split(',')
                arreglo[1] = 'pass'
                action = arreglo[0] + ', ' + arreglo[1]
            # FIN CORTE PASS
            # INICIO CALYREX
            if int(state[4]) == 0 and int(state[5]) == 0 and int(state[6]) == 0 and int(state[7]) == 0 and int(state[8]) == 1 and int(state[9]) == 0:
                # Izquierda
                arreglo = action.split(',')
                arreglo[0] = 'move 4'
                action = '/choose ' + arreglo[0] + ',' + arreglo[1]
            elif int(state[27]) == 0 and int(state[28]) == 0 and int(state[29]) == 0 and int(state[30]) == 0 and int(state[31]) == 1 and int(state[32]) == 0:
                # derecha
                arreglo = action.split(',')
                arreglo[1] = 'move 4'
                action = arreglo[0] + ', ' + arreglo[1]
            # FIN CALYREX
            action = action.replace("/choose ", "")
            arreglo = action.split(',')
            # PRIMERA ACCION
            arreglo[0] = arreglo[0].lstrip()
            primeraAccion = arreglo[0].split()
            moveSwitch1 = primeraAccion[0]
            if moveSwitch1 == 'pass':
                nMoveSwitch1 = 0
                dynamax1 = 0
                toMove1 = 0
            else:
                nMoveSwitch1= primeraAccion[1]
                if len(primeraAccion) == 3:
                    dynamax1 = 0
                    toMove1 = primeraAccion[2]
                if len(primeraAccion) == 4:
                    dynamax1 = 1
                    toMove1 = primeraAccion[3]
            # SEGUNDA ACCION
            arreglo[1] = arreglo[1].lstrip()
            segundaAccion = arreglo[1].split()
            moveSwitch2 = segundaAccion[0]
            if moveSwitch2 == 'pass':
                nMoveSwitch2 = 0
                dynamax2 = 0
                toMove2 = 0
            else:
                nMoveSwitch2= segundaAccion[1]
                if len(segundaAccion) == 3:
                    dynamax2 = 0
                    toMove2 = segundaAccion[2]
                if len(segundaAccion) == 4:
                    dynamax2 = 1
                    toMove2 = segundaAccion[3]
        nMoveSwitch1 = int(nMoveSwitch1)
        dynamax1 = int(dynamax1)
        toMove1 = int(toMove1)
        nMoveSwitch2 = int(nMoveSwitch2)
        dynamax2 = int(dynamax2)
        toMove2 = int(toMove2)
        # Se define los Text
        try:
            if moveSwitch1 == 'move':
                text1 = await getMoveName(battle, 0, nMoveSwitch1-1, dynamax1)
                text1Aux = text1.lower().replace(' ', '-')
                color1 = await getColorMove(text1Aux)
                text3 = getMoveTarget(battle, text1Aux, toMove1)
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
                print("NO DEBE ENTRAR ACÁ - moveSwitch1 == 'switch'")
            if moveSwitch2 == 'move':
                text2 = await getMoveName(battle, 1, nMoveSwitch2-1, dynamax2)
                text2Aux = text2.lower().replace(' ', '-')
                color2 = await getColorMove(text2Aux)
                text4 = getMoveTarget(battle, text2Aux, toMove2)
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
                print("NO DEBE ENTRAR ACÁ - moveSwitch2 == 'switch'")
        except:
            ...
        # Se envía al Queue la acción
        deltaReward, reward = battle.calculateRewards()
        if battle.error:
            deltaReward, reward = -9999, reward
            battle.cantInvActions += 1
            battle.cantInvActionsSeg += 1
            battle, moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2, battle.reward, battle.score = await action_battle2(self, battle, state)
            return battle, moveSwitch1, nMoveSwitch1, dynamax1, toMove1, moveSwitch2, nMoveSwitch2, dynamax2, toMove2,  battle.reward, battle.score
        else:
            battle.cantInvActionsSeg = 0
        if deltaReward > 1000:
            color5 = "#325928"
        elif deltaReward >= 0:
            color5 = "#565828"
        else:
            color5 = "#582831"
        if reward > 0:
            color6 = "#325928"
        else:
            color6 = "#582831"
        if not battle.isTraining:
            print("[REWARD]   {}".format([math.floor(deltaReward), math.floor(reward)]))
            print("[ACTION]   {}".format([text1, text3, text2, text4]))
            print("[COLOR]    {}".format([color1, color3, color2, color4, color5, color6]))
            battle.q.put([2, text1, color1, text2, color2, text3, color3, text4, color4, math.floor(deltaReward), color5, math.floor(reward), color6])
        return battle, moveSwitch1, int(nMoveSwitch1), dynamax1, int(toMove1), moveSwitch2, int(nMoveSwitch2), dynamax2, int(toMove2), math.floor(deltaReward), math.floor(reward)

    async def action_battle2(self, battle, state):
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
                    switch3F = battle.user.reserved[0].fainted
                    switch4F = battle.user.reserved[1].fainted
                    if switch3F == False and switch4F == True:
                        # Solo Pokémon 3 puede reemplazar 
                        nMoveSwitch1 = 3
                    elif switch3F == True and switch4F == False:
                        # Solo Pokémon 4 puede reemplazar
                        nMoveSwitch1 = 4
                    else:
                        # Pokémon 3 y 4 pueden reemplazar
                        nMoveSwitch1 = random.choice([3, 4])
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
                    switch3F = battle.user.reserved[0].fainted
                    switch4F = battle.user.reserved[1].fainted
                    if switch3F == False and switch4F == True:
                        # Solo Pokémon 3 puede reemplazar 
                        nMoveSwitch2 = 3
                    elif switch3F == True and switch4F == False:
                        # Solo Pokémon 4 puede reemplazar
                        nMoveSwitch2 = 4
                    else:
                        # Pokémon 3 y 4 pueden reemplazar
                        nMoveSwitch2 = random.choice([3, 4])
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
                    # Movimiento especial...
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
                            pokemon1move = []
                            if (battle.request_json).get('active','False')[0].get('moves','False')[0].get('disabled','False') == False:
                                pokemon1move.append(1)
                            if (battle.request_json).get('active','False')[0].get('moves','False')[1].get('disabled','False') == False:
                                pokemon1move.append(2)
                            if (battle.request_json).get('active','False')[0].get('moves','False')[2].get('disabled','False') == False:
                                pokemon1move.append(3)
                            if (battle.request_json).get('active','False')[0].get('moves','False')[3].get('disabled','False') == False:
                                pokemon1move.append(4)
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
                                switch3F = battle.user.reserved[0].fainted
                                switch4F = battle.user.reserved[1].fainted
                                if switch3F == False and switch4F == True:
                                    # Solo Pokémon 3 puede reemplazar 
                                    nMoveSwitch1 = 3
                                elif switch3F == True and switch4F == False:
                                    # Solo Pokémon 4 puede reemplazar
                                    nMoveSwitch1 = 4
                                else:
                                    # Pokémon 3 y 4 pueden reemplazar
                                    nMoveSwitch1 = random.choice([3, 4])
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
                                switch3F = battle.user.reserved[0].fainted
                                switch4F = battle.user.reserved[1].fainted
                                if switch3F == False and switch4F == True:
                                    # Solo Pokémon 3 puede reemplazar 
                                    nMoveSwitch1 = 3
                                elif switch3F == True and switch4F == False:
                                    # Solo Pokémon 4 puede reemplazar
                                    nMoveSwitch1 = 4
                                else:
                                    # Pokémon 3 y 4 pueden reemplazar
                                    nMoveSwitch1 = random.choice([3, 4])
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
                            pokemon2move = []
                            if (battle.request_json).get('active','False')[1].get('moves','False')[0].get('disabled','False') == False:
                                pokemon2move.append(1)
                            if (battle.request_json).get('active','False')[1].get('moves','False')[1].get('disabled','False') == False:
                                pokemon2move.append(2)
                            if (battle.request_json).get('active','False')[1].get('moves','False')[2].get('disabled','False') == False:
                                pokemon2move.append(3)
                            if (battle.request_json).get('active','False')[1].get('moves','False')[3].get('disabled','False') == False:
                                pokemon2move.append(4)
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
                                switch3F = battle.user.reserved[0].fainted
                                switch4F = battle.user.reserved[1].fainted
                                if switch3F == False and switch4F == True:
                                    # Solo Pokémon 3 puede reemplazar 
                                    nMoveSwitch2 = 3
                                elif switch3F == True and switch4F == False:
                                    # Solo Pokémon 4 puede reemplazar
                                    nMoveSwitch2 = 4
                                else:
                                    # Pokémon 3 y 4 pueden reemplazar
                                    nMoveSwitch2 = random.choice([3, 4])
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
                                switch3F = battle.user.reserved[0].fainted
                                switch4F = battle.user.reserved[1].fainted
                                if switch3F == False and switch4F == True:
                                    # Solo Pokémon 3 puede reemplazar
                                    nMoveSwitch2 = 3
                                elif switch3F == True and switch4F == False:
                                    # Solo Pokémon 4 puede reemplazar
                                    nMoveSwitch2 = 4
                                else:
                                    # Pokémon 3 y 4 pueden reemplazar
                                    if nMoveSwitch1 == 0:
                                        nMoveSwitch2 = random.choice([3, 4])
                                    else:
                                        if nMoveSwitch1 == 4:
                                            nMoveSwitch2 = 3
                                        else:
                                            nMoveSwitch2 = 4
                else:
                    moveSwitch2 = 'pass'
                    nMoveSwitch2 = 0
                    dynamax2 = 0
                    toMove2 = 0
            # Se define los Text
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
            return battle, moveSwitch1, int(nMoveSwitch1), dynamax1, int(toMove1), moveSwitch2, int(nMoveSwitch2), dynamax2, int(toMove2)