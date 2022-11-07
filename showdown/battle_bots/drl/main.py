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
from API.PokeAPI import getMoveColor
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
            with open("./neuroevolution/modelos/winner.pkl", "rb") as f:
                genome = pickle.load(f)
            # Archivo de Configuración
            config_path = "./neuroevolution/config-feedforward"
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
        if moveSwitch1 == 'move':
            text1 = await getMoveName(battle, 0, nMoveSwitch1-1, dynamax1)
            text1Aux = text1.lower().replace(' ', '-')
            color1 = getMoveColor(text1Aux)
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
            color2 = getMoveColor(text2Aux)
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
        # Se envía al Queue la acción
        deltaReward, reward = battle.calculateRewards()
        if battle.error:
            deltaReward, reward = -9999, reward
            battle.cantInvActions += 1
            battle.cantInvActionsSeg += 1
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