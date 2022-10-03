import neat
import os
import sys
import time
import queue
import json
import neuroevolution.config
import multiprocessing
from multiprocessing import Manager
from datetime import datetime
from os import path
from threading import Thread

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from run import AsyncDownload  

diccionario = json.loads(open("API/JSONs/movesANN.json","r").read())
USERS = [["Neuro1UNeat","Neuro1ONeat"],["Neuro2UNeat","Neuro2ONeat"]]
TEAMS = ["Eduardo Cunha","Guillermo Castilla"]


def eval_genome(genome, config):
    # Se crea la red neuronal en base al genoma
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # Si es el primer turno de la batalla...
    if neuroevolution.config.iterAgent0 == 0 and neuroevolution.config.iterAgent1 == 0:
        # Parametros Batalla
        userUsername = USERS[(genome.key-1)%len(USERS)][0]
        opponentUsername = USERS[(genome.key-1)%len(USERS)][1]
        userTeamname = "Eduardo Cunha"
        # Se determina el equipo del oponente
        if (genome.key-1)%len(USERS) == 0:
            opponentTeamname = TEAMS[neuroevolution.config.contBattles0%len(TEAMS)]
        else:
            opponentTeamname = TEAMS[neuroevolution.config.contBattles1%len(TEAMS)]
        # Colas
        q1 = queue.Queue()
        q2 = queue.Queue()
        q50 = queue.Queue()
        qm50 = queue.Queue()
        q150 = queue.Queue()
        # Inicio Batalla - Agente Random
        opponentThread = Thread(target=AsyncDownload, args=(q2,q2,q2,q2,opponentUsername,"123456",opponentTeamname,"ACCEPT_CHALLENGE",userUsername,True,False,1,))
        opponentThread.start()
        time.sleep(0.2)
        # Inicio Batalla - Agente Inteligente
        userThread = Thread(target=AsyncDownload, args=(q1,q50,qm50,q150,userUsername,"123456","Eduardo Cunha","CHALLENGE_USER",opponentUsername,True,True,1,))
        userThread.start()
    else:
        # Colas
        q1 = neuroevolution.config.q1
        q2 = neuroevolution.config.q2
        q50 = neuroevolution.config.q50
        qm50 = neuroevolution.config.qm50
        q150 = neuroevolution.config.q150

    # Agente Inteligente
    while True:
        try:
            cola = q1.get(0)
            if cola[0] == 10:
                # Recibo el array del estado actual del entorno
                state = cola[1]
                # Si es entrenamiento... para este caso siempre lo será
                if cola[5]:
                    # Envío el indicativo de que no requiero una acción aleatoria
                    q50.put([50,False])
                    # Obtengo el arreglo correspondiente a las acciones
                    action_values = net.activate(state)
                    # Obtengo el indice de la acción de acuerdo al estado actual del entorno
                    action_value = int(action_values.index(max(action_values)))
                    print("action_value: {}".format(action_value))
                    # Obtengo la acción en string
                    action = '/choose ' + diccionario[str(action_value)]
                    # Lo imprimo más el indice
                    print("action: {} index: {}".format(action, action_value))
                    # Envío la acción en formato de mensaje
                    q150.put([150,action])
                    # Recibo recompensa obtenida por la acción
                    while True:
                        try:
                            cola = q1.get(0)
                            #print("cola: {}".format(cola))
                            if cola[0] == 10:
                                score = cola[4]
                                q1.put(cola)
                            elif cola[0] == -10:
                                score = cola[4]
                                q1.put(cola)
                            else:
                                print("ERROR - EN SEGUNDO TURNO NO DEBE HABER MAS OPCIÓN")
                            print("break")
                            break
                        except:
                            ...
                    # score = ...
                    # Aumento en indices de iteraciones
                    if (genome.key-1)%len(USERS) == 0:
                        neuroevolution.config.iterAgent0 = neuroevolution.config.iterAgent0 + 1
                        print("neuroevolution.config.iterAgent0: {}".format(neuroevolution.config.iterAgent0))
                    else:
                        neuroevolution.config.iterAgent1 = neuroevolution.config.iterAgent1 + 1
                        print("neuroevolution.config.iterAgent1: {}".format(neuroevolution.config.iterAgent1))
                    break
                else:
                    print("ERROR - SOLICITA ACCIÓN RANDOM")
            elif cola[0] == -10 or cola[0] == -1:
                print("fin de batalla")
                # Fin de Batalla - se reinicia los indices
                if (genome.key-1)%len(USERS) == 0:
                    neuroevolution.config.iterAgent0 = 0
                    neuroevolution.config.contBattles0 = neuroevolution.config.contBattles0 + 1
                else:
                    neuroevolution.config.iterAgent1 = 0
                    neuroevolution.config.contBattles1 = neuroevolution.config.contBattles1 + 1
                break
        except:
            ...
    # Se asigna el fitness
    genome.fitness = score
    print(genome.fitness)
    return genome.fitness


def inicio():
    config_path = "./neuroevolution/config-feedforward"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, 3)
    print("terminado")
    # Luego de culminar toda la iteración, se muestra el genoma ganador...
    #print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    inicio()