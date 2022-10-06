import os
import sys
import json
import neat
import time
import queue
import pickle
import traceback
import multiprocessing

from os import path
from threading import Thread
from datetime import datetime
from run import AsyncDownload

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

DICCIONARIO = json.loads(open("API/JSONs/movesANN.json","r").read())
USERS = [["Neuro1UNeat","Neuro1ONeat"],["Neuro2UNeat","Neuro2ONeat"]]
TEAMS = ["Eduardo Cunha","Guillermo Castilla"]


def eval_genome(genome, config, iterador, contador, q1, q2, q50, qm50, q150):
    # Se crea la red neuronal en base al genoma
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # Si es la primera interacción... se inicia una batalla
    if iterador[(genome.key-1)%len(USERS)] == 0:
        # Parametros Batalla
        userUsername = USERS[(genome.key-1)%len(USERS)][0]
        opponentUsername = USERS[(genome.key-1)%len(USERS)][1]
        userTeamname = "Eduardo Cunha"
        # Se determina el equipo del oponente
        opponentTeamname = TEAMS[contador[(genome.key-1)%len(USERS)]%len(TEAMS)]
        # Inicio Batalla - Agente Random
        opponentThread = Thread(target=AsyncDownload, args=(q2[(genome.key-1)%len(USERS)],q2[(genome.key-1)%len(USERS)],q2[(genome.key-1)%len(USERS)],q2[(genome.key-1)%len(USERS)],opponentUsername,"123456",opponentTeamname,"ACCEPT_CHALLENGE",userUsername,True,False,1,))
        opponentThread.start()
        time.sleep(0.3)
        # Inicio Batalla - Agente Inteligente
        userThread = Thread(target=AsyncDownload, args=(q1[(genome.key-1)%len(USERS)],q50[(genome.key-1)%len(USERS)],qm50[(genome.key-1)%len(USERS)],q150[(genome.key-1)%len(USERS)],userUsername,"123456","Eduardo Cunha","CHALLENGE_USER",opponentUsername,True,True,1,))
        userThread.start()
    # Agente Inteligente
    reward1 = 1
    reward2 = 1
    while True:
        if q1[(genome.key-1)%len(USERS)].qsize() > 0:
            cola = q1[(genome.key-1)%len(USERS)].get(0)
            if cola[0] == 10:
                # Obtengo el estado actual del entorno
                state = cola[1]
                # Obtengo el reward anterior
                reward1 = cola[3]
                # Envío el indicativo de que no requiero una acción aleatoria
                q50[(genome.key-1)%len(USERS)].put([50,False])
                # Obtengo la salida de la red neuronal
                action_values = net.activate(state)
                print("action_values: {}".format(action_values))
                # Obtengo el indice de la acción
                action_value = int(action_values.index(max(action_values)))
                # VALIDACIÓN
                action = '/choose ' + DICCIONARIO[str(action_value)]
                print("ANN: {} Index: {} Action: {}".format((genome.key-1)%len(USERS),action_value,action))
                # Envío la acción en formato de mensaje
                q150[(genome.key-1)%len(USERS)].put([150,action])
                # Recibo la recompensa obtenida por la acción
                while True:
                    if q1[(genome.key-1)%len(USERS)].qsize() > 0:
                        cola = q1[(genome.key-1)%len(USERS)].get(0)
                        reward2 = cola[3]
                        q1[(genome.key-1)%len(USERS)].put(cola)
                        break
                # Aumento en indices de iteraciones
                iterador[(genome.key-1)%len(USERS)] += 1
            else:
                # Fin de Batalla - se reinicia los indices
                iterador[(genome.key-1)%len(USERS)] = 0
                contador[(genome.key-1)%len(USERS)] += 1
            break
    # Se asigna el fitness
    if reward2 == -9999:
        genome.fitness = reward2 * 1.0
    else:
        genome.fitness = (reward2-reward1)*100.0/(abs(reward1))
    # Save the winner.
    titulo = "neuroevolution/modelos/0/winner" + str((genome.key-1)%len(USERS))
    with open(titulo, 'wb') as f:
        pickle.dump(genome, f)
    return genome.fitness


def inicio():
    # Parámetros
    q = multiprocessing.Manager()
    q1 = []
    q2 = []
    q50 = []
    qm50 = []
    q150 = []
    iterador = q.list()
    contador = q.list()
    # Creación de colas
    q1_1 = q.Queue()
    q1_2 = q.Queue()
    q2_1 = q.Queue()
    q2_2 = q.Queue()
    q50_1 = q.Queue()
    q50_2 = q.Queue()
    qm50_1 = q.Queue()
    qm50_2 = q.Queue()
    q150_1 = q.Queue()
    q150_2 = q.Queue()
    # Asignación
    q1.append(q1_1)
    q1.append(q1_2)
    q2.append(q2_1)
    q2.append(q2_2)
    q50.append(q50_1)
    q50.append(q50_2)
    qm50.append(qm50_1)
    qm50.append(qm50_2)
    q150.append(q150_1)
    q150.append(q150_2)
    iterador.append(0)
    iterador.append(0)
    contador.append(0)
    contador.append(0)
    # Neuroevolución
    config_path = "./neuroevolution/config-feedforward"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome, iterador, contador, q1, q2, q50, qm50, q150)
    # Se evalúan las redes neuronales en X cantidad de interacciones
    try:
        winner = p.run(pe.evaluate, 100)
    except:
        traceback.print_exc()
    print("NEAT FIN")
    # Luego de culminar toda la iteración, se muestra el genoma ganador...
    #print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    inicio()