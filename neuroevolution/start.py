from __future__ import print_function

import os
import sys
import math
import json
import neat
import time
import queue
import pickle
import visualize
import traceback
import multiprocessing


from os import path
from pathlib import Path
from threading import Thread
from datetime import datetime
from run import AsyncDownload

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

DICCIONARIO = json.loads(open("API/JSONs/movesANN.json","r").read())
USERS = [["Neuro1UNeat","Neuro1ONeat"],["Neuro2UNeat","Neuro2ONeat"],["Neuro3UNeat","Neuro3ONeat"],["Neuro4ONeat","Neuro4UNeat"],["Neuro5ONeat","Neuro5UNeat"],["Neuro6ONeat","Neuro6UNeat"]]
TEAMS = ["Eduardo Cunha","Guillermo Castilla","James Baek","Paul Chua","Calvin Foster","Meaghan Rattle","Davide Carrer","David Koutesh"]
STATE = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 64.0, 187.0, 141.0, 106.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 144.0, 146.0, 222.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 167.0, 126.0, 179.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 158.0, 182.0, 180.0, 209.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 141.0, 366.0, 155.0, 179.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.75, 0.0, 0.0, 1.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 126.0, 238.0, 176.0, 222.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.75, 0.0, 0.0, 1.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def eval_genome(genome, config, iterador, contador, q1, q2, q50, qm50, q150, listas):
    # Se crea la red neuronal en base al genoma
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # Si es la primera interacción... se inicia una batalla
    if iterador[(genome.id)%len(USERS)] == 0:
        # Parametros Batalla
        userUsername = USERS[(genome.id)%len(USERS)][0]
        opponentUsername = USERS[(genome.id)%len(USERS)][1]
        userTeamname = "Training Team"
        # Se determina el equipo del oponente
        opponentTeamname = TEAMS[contador[(genome.id)%len(USERS)]%len(TEAMS)]
        # Inicio Batalla - Agente Random
        opponentThread = None
        opponentThread = Thread(target=AsyncDownload, args=(q2[(genome.id)%len(USERS)],q2[(genome.id)%len(USERS)],q2[(genome.id)%len(USERS)],q2[(genome.id)%len(USERS)],opponentUsername,"123456",opponentTeamname,"ACCEPT_CHALLENGE",userUsername,True,False,1,))
        opponentThread.start()
        #print("OPONENTE INICIADO: {}".format(genome.id))
        # Inicio Batalla - Agente Inteligente
        bucle1 = True
        while bucle1:
            bucle2 = True
            #print("INICIANDO USUARIO...: {}".format(genome.id))
            userThread = None
            userThread = Thread(target=AsyncDownload, args=(q1[(genome.id)%len(USERS)],q50[(genome.id)%len(USERS)],qm50[(genome.id)%len(USERS)],q150[(genome.id)%len(USERS)],userUsername,"123456",userTeamname,"CHALLENGE_USER",opponentUsername,True,True,1,))
            userThread.start()
            # ¿Recibo confirmación del oponente que he comenzado la batalla?
            while bucle2:
                if q2[(genome.id)%len(USERS)].qsize() > 0:
                    cola = q2[(genome.id)%len(USERS)].get(0)
                    if cola[0] == 0:
                        #print("USUARIO INICIADO: {}".format(genome.id))
                        qm50[(genome.id)%len(USERS)].put([0])
                        bucle1 = False
                        bucle2 = False
                    elif cola[0] == -1:
                        qm50[(genome.id)%len(USERS)].put([-1])
                        bucle2 = False
                    else:
                        ...
    # Agente Inteligente
    reward1 = 0
    reward2 = 0
    action_value = 0
    while True:
        if q1[(genome.id)%len(USERS)].qsize() > 0:
            cola = q1[(genome.id)%len(USERS)].get(0)
            if cola[0] == 10:
                # Obtengo el estado actual del entorno
                state = cola[1]
                # Obtengo el reward anterior
                reward1 = cola[4]
                # Envío el indicativo de que no requiero una acción aleatoria
                q50[(genome.id)%len(USERS)].put([50,False])
                # Obtengo la salida de la red neuronal
                action_values = net.activate(state)
                # Acciones Especiales - Como no recibo un forceswitch... si el pokémon está debilitado, corto la acción elegida por la red neuronal como 'pass'
                # Obtengo el indice de la acción
                action_value = int(action_values.index(max(action_values)))
                # Y el mensaje de la acción
                action = '/choose ' + DICCIONARIO[str(action_value)]
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
                # Envío la acción en formato de mensaje
                q150[(genome.id)%len(USERS)].put([150,action])
                # Recibo la recompensa obtenida por la acción
                while True:
                    if q1[(genome.id)%len(USERS)].qsize() > 0:
                        cola = q1[(genome.id)%len(USERS)].get(0)
                        # Si cola[1] es -10, recibo reward de la última interacción y FIN de esa batalla
                        if cola[0] == -10:
                            #print("TERMINADO: {}".format(genome.id))
                            reward2 = cola[4]
                            # Archivos de Registro - WinLoss$.txt
                            numBattle =str(contador[(genome.id)%len(USERS)]).zfill(5)
                            indexOT = str(contador[(genome.id)%len(USERS)]%len(TEAMS))
                            scoreBatalla = reward2
                            win, loss = 0, 0
                            if int(scoreBatalla) >= 0:
                                win = str(1)
                            else:
                                loss = str(1)
                            if int(scoreBatalla) >= 0:
                                scoreBatalla = str(scoreBatalla).zfill(6)
                                scoreBatalla = "+{}".format(scoreBatalla)
                            else:
                                scoreBatalla = str(scoreBatalla).zfill(7)
                            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                            rutaWinLoss = "./neuroevolution/reportes/WinLoss/" + str((genome.id)%len(USERS)) + ".txt"
                            with open(rutaWinLoss, 'a') as WLScore:
                                WLScore.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(numBattle, win, loss, indexOT, scoreBatalla, dt_string))
                            # Archivos de Registro - Score$.txt
                            rutaScore = "./neuroevolution/reportes/Score/" + str((genome.id)%len(USERS)) + ".txt"
                            with open(rutaScore, 'a') as fScore:
                                fScore.write("{}\n".format(scoreBatalla))
                            # Archivos de Registro - InvalidActions$.txt
                            perInvalidActions = cola[5]
                            rutaInvalidActions = "./neuroevolution/reportes/InvalidActions/" + str((genome.id)%len(USERS)) + ".txt"
                            with open(rutaInvalidActions, 'a') as IAScore:
                                IAScore.write("{}\n".format(perInvalidActions))
                            # Fin de Batalla - Actualización de Índices
                            iterador[(genome.id)%len(USERS)] = 0
                            contador[(genome.id)%len(USERS)] += 1
                        else:
                            q1[(genome.id)%len(USERS)].put(cola)
                            # Actualización de Índices
                            iterador[(genome.id)%len(USERS)] += 1
                        break
            else:
                print("ERROR: SE HA RECIBIDO UN VALOR DISTINTO DE 10: {}".format(int(cola[0])))
            break
    # Se asigna el fitness
    if cola[3] == -9999:    # -9999 - Acción Invalida
        genome.fitness = cola[3] * 1.0
    else:
        genome.fitness = cola[3] * 1.0
    # listas
    listas[(genome.id)%len(USERS)].append(cola[3])
    # Save the winner.
    #titulo = "neuroevolution/modelos/0/winner" + str((genome.id)%len(USERS))
    #with open(titulo, 'wb') as f:
    #    pickle.dump(genome, f)
    # Reporte CMD
    #print("ANN: {} Iter: {} Index: {} R0: {} R1: {} R2: {} C3: {}".format((genome.key-1)%len(USERS),iterador[(genome.key-1)%len(USERS)],action_value, genome.fitness, reward1, reward2, cola[3]))
    print("TERMINADO - {}".format(genome.id))
    return genome.fitness


def inicio():
    # Se comparte los siguientes objetos entre los 8 procesos...
    q = multiprocessing.Manager()
    # Parámetros
    q1 = []
    q2 = []
    q50 = []
    qm50 = []
    q150 = []
    iterador = q.list()
    contador = q.list()
    listas = q.list()
    # Creación de colas
    q1_1 = q.Queue()
    q1_2 = q.Queue()
    q1_3 = q.Queue()
    q1_4 = q.Queue()
    q1_5 = q.Queue()
    q1_6 = q.Queue()
    q1_7 = q.Queue()
    q1_8 = q.Queue()
    # --------------
    q2_1 = q.Queue()
    q2_2 = q.Queue()
    q2_3 = q.Queue()
    q2_4 = q.Queue()
    q2_5 = q.Queue()
    q2_6 = q.Queue()
    q2_7 = q.Queue()
    q2_8 = q.Queue()
    # --------------
    q50_1 = q.Queue()
    q50_2 = q.Queue()
    q50_3 = q.Queue()
    q50_4 = q.Queue()
    q50_5 = q.Queue()
    q50_6 = q.Queue()
    q50_7 = q.Queue()
    q50_8 = q.Queue()
    # --------------
    qm50_1 = q.Queue()
    qm50_2 = q.Queue()
    qm50_3 = q.Queue()
    qm50_4 = q.Queue()
    qm50_5 = q.Queue()
    qm50_6 = q.Queue()
    qm50_7 = q.Queue()
    qm50_8 = q.Queue()
    # --------------
    q150_1 = q.Queue()
    q150_2 = q.Queue()
    q150_3 = q.Queue()
    q150_4 = q.Queue()
    q150_5 = q.Queue()
    q150_6 = q.Queue()
    q150_7 = q.Queue()
    q150_8 = q.Queue()
    # Asignación
    q1.append(q1_1)
    q1.append(q1_2)
    q1.append(q1_3)
    q1.append(q1_4)
    q1.append(q1_5)
    q1.append(q1_6)
    q1.append(q1_7)
    q1.append(q1_8)
    q2.append(q2_1)
    q2.append(q2_2)
    q2.append(q2_3)
    q2.append(q2_4)
    q2.append(q2_5)
    q2.append(q2_6)
    q2.append(q2_7)
    q2.append(q2_8)
    q50.append(q50_1)
    q50.append(q50_2)
    q50.append(q50_3)
    q50.append(q50_4)
    q50.append(q50_5)
    q50.append(q50_6)
    q50.append(q50_7)
    q50.append(q50_8)
    qm50.append(qm50_1)
    qm50.append(qm50_2)
    qm50.append(qm50_3)
    qm50.append(qm50_4)
    qm50.append(qm50_5)
    qm50.append(qm50_6)
    qm50.append(qm50_7)
    qm50.append(qm50_8)
    q150.append(q150_1)
    q150.append(q150_2)
    q150.append(q150_3)
    q150.append(q150_4)
    q150.append(q150_5)
    q150.append(q150_6)
    q150.append(q150_7)
    q150.append(q150_8)
    lista0 = q.list()
    lista1 = q.list()
    lista2 = q.list()
    lista3 = q.list()
    lista4 = q.list()
    lista5 = q.list()
    listas.append(lista0)
    listas.append(lista1)
    listas.append(lista2)
    listas.append(lista3)
    listas.append(lista4)
    listas.append(lista5)
    print("listas: {}".format(listas))
    iterador.append(0)
    iterador.append(0)
    iterador.append(0)
    iterador.append(0)
    iterador.append(0)
    iterador.append(0)
    iterador.append(0)
    iterador.append(0)
    contador.append(0)
    contador.append(0)
    contador.append(0)
    contador.append(0)
    contador.append(0)
    contador.append(0)
    contador.append(0)
    contador.append(0)
    # Neuroevolución
    config_path = "./neuroevolution/config-feedforward"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    # Restauración
    p = neat.Checkpointer.restore_checkpoint('./neuroevolution/modelos/neat-checkpoint-28500')
    # Se agrega un reporte cuyo progreso será mostrado en el terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, 25, './neuroevolution/modelos/neat-checkpoint-'))     # Cada 20 iteraciones
    # Se evaluará de manera paralela los genomas
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome, iterador, contador, q1, q2, q50, qm50, q150, listas)
    # Se evalúan las redes neuronales en 50 cantidad de interacciones, luego se agrupa las gráficas
    try:
        winner = p.run(pe.evaluate, 55)
    except:
        traceback.print_exc()
    print("--- FIN DEL PROGRAMA ---")
    # Save the winner.
    titulo = "neuroevolution/modelos/winner"
    with open(titulo, 'wb') as f:
        pickle.dump(winner, f)
    # Se elabora las gráficas correspondientes al entrenamiento
    # visualize.draw_net(config, winner, True)  # dibuja la topologia ganadora
    #visualize.plot_stats(stats, ylog=False, view=True,filename='./neuroevolution/reportes/avg_fitness.svg')         # visualiza estadisticas
    #visualize.plot_species(stats, view=True, filename='./neuroevolution/reportes/speciation.svg')                   # visualiza especies
    # Prueba pequeña:
    netWinner = neat.nn.FeedForwardNetwork.create(winner, config)
    action_values = netWinner.activate(STATE)
    print("action_value: {}".format(int(action_values.index(max(action_values)))))
    print("0: {}".format(listas[0]))
    print("1: {}".format(listas[1]))
    print("2: {}".format(listas[2]))
    print("3: {}".format(listas[3]))
    print("4: {}".format(listas[4]))
    print("5: {}".format(listas[5]))
    # Luego de culminar toda la iteración, se muestra el genoma ganador...
    #print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    inicio()