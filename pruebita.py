from __future__ import print_function

import os
import sys
import json
import neat
import time
import queue
import pickle
import visualize
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
USERS = [["Neuro1UNeat","Neuro1ONeat"],["Neuro2UNeat","Neuro2ONeat"],["Neuro3UNeat","Neuro3ONeat"],["Neuro4UNeat","Neuro4ONeat"],["Neuro5UNeat","Neuro5ONeat"],["Neuro6UNeat","Neuro6ONeat"],["Neuro7UNeat","Neuro7ONeat"],["Neuro8UNeat","Neuro8ONeat"]]
TEAMS = ["Eduardo Cunha","Guillermo Castilla"]
STATE = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 64.0, 187.0, 141.0, 106.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 144.0, 146.0, 222.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 167.0, 126.0, 179.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 158.0, 182.0, 180.0, 209.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 141.0, 366.0, 155.0, 179.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.75, 0.0, 0.0, 1.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 126.0, 238.0, 176.0, 222.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.75, 0.0, 0.0, 1.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]

def eval_genome(genome, config, iterador, contador, q1, q2, q50, qm50, q150):
    # Se crea la red neuronal en base al genoma
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    action_values = net.activate(STATE)
    action_value = int(action_values.index(max(action_values)))
    #action = '/choose ' + DICCIONARIO[str(action_value)]
    #print("ANN: {} Index: {}".format((genome.key-1)%len(USERS),action_value))
    if action_value == 10 or action_value == 11 or action_value == 12 or action_value == 13 or action_value == 14 or action_value == 15 or action_value == 16 or action_value == 17 or action_value == 18 or action_value == 19 or action_value == 20 or action_value == 21:
        genome.fitness = 100.0
        print("--- LOGRADO ---")
    else:
        genome.fitness = -9999.0
    return genome.fitness


def inicio():
    # Neuroevolución
    config_path = "./neuroevolution/config-feedforward"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, 300, './neuroevolution/modelos/neat-checkpoint-'))
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome, None, None, None, None, None, None, None)
    # Se evalúan las redes neuronales en X cantidad de interacciones
    winner = p.run(pe.evaluate, 50)
    # ---
    p = neat.Checkpointer.restore_checkpoint('./neuroevolution/modelos/neat-checkpoint-49')
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(6, 300, './neuroevolution/modelos/neat-checkpoint-'))
    winner = p.run(pe.evaluate, 60)
    # ---
    #visualize.draw_net(config, winner, True)  # dibuja la topologia ganadora
    visualize.plot_stats(stats, ylog=False, view=True)  # visualiza estadisticas
    visualize.plot_species(stats, view=True)   # visualiza especies
    # ---
    netWinner = neat.nn.FeedForwardNetwork.create(winner, config)
    action_values = netWinner.activate(STATE)
    print("NEAT FIN")
    #print("action_values: {}".format(action_values))
    print("action_value: {}".format(int(action_values.index(max(action_values)))))
    # Luego de culminar toda la iteración, se muestra el genoma ganador...
    #print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    inicio()