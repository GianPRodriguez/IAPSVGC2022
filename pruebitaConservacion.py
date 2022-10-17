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
import random

from os import path
from threading import Thread
from datetime import datetime
from run import AsyncDownload

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)


def eval_genome(genome, config, iterador, contador, q1, q2, q50, qm50, q150):
    # Se crea la red neuronal en base al genoma
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    genome.fitness = random.randint(0, 9)
    print("key: {} - id: {} - fitness: {}".format(genome.key, genome.id, genome.fitness))
    return genome.fitness


def inicio():
    # Neuroevolución
    config_path = "./neuroevolution/config-feedforward"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome, None, None, None, None, None, None, None)
    winner = p.run(pe.evaluate, 3)
    print("NEAT FIN")
    print(winner)


if __name__ == "__main__":
    inicio()