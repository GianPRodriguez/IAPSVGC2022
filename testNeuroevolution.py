import neat
import multiprocessing
import time
from datetime import datetime
from run import AsyncDownload
from os import path

xor_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
xor_outputs = [[0.0], [1.0], [1.0], [0.0]]
USERS = [["PopplioPeru98","mallow123"],["Delta VGC 2022","Mallow"]]

def eval_genome(genome, config):
    #print("genome.key: {}".format(genome.key))
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    print("genome.key: {}".format(genome.key))
    # Parametros Batalla
    #q = multiprocessing.Queue()
    #username = USERS[genome.key-1][0]
    #password = USERS[genome.key-1][1]
    #teamname = 'Eduardo Cunha'
    #botmode = "CHALLENGE_USER"
    #opponentUsername = "KimJongUnVGC2020"
    # Reporte
    #nameFile = str(genome.key-1) + '.txt'
    #if path.exists(nameFile):
    #    with open(nameFile, "r") as file:
    #        last_line = file.readlines()[-1]
    #        numBattle = str(int(last_line.split('\t')[0])+1).zfill(5)
    #    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    #    with open(nameFile, 'a') as outfile:
    #        outfile.write("{}\t{}\txx\t{}\n".format(numBattle, genome.key-1, dt_string))
    #else:
    #    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    #    with open(nameFile, 'w') as outfile:
    #        outfile.write("00001\t{}\txx\t{}\n".format(genome.key-1, dt_string))
    # Inicio Batalla
    #AsyncDownload(q,username,password,teamname,botmode,opponentUsername,True,1)
    # Completar
    genome.fitness = 4.0
    # Cada evaluación es con un equipo diferente, el fitness es el fitness final
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        genome.fitness -= (output[0] - xo[0]) ** 2
    return genome.fitness

if __name__ == '__main__':
    # Configuración del NEAT
    config_path = "./neuroevolution/config-feedforward"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    # Creación de la población
    p = neat.Population(config)
    # Run NEAT
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, 6)  #172800
    # ¿Cómo guardar los mejores luego de n corridas?

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))