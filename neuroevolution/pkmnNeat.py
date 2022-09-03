import neat
import multiprocessing

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # Completar
    genome.fitness = 4.0
    # Cada evaluación es con un equipo diferente, el fitness es el fitness final
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        genome.fitness -= (output[0] - xo[0]) ** 2
    return genome.fitness

if __name__ == '__main__':
    # Configuración del NEAT
    config_path = "./config-feedforward"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    # Creación de la población
    p = neat.Population(config)
    # Run NEAT
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, 21600)  #172800
    # ¿Cómo guardar los mejores luego de n corridas?

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))