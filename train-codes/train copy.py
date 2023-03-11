# https://neat-python.readthedocs.io/en/latest/
# https://gym.openai.com/docs/

import retro
import numpy as np
import cv2
import neat
import pickle
from rominfo import *

env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2', record='.')

generation = 539
highest = 0

# Se devemos mostrar a tela do jogo (+ lento) ou não (+ rápido)
show = True

# funtion to evaluate genomes during training process
def eval_genomes(genomes, config):

    global highest
    global generation
    generation += 1

    # log executions
    log = True
    logFlowControl = 250
    
    for genomeId, genome in genomes:
        # get environment print screen
        observation = env.reset()

        # set shape size to input in neural network
        inx, iny, cte = env.observation_space.shape
        inx = int(inx/8) #224/8 = 28
        iny = int(iny/8) #256/8 = 32    28*32 = 896 inputs for neat

        # create the neural network
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        # initialize variables
        done = False
        fitness = 0
        posx = 0
        lastPosition = 0
        stoppedIterations = 0
        iterations = 0

        imgarray = []

        # main loop
        while not done:
           
            # show gameplay
            if show:
                env.render()
            
            # prepare the print screen to use as neural network input
            observation = cv2.resize(observation, (inx, iny))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (inx, iny))

            imgarray = np.ndarray.flatten(observation)
            #print(imgarray)
            # process the print screen through the neural network to obtain the output (actions)
            nnOutput = net.activate(imgarray)
            #print(nnOutput)
            # apply actions to game environment to get new observation (print screen), reward gain by applying the action, and current values of info parameters (data.json)
            observation, reward, done, info = env.step(nnOutput)

            # get info
            x = getXY(getRam(env))[0]

            posx = x

            # reward for position
            if posx > lastPosition:
                lastPosition = posx
                fitness += 1
                stoppedIterations = 0
            else:
                stoppedIterations += 1

            if posx > 4970:
                fitness += 100000
                genome.fitness = fitness
                done = True                

            # counter to stop
            if stoppedIterations > 100:
                done = True

            iterations += 1
            # logs
            if log and iterations % logFlowControl == 0:
                print(  '\ngeneration: ', generation, 
                        '\nx-position:', posx,
                        '\ngenomeId: ', genomeId,
                        '\nfitness: ', fitness
                    )
            if done and log:
                print(  '\ngeneration: ', generation, 
                        '\nx position:', posx, 
                        '\ngenomeId: ', genomeId, 
                        '\nfitness: ', fitness, 
                        '\nMaximum iterations in stopped x position'
                    )

            # set genome fitness to train the neural network
            genome.fitness = fitness


# load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    'conf-super-mario-brother-neat')

# set configuration
population = neat.Population(config)
population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-539')

# report trainning
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(neat.Checkpointer(10)) #every x generations save a checkpoint

# run trainning
winner = population.run(eval_genomes)

# save the winner
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

# close environment
env.close()