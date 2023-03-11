# https://neat-python.readthedocs.io/en/latest/
# https://neat-python.readthedocs.io/en/latest/xor_example.html
# https://gym.openai.com/docs/
# requirements
# opencv-python
# neat-python

import retro
import numpy as np
import cv2
import neat
import pickle
import os
from rominfo import *
from sys import argv # used for get filename passed as argument

env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2', players=1)

# bool to show gameplay
show = True
log = False

# funtion to evaluate genomes during training process
def eval_genomes(genomes, config):
    global generation
    global log

    # update generation
    generation += 1

    # log iterations flow
    logFlowControl = 250
    
    for genomeId, genome in genomes:
        # get screen matrix
        observation = env.reset()
        
        # get the size of screen and set size variables X and Y
        sizeX, sizeY, sizeZ = env.observation_space.shape
        sizeX = int(sizeX/8) # 224/8 = 28
        sizeY = int(sizeY/8) # 256/8 = 32    28*32 = 896 inputs for neat

        # create the neural network
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        # control variables
        done = False        # used to stop genome execution
        fitness = 0         # used to control fitness of current genome on neural network
        bestFitness = 0     # used to save the best genome of current generation
        posx = 0            # used to get store current Mario X position
        lastPosition = 0    # used to store last position and control stooped iterations to end stag genome
        stoppedIterations = 0
        iterations = 0

        while not done:
           
            # show gameplay
            if show:
                env.render()
            
            observation = cv2.resize(observation, (sizeX, sizeY))       # resize current game image matrix RGB
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY) # convert matrix RGB to GRAYSCALE
            observation = np.reshape(observation, (sizeX, sizeY))       # reorder matrix 32x28 to 28x32

            imgArray = np.ndarray.flatten(observation)          # serialize matrix vector
            imgArray = np.interp(imgArray, (0, 254), (-1, +1))  # standardize values 0,254 to -1,1

            actions = net.activate(imgArray)   # activate neural network with matrix and get actions from network

            observation, reward, done, info = env.step(actions) # run actions on game and get current status

            posx = getXY(getRam(env))[0]    # get Mario X Position using RAM value

            # update fitness if mario is in new x position, otherwise update stoppedIteration counter
            if posx > lastPosition:
                lastPosition = posx
                fitness += 1
                stoppedIterations = 0
            else:
                stoppedIterations += 1

            # if Mario X position reach end of level set fitness to 100000
            if posx > 4970:
                #fitness += 100000 # if want the first mario to end the level uncomment this line
                done = True                

            # stop genome execution if reach 100 stoppedIterations
            if stoppedIterations > 100:
                done = True

            iterations += 1 # update iteration counter

            # log based in flow control
            if log and iterations % logFlowControl == 0:
                print('\ngeneration: ', generation, 
                        '\nx-position:', posx,
                        '\ngenomeId: ', genomeId,
                        '\nfitness: ', fitness
                    )

            # log when done reach maximum iterations in same X position or end level
            if done and log:
                print('\ngeneration: ', generation, 
                        '\nx position:', posx, 
                        '\ngenomeId: ', genomeId, 
                        '\nfitness: ', fitness, 
                        '\nMaximum iterations in stopped x position'
                    )

            # set genome fitness to train the neural network
            genome.fitness = fitness

        # save the best genome of current generation
        if fitness > bestFitness:
            bestFitness = fitness
            with open('winner'+str(generation)+'.pkl', 'wb') as output:
                pickle.dump(genome, output, 1)

# get input parameters to set start generation and log flag
if (len(argv) == 2):
    generation = int(argv[1])
    log = False
elif (len(argv) == 3):
    generation = int(argv[1])
    if (argv[2] == '-L'):
        log = True
    else:
        log = False

# load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    'conf-super-mario-brother-neat')

# set configuration
population = neat.Population(config)
localDir = os.path.dirname(__file__)
if (generation > 0):
    checkPointPath = localDir + '/checkpoint/neat-checkpoint-'+ str(generation)
    fileExists = os.path.isfile(checkPointPath)
    if not fileExists:
        auxGen = generation
        while auxGen > 0:
            checkPointPath = localDir + '/checkpoint/neat-checkpoint-'+ str(auxGen)
            fileExists = os.path.isfile(checkPointPath)
            if fileExists:
                population = neat.Checkpointer.restore_checkpoint(checkPointPath)
                print('Starting with nearest checkpoint ', str(auxGen))
                
                break
            else:
                auxGen -= 1
        generation = auxGen
    else:
        population = neat.Checkpointer.restore_checkpoint(checkPointPath)

# report neural network status
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(neat.Checkpointer(5)) # every generation save a checkpoint

# evaluate genomes and get the winner
winner = population.run(eval_genomes)

# save the winner
with open('winner'+str(generation)+'.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

# close environment
env.close()