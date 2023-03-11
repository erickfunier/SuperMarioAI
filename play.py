import retro
import numpy as np
import cv2 
import neat
import pickle
from rominfo import *
from sys import argv # used for get filename passed as argument

env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2', players=1)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'conf-super-mario-brother-neat')


if (len(argv) == 2):
    with open('winner'+argv[1]+'.pkl', 'rb') as input_file:
        genome = pickle.load(input_file)
else:
    with open('winner.pkl', 'rb') as input_file:
        genome = pickle.load(input_file)


population = neat.Population(config)

# get screen matrix
observation = env.reset()

# get the size of screen and set size variables X and Y
sizeX, sizeY, sizeZ = env.observation_space.shape
sizeX = int(sizeX/8) # 224/8 = 28
sizeY = int(sizeY/8) # 256/8 = 32   28*32 = 896 inputs for neat

# create the neural network
net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

# control variables
done = False        # used to stop genome execution
fitness = 0         # used to control fitness of current genome on neural network
posx = 0            # used to get store current Mario X position
lastPosition = 0    # used to store last position and control stooped iterations to end stag genome
stoppedIterations = 0
iterations = 0

while not done:
    
    # show gameplay
    env.render()

    observation = cv2.resize(observation, (sizeX, sizeY))       # resize current game image matrix RGB
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY) # convert matrix RGB to GRAYSCALE
    observation = np.reshape(observation, (sizeX,sizeY))        # reorder matrix 32x28 to 28x32

    imgArray = np.ndarray.flatten(observation)          # serialize matrix vector
    imgArray = np.interp(imgArray, (0, 254), (-1, +1))  # standardize values 0,254 to -1,1
    
    actions = net.activate(imgArray)    # activate neural network with matrix and get actions from network
    
    observation, reward, done, info = env.step(actions) # run actions on game and get current status
    
    posx = getXY(getRam(env))[0]    # get Mario X Position using RAM value
    
    # update fitness if mario is in new x position, otherwise update stoppedIteration counter
    if posx > lastPosition:
        fitness += 1
        lastPosition = posx
        stoppedIterations = 0
    else:
        stoppedIterations += 1
    
    # if Mario X position reach end of level set fitness to 100000
    if posx > 4970:
        fitness += 100000
        done = True

    # stop genome execution if reach 100 stoppedIterations
    if stoppedIterations > 100:
        done = True

    iterations += 1 # update iteration counter