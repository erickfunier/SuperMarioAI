import retro
import numpy as np
import cv2 
import neat
import pickle
from rominfo import *

env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2')

#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-539')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'conf-super-mario-brother-neat')

with open('winner.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)

imgarray = []

xpos_end = 0

p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)


ob = env.reset()

inx, iny, inc = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

current_max_fitness = 0
fitness_current = 0
frame = 0
counter = 0
xpos = 0
xpos_max = 0

done = False
# create a window to show the gameplay
#cv2.namedWindow('SonicTheHedgehog-Genesis | NEAT-Python | jubatistim', cv2.WINDOW_NORMAL)
#cv2.moveWindow("SonicTheHedgehog-Genesis | NEAT-Python | jubatistim", 950, 100)
#cv2.resizeWindow('SonicTheHedgehog-Genesis | NEAT-Python | jubatistim', 800,600)

while not done:
    
    # show gameplay
    #shwimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
    #cv2.imshow('SonicTheHedgehog-Genesis | NEAT-Python | jubatistim', shwimg)
    #cv2.waitKey(1)
    env.render()

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx,iny))

    imgarray = np.ndarray.flatten(ob)
    
    nnOutput = net.activate(imgarray)
    
    ob, rew, done, info = env.step(nnOutput)
    
    x = getXY(getRam(env))[0]
    xpos = x
    
    
    if xpos > xpos_max:
        fitness_current += 1
        xpos_max = xpos
        counter = 0
    else:
        counter +=1
    
    
    if xpos > 4970:
        fitness_current += 10000
        done = True
        
    if done or counter == 250:
        done = True
        #print(genome_id, fitness_current)