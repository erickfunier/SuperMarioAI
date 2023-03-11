import numpy as np

'''
Extração de atributos da memória RAM do jogo Super Mario World
Informações retiradas de: https://www.smwcentral.net/?p=nmap&m=smwram
'''
def getXY(ram):
    '''
    getXY(ram): retorna informação da posição do agente
    embora layer1? não seja utilizada no momento, pode ser útil em algumas 
    alterações do algoritmo de aprendizado.
    '''
    
    # Coordenadas x, y em relação a fase inteira 
    # Elas estão armazenadas em 2 bytes cada
    # no formato little endian
    marioX = ram[0x95]*256 + ram[0x94]
    marioY = ram[0x97]*256 + ram[0x96]
    
    # Coordenada da parte visível do site
    layer1x = ram[0x1B]*256 + ram[0x1A]
    layer1y = ram[0x1D]*256 + ram[0x1C]
    
    return marioX.astype(np.int16), marioY.astype(np.int16), layer1x.astype(np.int16), layer1y.astype(np.int16)
    
def getRam(env):
    return np.array(list(env.data.memory.blocks[8257536]))