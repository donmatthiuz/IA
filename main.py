import numpy as np
import random
import csv
from Agente import *


# Definir el entorno

# Semilla para replicar el entorno

np.random.seed(122)
random.seed(122)

dimension = 4
num_hoyos = random.randint(1, 3)

posiciones_inicio = [(0, 0), (0, 3), (3, 0), (3, 3)]
posiciones_meta = [(3, 3), (3, 0), (0, 3), (0, 0)]

indice = random.randint(0, 3)
inicio = posiciones_inicio[indice]
meta = posiciones_meta[indice]

# Generar hoyos aleatorios
posiciones_posibles = [(i, j) for i in range(dimension) for j in range(dimension) if (i, j) != inicio and (i, j) != meta]
hoyos = random.sample(posiciones_posibles, num_hoyos)

def imprimir_entorno():
    entorno = [['O' for _ in range(dimension)] for _ in range(dimension)]
    entorno[inicio[0]][inicio[1]] = 'I'  # Inicio
    entorno[meta[0]][meta[1]] = 'M'  # Meta
    for h in hoyos:
        entorno[h[0]][h[1]] = 'X'  # Hoyos
    
    # Imprimir el entorno en consola
    for fila in entorno:
        print(" ".join(fila))
    
    # Guardar el entorno en un archivo CSV
    with open('frozen_lake.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(entorno)

print("Frozen Lake:")
imprimir_entorno()




## Ejecutamos el agente

for i in range(0,20000):

    ex = FrozenLake(filename='.\\frozen_lake.csv')

    policy = ex.generate_random_policy()
    print("Política aleatoria generada:", policy)


    reward, state_history = ex.execute_policy(policy)
    print("Recompensa obtenida:", reward)

    if reward == 1:
        ex.plot_grid(state_history)