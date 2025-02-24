import numpy as np
import matplotlib.pyplot as plt
def dibujar_camino(matriz, camino, factor_escala=8):
    colores = {
        0: [255, 255, 255],  
        1: [0, 0, 0],        
        2: [255, 0, 0],      
        3: [0, 255, 0],     
        "camino": [254, 0, 220] 
    }

    imagen = np.zeros((matriz.shape[0], matriz.shape[1], 3), dtype=np.uint8)

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            imagen[i, j] = colores[matriz[i, j]]

    
    for (x, y) in camino:
        imagen[x, y] = colores["camino"]

    plt.figure(figsize=(matriz.shape[1] // factor_escala, matriz.shape[0] // factor_escala))
    plt.imshow(imagen, interpolation="nearest")
    plt.axis("off")
    plt.show()