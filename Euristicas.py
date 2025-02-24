import math

def heuristica_manhattan(nodo_actual, nodo_objetivo):
    x1, y1 = nodo_actual
    x2, y2 = nodo_objetivo
    return abs(x1 - x2) + abs(y1 - y2)

def heuristica_euclidiana(nodo_actual, nodo_objetivo):
    x1, y1 = nodo_actual
    x2, y2 = nodo_objetivo
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
