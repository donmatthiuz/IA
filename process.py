import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
from abc import ABC, abstractmethod
from graphic import dibujar_camino
from Euristicas import heuristica_euclidiana, heuristica_manhattan
import heapq 

class SearchProblem(ABC):
    @abstractmethod
    def get_initial_state(self):
        pass

    @abstractmethod
    def actions(self, state):
        pass

    @abstractmethod
    def step_cost(self, state, action, next_state):
        pass

    @abstractmethod
    def goal_test(self, state):
        pass

class MazeProblem(SearchProblem):

    def __init__(self, matriz, start, goals):
        self.matriz = matriz
        self.start = start
        self.goals = set(goals)
        self.rows = len(matriz)
        self.cols = len(matriz[0])

    def get_initial_state(self):
        return self.start

    def actions(self, state):
        x, y = state
        possible_moves = [
            ("UP", (x - 1, y)),
            ("DOWN", (x + 1, y)),
            ("LEFT", (x, y - 1)),
            ("RIGHT", (x, y + 1))
        ]
        
        valid_moves = []
        for action, (nx, ny) in possible_moves:
            if 0 <= nx < self.rows and 0 <= ny < self.cols and self.matriz[nx][ny] != 1:
                valid_moves.append(action)

        return valid_moves

    def step_cost(self, state, action, next_state):
        return 1  

    def goal_test(self, state):
        return state in self.goals
    
class GraphSearch:
    
    def __init__(self, problem, heuristica = None):
        self.problem = problem
        self.heuristica = heuristica

    def breadth_first_search(self):
        start = self.problem.get_initial_state()
        queue = deque([(start, [])])  
        visited = set()

        while queue:
            state, path = queue.popleft()

            
            if self.problem.goal_test(state):
                return path + [state]

            
            if state not in visited:
                visited.add(state)

                
                for action in self.problem.actions(state):
                    next_state = self.get_next_state(state, action)
                    if next_state not in visited:
                        queue.append((next_state, path + [state]))

        return None
    def depth_first_search(self):
        start = self.problem.get_initial_state()
        stack = [(start, [])]
        visited = set()
        visited.add(start)

        while len(stack)>0:
            nodoActual, path = stack.pop()
            if self.problem.goal_test(nodoActual):
                return path + [nodoActual]
            for action in self.problem.actions(nodoActual):  
                next_state = self.get_next_state(nodoActual, action)  
                if next_state not in visited:
                    stack.append((next_state, path + [nodoActual]))
                    visited.add(next_state)
        return None  
    
    def a_Star(self):
        start = self.problem.get_initial_state()
        goal = list(self.problem.goals)[0] 

        lista_abiertos = []  
        heapq.heappush(lista_abiertos, (0, start, [])) 

        g_costs = {start: 0} 
        visited = set()

        while lista_abiertos:
            _, nodo_actual, path = heapq.heappop(lista_abiertos)

            if self.problem.goal_test(nodo_actual):
                return path + [nodo_actual]

            if nodo_actual in visited:
                continue
            visited.add(nodo_actual)

            for action in self.problem.actions(nodo_actual):
                next_state = self.get_next_state(nodo_actual, action)
                new_cost = g_costs[nodo_actual] + self.problem.step_cost(nodo_actual, action, next_state)

                if next_state not in g_costs or new_cost < g_costs[next_state]:
                    g_costs[next_state] = new_cost
                    f_cost = new_cost + self.heuristica(next_state, goal)
                    heapq.heappush(lista_abiertos, (f_cost, next_state, path + [nodo_actual]))

        return None  


    def get_next_state(self, state, action):
        x, y = state
        if action == "UP":
            return (x - 1, y)
        elif action == "DOWN":
            return (x + 1, y)
        elif action == "LEFT":
            return (x, y - 1)
        elif action == "RIGHT":
            return (x, y + 1)
        return state  

def cargar_imagen(ruta):
    imagen = cv2.imread(ruta)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    return imagen

def es_color(pixel, referencia, tolerancia=30):
    return np.all(np.abs(pixel - referencia) <= tolerancia)

def imagen_a_matriz(imagen, tam_bloque=10):
    alto, ancho, _ = imagen.shape
    if alto != ancho:
        print("Error: La imagen no es cuadrada.")
        return None, None, None, None, None

    filas, cols = alto // tam_bloque, ancho // tam_bloque
    matriz = np.zeros((filas, cols), dtype=int)
    inicio = None
    metas = []

    COLOR_NEGRO = np.array([0, 0, 0])
    COLOR_BLANCO = np.array([255, 255, 255])
    COLOR_ROJO = np.array([255, 0, 0])
    COLOR_VERDE = np.array([0, 255, 0])

    for i in range(filas):
        for j in range(cols):
            bloque = imagen[i * tam_bloque:(i + 1) * tam_bloque, j * tam_bloque:(j + 1) * tam_bloque]
            color_promedio = np.mean(bloque, axis=(0, 1))

            if es_color(color_promedio, COLOR_BLANCO):
                matriz[i, j] = 0
            elif es_color(color_promedio, COLOR_NEGRO):
                matriz[i, j] = 1
            elif es_color(color_promedio, COLOR_ROJO):
                matriz[i, j] = 2
                inicio = (i, j)
            elif es_color(color_promedio, COLOR_VERDE):
                matriz[i, j] = 3
                metas.append((i, j))

    return matriz, inicio, metas, filas, cols

def dibujar_matriz(matriz, factor_escala=8):
    colores = {0: [255, 255, 255], 1: [0, 0, 0], 2: [255, 0, 0], 3: [0, 255, 0]}
    imagen = np.zeros((matriz.shape[0], matriz.shape[1], 3), dtype=np.uint8)

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            imagen[i, j] = colores[matriz[i, j]]

    plt.figure(figsize=(matriz.shape[1] // factor_escala, matriz.shape[0] // factor_escala))
    plt.imshow(imagen, interpolation="nearest")
    plt.axis("off")
    plt.show()

def procesar_imagen(ruta):
    imagen = cargar_imagen(ruta)
    matriz, inicio, metas, filas, cols = imagen_a_matriz(imagen, tam_bloque=10)
    if inicio is None or not metas:
        print("Error: La imagen no tiene un inicio o metas válidas.")
        return

    dibujar_matriz(matriz)

    maze = matriz
    start_pos = inicio
    goal_positions = metas

    problem = MazeProblem(maze, start_pos, goal_positions)
    
    search = GraphSearch(problem, heuristica_euclidiana)
    solution_path = search.a_Star()

    if solution_path:
        print("Camino para solucionar el laberinto:", solution_path)
        dibujar_camino(matriz=matriz, camino=solution_path)
    else:
        print("No se encontró solución.")

procesar_imagen("Test.bmp")