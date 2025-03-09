import gymnasium as gym
import numpy as np
class FrozenLakeMDP:
    def __init__(self, nombre_entorno="FrozenLake-v1", es_resbaladizo=True):
        self.entorno = gym.make(nombre_entorno, is_slippery=es_resbaladizo)

        self.n_estados = self.entorno.observation_space.n  # 16 estados
        self.n_acciones = self.entorno.action_space.n  # 4 acciones (izq, abajo, der, arriba)

        self.descripcion = """
        El problema del Lago Congelado es un MDP donde un agente debe llegar a la meta sin caer en agujeros.
        - Estados: 16 (correspondientes a un tablero 4x4).
        - Acciones: 4 (izquierda, abajo, derecha, arriba).
        - Transiciones: Probabilísticas si el hielo es resbaladizo.
        - Recompensas: 1 por llegar a la meta, 0 en cualquier otro caso.
        """

        self.gamma = 0.99

    def obtener_definicion_problema(self):
        return self.descripcion