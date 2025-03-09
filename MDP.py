import gymnasium as gym
import numpy as np

class MDPFramework:
    def __init__(self, nombre_entorno="FrozenLake-v1", es_resbaladizo=True, gamma=0.99):
        self.entorno = gym.make(nombre_entorno, is_slippery=es_resbaladizo)

        self.n_estados = self.entorno.observation_space.n
        self.n_acciones = self.entorno.action_space.n
        self.gamma = gamma

        self.transiciones = self._extraer_transiciones()
        self.recompensas = self._extraer_recompensas()

    def _extraer_transiciones(self):
        transiciones = {}
        entorno_sin_envuelta = self.entorno.unwrapped

        for estado in range(self.n_estados):
            transiciones[estado] = {}
            for accion in range(self.n_acciones):
                transiciones[estado][accion] = entorno_sin_envuelta.P[estado][accion]

        return transiciones

    def _extraer_recompensas(self):
        recompensas = np.zeros((self.n_estados, self.n_acciones))
        entorno_sin_envuelta = self.entorno.unwrapped

        for estado in range(self.n_estados):
            for accion in range(self.n_acciones):
                for prob, estado_siguiente, recompensa, terminado in entorno_sin_envuelta.P[estado][accion]:
                    recompensas[estado][accion] += prob * recompensa

        return recompensas

    def obtener_probabilidades_transicion(self, estado, accion):
        return self.transiciones[estado][accion]

    def obtener_recompensa(self, estado, accion):
        return self.recompensas[estado][accion]

    def obtener_acciones_posibles(self, estado):
        return list(range(self.n_acciones))

    def reiniciar(self):
        return self.entorno.reset()[0]

    def paso(self, accion):
        return self.entorno.step(accion)

    def cerrar(self):
        self.entorno.close()