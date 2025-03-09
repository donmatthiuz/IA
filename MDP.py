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





env = gym.make("FrozenLake-v1", is_slippery=True)

# Inicialización de la tabla Q
Q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.8   # Tasa de aprendizaje
gamma = 0.95  # Factor de descuento
epsilon = 0.1 # Probabilidad de exploración

# Entrenamiento del agente
for episode in range(1000):
    state = env.reset()  # Estado inicial
    done = False
    while not done:
        if np.random.rand() < epsilon:  # Exploración
            action = env.action_space.sample()
        else:  # Explotación
            action = np.argmax(Q[state])
        
        next_state, reward, done, info = env.step(action)
        
        # Actualización de la tabla Q
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        state = next_state

    if episode % 100 == 0:
        print(f"Episode {episode} complete")

# Evaluación del agente
state = env.reset()
done = False
while not done:
    action = np.argmax(Q[state])
    state, reward, done, info = env.step(action)
    env.render()  # Mostrar el progreso del agente
