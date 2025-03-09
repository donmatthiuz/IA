import gymnasium as gym
import numpy as np

# Crear el entorno FrozenLake
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")


# Inicialización de la tabla Q
Q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.8   # Tasa de aprendizaje
gamma = 0.95  # Factor de descuento
epsilon = 0.1 # Probabilidad de exploración

# Entrenamiento del agente
for episode in range(1000):
    state = env.reset()[0]  # Obtener solo el primer valor de la tupla (el estado)
    done = False
    while not done:
        if np.random.rand() < epsilon:  # Exploración
            action = env.action_space.sample()
        else:  # Explotación
            action = np.argmax(Q[state])
        
        next_state, reward, done, truncated, info = env.step(action)

        
        # Actualización de la tabla Q
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        state = next_state

    if episode % 100 == 0:
        print(f"Episode {episode} complete")

# Evaluación del agente
state = env.reset()[0]  # Asegurarnos de que 'state' sea un entero
done = False
while not done:
    action = np.argmax(Q[state])
    next_state, reward, done, truncated, info = env.step(action)
    env.render()  # Mostrar el progreso del agente
