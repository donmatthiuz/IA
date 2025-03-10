import gymnasium as gym

class FrozenLakeMDP:
    def __init__(self, nombre_entorno="FrozenLake-v1", es_resbaladizo=True, alpha=0.8, gamma=0.95, epsilon=0.2):
      
      
        self.entorno = gym.make(nombre_entorno, is_slippery=es_resbaladizo)

      
        self.n_estados = self.entorno.observation_space.n
        self.n_acciones = self.entorno.action_space.n

        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = epsilon

   