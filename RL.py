import abc
import gymnasium as gym
import numpy as np

class RL(abc.ABC):
    def __init__(self, env, alpha=0.8, gamma=0.95, epsilon=0.5):
        
        self.env = env
        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])  # Tabla Q inicializada a ceros

    @abc.abstractmethod
    def fit(self, episodes):
        
        pass

    @abc.abstractmethod
    def predict(self, num_evaluations=100):
        
        pass

    def select_action(self, state):
        
        if np.random.rand() < self.epsilon:  # Exploración
            return self.env.action_space.sample()
        else:  # Explotación
            return np.argmax(self.Q[state])

