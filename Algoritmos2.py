import random
from time import sleep
from IPython.display import clear_output
import numpy as np
from collections import defaultdict

class MonteCarlo_Libre_Modelos:
    def __init__(self, env, alpha=0.8, gamma=0.95, epsilon=0.5):
        self.env = env
        self.alpha = alpha  
        self.gamma = gamma  
        self.epsilon = epsilon 
        self.policy = self.create_random_policy() 
        self.Q = self.create_state_action_dictionary() 

    def create_random_policy(self):
        """Genera una política aleatoria para cada estado."""
        policy = {}
        for key in range(0, self.env.observation_space.n):
            p = {}
            for action in range(0, self.env.action_space.n):
                p[action] = 1 / self.env.action_space.n
            policy[key] = p
        return policy

    def create_state_action_dictionary(self):
        """Crea un diccionario Q para almacenar los valores de las acciones."""
        Q = {}
        for state in range(self.env.observation_space.n):
            Q[state] = {action: 0.0 for action in range(self.env.action_space.n)}
        return Q

    def run_game(self, display=False):
      """Ejecuta un episodio en el entorno siguiendo la política."""
      state = self.env.reset()  # Inicializa el estado usando reset
      episode = []
      done = False

      while not done:

          if isinstance(state, tuple):
              state = state[0]


          n = random.uniform(0, sum(self.policy[state].values()))
          cumulative_prob = 0
          for action, prob in self.policy[state].items():
              cumulative_prob += prob
              if n < cumulative_prob:
                  action_taken = action
                  break

          next_state, reward, done, truncated, info = self.env.step(action_taken)
          episode.append((state, action_taken, reward))
          state = next_state

      return episode


    def test_policy(self, episodes=100):
        """Testea la política con varios episodios."""
        wins = 0
        for _ in range(episodes):
            episode = self.run_game(display=False)
            if episode[-1][-1] == 1:  # Si la última recompensa es 1, el agente ganó
                wins += 1
        return wins / episodes

    def monte_carlo_e_soft(self, episodes=100):
        """Entrena al agente usando el algoritmo Monte Carlo Epsilon-Soft."""
        returns = {}  

        for _ in range(episodes):
            G = 0
            episode = self.run_game(display=False)


            for state, action, reward in reversed(episode):
                G = self.gamma * G + reward  # Cálculo de la recompensa acumulada

                state_action = (state, action)
                if state_action not in [(s, a) for s, a, _ in episode[:-1]]:  # Si es la primera vez que vemos esta (estado, acción)
                    if state_action not in returns:
                        returns[state_action] = []
                    returns[state_action].append(G)


                    self.Q[state][action] = sum(returns[state_action]) / len(returns[state_action])


                    max_action = max(self.Q[state], key=self.Q[state].get)
                    for a in self.policy[state]:
                        if a == max_action:
                            self.policy[state][a] = 1 - self.epsilon + self.epsilon / self.env.action_space.n
                        else:
                            self.policy[state][a] = self.epsilon / self.env.action_space.n

        return self.policy



















