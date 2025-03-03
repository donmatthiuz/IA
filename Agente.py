import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FrozenLake:
    def __init__(self, filename, reward=None):
        if reward is None:
            reward = {'O': 0.0, 'M': 1.0, 'X': -1.0, 'I': 0.0}  # O no hay costo, I es el inicio, 

        
        with open(filename, 'r') as file:
            self.map = np.array([list(s.strip().split(",")) for s in file.readlines()])

        self.num_rows = self.map.shape[0]
        self.num_cols = self.map.shape[1]
        self.num_states = self.num_rows * self.num_cols
        self.num_actions = 4
        self.reward = reward
        self.reward_function = self.get_reward_table()
        self.transition_model = self.get_transition_model()

    def get_reward_table(self):
       
        reward_table = np.zeros((self.num_rows, self.num_cols))
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                reward_table[i][j] = self.reward.get(self.map[i][j], 0.0)  # Default 0 si no está en reward
        return reward_table

    def get_state_from_pos(self, pos):
        return pos[0] * self.num_cols + pos[1]

    def get_pos_from_state(self, state):
        return state // self.num_cols, state % self.num_cols
    
    def generate_random_policy(self):
      return np.random.randint(self.num_actions, size=self.num_states)

    def execute_policy(self, policy, start_pos=(2, 0)):
      s = self.get_state_from_pos(start_pos)
      total_reward = 0
      state_history = [s]
      while True:
        temp = self.transition_model[s, policy[s]]
        s_prime = np.random.choice(self.num_states, p=temp)
        state_history.append(s_prime)
        r = float(self.reward_function.flatten()[s_prime])  # ✅ Garantiza un solo valor


        total_reward += float(r)
        s = s_prime
        if np.any(r == 1) or np.any(r == -1):
          break
      return total_reward

    def get_reward_function(self):
        reward_table = np.zeros(self.num_states)
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                s = self.get_state_from_pos((r, c))
                reward_table[s] = self.reward[self.map[r, c]]
        return reward_table

    def get_transition_model(self, random_rate=0.33): #El frozen lake si tiene una probabilidad de 33% de ir en la direccion deseada
        transition_model = np.zeros((self.num_states, self.num_actions, self.num_states))
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                s = self.get_state_from_pos((r, c))
                neighbor_s = np.zeros(self.num_actions)
                if self.map[r, c] == 0:
                      for a in range(self.num_actions):
                          new_r, new_c = r, c
                          if a == 0:
                              new_r = max(r - 1, 0)
                          elif a == 1:
                              new_c = min(c + 1, self.num_cols - 1)
                          elif a == 2:
                              new_r = min(r + 1, self.num_rows - 1)
                          elif a == 3:
                              new_c = max(c - 1, 0)
                          if self.map[new_r, new_c] == 3:
                              new_r, new_c = r, c
                          s_prime = self.get_state_from_pos((new_r, new_c))
                          neighbor_s[a] = s_prime
                else:
                    neighbor_s = np.ones(self.num_actions) * s
                for a in range(self.num_actions):
                    transition_model[s, a, int(neighbor_s[a])] += 1 - random_rate
                    transition_model[s, a, int(neighbor_s[(a + 1) % self.num_actions])] += random_rate / 2.0
                    transition_model[s, a, int(neighbor_s[(a - 1) % self.num_actions])] += random_rate / 2.0
        return transition_model


# Cargar el entorno
grid = FrozenLake(filename='./frozen_lake.csv')

# Generar una política aleatoria
policy = grid.generate_random_policy()
print("Política aleatoria generada:", policy)

# Ejecutar la política desde la posición de inicio
reward = grid.execute_policy(policy, start_pos=(0, 0))
print("Recompensa obtenida:", reward)
