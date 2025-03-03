import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FrozenLake:
    def __init__(self, filename, reward=None):
        if reward is None:
            reward = {0: 0, 1: 1.0, 2: -1.0}

        ecode = {'O': 0, 'M': 1, 'X': 2, 'I': 0}
        file = open(filename)
        self.map = np.array([list(s.strip().split(",")) for s in file.readlines()])
        
        zero_matrix = np.zeros((self.map.shape[0], self.map.shape[1]), dtype=int)

        self.posicion_inicial = (2, 0)

        for r in range(self.map.shape[0]):
            for c in range(self.map.shape[1]):
                valor = ecode[self.map[r, c]]
                if self.map[r, c] == 'I':
                    self.posicion_inicial = (r,c)
                zero_matrix[r, c] = valor

        self.map = []
        self.map = zero_matrix
        print(self.posicion_inicial)
        print(self.map)
       
        self.num_rows = self.map.shape[0]
        self.num_cols = self.map.shape[1]
        self.num_states = self.num_rows * self.num_cols
        self.num_actions = 4
        self.reward = reward
        self.reward_function = self.get_reward_function()
        self.transition_model = self.get_transition_model()

    def get_state_from_pos(self, pos):
        return pos[0] * self.num_cols + pos[1]

    def get_transition_model(self, random_rate=0.10):
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

    def get_reward_function(self):
        reward_table = np.zeros(self.num_states)
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                s = self.get_state_from_pos((r, c))
                reward_table[s] = self.reward[self.map[r, c]]
        return reward_table

    def generate_random_policy(self):
        return np.random.randint(self.num_actions, size=self.num_states)

    def execute_policy(self, policy, max_steps=20000):
        s = self.get_state_from_pos(self.posicion_inicial)
        total_reward = 0
        state_history = [s]
        steps = 0  

        while steps < max_steps:
            temp = self.transition_model[s, policy[s]]
            s_prime = np.random.choice(self.num_states, p=temp)
            state_history.append(s_prime)
            r = self.reward_function[s_prime]
            total_reward += r
            s = s_prime
            if r == 1 or r == -1:
                break
            steps += 1

        return total_reward, state_history  

    def get_pos_from_state(self, state):
        return state // self.num_cols, state % self.num_cols

    def plot_grid(self, state_history):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xticks(np.arange(self.num_cols + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.num_rows + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        for r in range(self.num_rows):
            for c in range(self.num_cols):
                cell_value = self.map[r, c]
                color = "white"
                if cell_value == 0:
                    color = "cyan"  
                elif cell_value == 2:
                    color = "black" 
                elif cell_value == 1:
                    color = "green"
                
                if (r,c) == self.posicion_inicial:
                    color = "yellow"
                rect = patches.Rectangle((c - 0.5, self.num_rows - r - 1.5), 1, 1, linewidth=1, edgecolor="black", facecolor=color)
                ax.add_patch(rect)

        path_x = []
        path_y = []
        for state in state_history:
            r, c = self.get_pos_from_state(state)
            path_x.append(c)
            path_y.append(self.num_rows - r - 1)  

        ax.plot(path_x, path_y, marker="o", color="blue", markersize=8, linestyle="-", linewidth=2, label="Recorrido")
        ax.legend()
        plt.show()




