from RL import RL
import numpy as np

class Qlearning(RL):
    def __init__(self, env, alpha=0.8, gamma=0.95, epsilon=0.5):
        super().__init__(env, alpha, gamma, epsilon)
    
    def fit(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()[0]
            done = False
            
            while not done:

                action = self.select_action(state)
                

                next_state, reward, done, truncated, info = self.env.step(action)
                

                best_next_action = np.argmax(self.Q[next_state])  
                self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, best_next_action] - self.Q[state, action])
                
               
                state = next_state

            if episode % 100 == 0:
                print(f"Q-Learning - Episode {episode} complete")

    def predict(self, num_evaluations=100):
        successful_episodes = 0
        for eval_episode in range(num_evaluations):
            state = self.env.reset()[0]
            done = False
            steps = 0

            while not done:
                action = np.argmax(self.Q[state]) 
                next_state, reward, done, truncated, info = self.env.step(action)
                steps += 1

                if reward == 1: 
                    successful_episodes += 1
                    
                    break

                state = next_state

        success_rate = successful_episodes / num_evaluations * 100
        
        return success_rate


class Sarsa(RL):
    def __init__(self, env, alpha=0.8, gamma=0.95, epsilon=0.5):
        super().__init__(env, alpha, gamma, epsilon)
    
    def fit(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()[0]

            action = self.select_action(state)
            done = False
            
            while not done:
                next_state, reward, done, truncated, info = self.env.step(action)
                

                next_action = self.select_action(next_state)
                

                self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])
                

                state = next_state
                action = next_action

            if episode % 100 == 0:
                print(f"Sarsa - Episode {episode} complete")

    def predict(self, num_evaluations=100):
        successful_episodes = 0
        for eval_episode in range(num_evaluations):
            state = self.env.reset()[0]
            done = False
            steps = 0

            while not done:
                action = np.argmax(self.Q[state])  
                next_state, reward, done, truncated, info = self.env.step(action)
                steps += 1

                if reward == 1: 
                    successful_episodes += 1
                    break

                state = next_state

        success_rate = successful_episodes / num_evaluations * 100
        return success_rate