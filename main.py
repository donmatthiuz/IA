import gymnasium as gym
from Algoritmos import *
from Algoritmos2 import *
from FrozenLakeMDP import FrozenLakeMDP

env = FrozenLakeMDP()


qlearning_agent = Sarsa(env.entorno, alpha=env.alpha, gamma=env.gamma, 
                            epsilon=env.epsilon)


qlearning_agent.fit(1000)


success_rate = qlearning_agent.predict(100)

print(success_rate)


# agent = MonteCarlo_Modelo(env.entorno, alpha=0.8,  epsilon=0.5)
# agent.monte_carlo_e_soft(episodes=1000)
# print("Tasa de éxito:", agent.test_policy(episodes=100))




