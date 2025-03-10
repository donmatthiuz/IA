import gymnasium as gym
from Algoritmos import *
from FrozenLakeMDP import FrozenLakeMDP

env = FrozenLakeMDP()


qlearning_agent = Sarsa(env.entorno, alpha=env.alpha, gamma=env.gamma, 
                            epsilon=env.epsilon)


qlearning_agent.fit(1000)


success_rate = qlearning_agent.predict(100)

print(success_rate)
