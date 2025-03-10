import gymnasium as gym
import matplotlib.pyplot as plt
from Algoritmos import *
from Algoritmos2 import *
from FrozenLakeMDP import FrozenLakeMDP

# Define different episode counts to test
epochs_list = [500, 1000, 2000, 5000]

# Store results
sarsa_results = []
qlearning_results = []
montecarlo_results = []

env = FrozenLakeMDP()

# Q-Learning
qlearning = Qlearning(env.entorno, alpha=env.alpha, gamma=env.gamma, epsilon=env.epsilon)
qlearning.fit(1000)
q_success_rate = qlearning.predict(100)
qlearning_results.append(q_success_rate)


#----------------------------------------------------------------
# SI SE DESEA GRAFICAR Y COMPARAR RENDIMIENTO ENTRE ALGORITMOS
#----------------------------------------------------------------

# for epochs in epochs_list:
#     # Sarsa
#     sarsa = Sarsa(env.entorno, alpha=env.alpha, gamma=env.gamma, epsilon=env.epsilon)
#     sarsa.fit(epochs)
#     sarsa_success_rate = sarsa.predict(100)
#     sarsa_results.append(sarsa_success_rate)
    
#     # Q-Learning
#     qlearning = Qlearning(env.entorno, alpha=env.alpha, gamma=env.gamma, epsilon=env.epsilon)
#     qlearning.fit(epochs)
#     q_success_rate = qlearning.predict(100)
#     qlearning_results.append(q_success_rate)
    
#     # Monte Carlo
#     agent = MonteCarlo_Libre_Modelos(env.entorno, alpha=0.8, epsilon=0.5)
#     agent.monte_carlo_e_soft(episodes=epochs)
#     montecarlo_success_rate = agent.test_policy(episodes=100)
#     montecarlo_results.append(montecarlo_success_rate)

# # Plot results
# plt.figure(figsize=(10, 6))
# plt.plot(epochs_list, sarsa_results, marker='o', linestyle='-', label='Sarsa')
# plt.plot(epochs_list, qlearning_results, marker='s', linestyle='-', label='Q-Learning')
# plt.plot(epochs_list, montecarlo_results, marker='^', linestyle='-', label='Monte Carlo')

# plt.xlabel("Number of Episodes")
# plt.ylabel("Success Rate (%)")
# plt.title("Performance Comparison of RL Algorithms on Frozen Lake")
# plt.legend()
# plt.grid()
# plt.show()
