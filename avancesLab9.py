import random

class HMM:
    def __init__(self, states, observations, initial_prob, transition_prob, emission_prob):
        self.states = states
        self.observations = observations
        self.initial_prob = initial_prob
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob

    def generate_sequence(self, length):
        # Generar una secuencia de observaciones basada en el HMM
        sequence = []
        
        # Elegir un estado inicial basado en las probabilidades iniciales
        current_state = random.choices(self.states, weights=[self.initial_prob[s] for s in self.states])[0]
        
        for _ in range(length):
            # Generar una observación basada en el estado actual
            obs = random.choices(self.observations, weights=[self.emission_prob[current_state][o] for o in self.observations])[0]
            sequence.append(obs)

            # Elegir el siguiente estado basado en la probabilidad de transición
            current_state = random.choices(self.states, weights=[self.transition_prob[current_state][s] for s in self.states])[0]

        return sequence

    def forward(self, observations):
        # Implementar el algoritmo forward
        alpha = []

        # Paso de inicialización
        initial_alpha = {}
        for state in self.states:
            initial_alpha[state] = self.initial_prob[state] * self.emission_prob[state][observations[0]]
        alpha.append(initial_alpha)

        # Paso de inducción
        for t in range(1, len(observations)):
            current_alpha = {}
            for current_state in self.states:
                prob_sum = 0
                for previous_state in self.states:
                    prob_sum += alpha[t-1][previous_state] * self.transition_prob[previous_state][current_state]
                current_alpha[current_state] = prob_sum * self.emission_prob[current_state][observations[t]]
            alpha.append(current_alpha)

        return alpha

    def backward(self, observations):
        # Aquí ira la implementación del algoritmo backward
        pass

    def compute_state_probabilities(self, observations):
        # Aquí se combinarán forward y backward
        pass


# Parámetros del modelo
states = ['Sunny', 'Rainy']
observations = ['Sunny', 'Sunny', 'Rainy']

initial_prob = {'Sunny': 0.5, 'Rainy': 0.5}

transition_prob = {
    'Sunny': {'Sunny': 0.8, 'Rainy': 0.2},
    'Rainy': {'Sunny': 0.4, 'Rainy': 0.6}
}

emission_prob = {
    'Sunny': {'Sunny': 0.8, 'Rainy': 0.2},
    'Rainy': {'Sunny': 0.3, 'Rainy': 0.7}
}

# Instanciar el modelo
hmm = HMM(states, observations, initial_prob, transition_prob, emission_prob)
