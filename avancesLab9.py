class HMM:
    def __init__(self, states, observations, initial_prob, transition_prob, emission_prob):
        self.states = states
        self.observations = observations
        self.initial_prob = initial_prob
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob

    def generate_sequence(self, length):
        # Aquí irá la lógica para generar una secuencia
        pass

    def forward(self, observations):
        # Aquí irá la implementación del algoritmo forward
        pass

    def backward(self, observations):
        # Aquí irá la implementación del algoritmo backward
        pass

    def compute_state_probabilities(self, observations):
        # Aquí se combinarán forward y backward
        pass

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

hmm = HMM(states, observations, initial_prob, transition_prob, emission_prob)
