import numpy as np


class LTSM_Unit:

    # Initalisation
    def __init__(self, n_inputs):
        self._Forget_Gate = Sigmoid_Neuron(n_inputs)
        self._Input_Gate = Sigmoid_Neuron(n_inputs)
        self._State_Candidate_Gate = Tanh_Neuron(n_inputs)
        self._Output_Gate = Sigmoid_Neuron(n_inputs)

    # Forward pass
    def forward(self):
        pass

    def backward(self):
        pass

class Sigmoid_Neuron:

    def __init__(self, n_inputs):
        self._weights = 0.01 * np.random.randn(n_inputs) 
        self._biases = np.zeros(1)

    def forward(self, inputs):
        # Inputs is a 1xn array of n data points
        unactivated_output = np.dot(inputs, self._weights) + self._biases
        self._output = 1/(1 + np.exp(-unactivated_output))


    def backward(self):
        pass

class Tanh_Neuron:

    def __init__(self, n_inputs):
        self._weights = 0.01 * np.random.randn(n_inputs) 
        self._biases = np.zeros(1)

    def forward(self, inputs):
        # Inputs is a 1xn array of n data points
        unactivated_output = np.dot(inputs, self._weights) + self._biases
        self._output = np.tanh(unactivated_output)

    def backward(self):
        pass