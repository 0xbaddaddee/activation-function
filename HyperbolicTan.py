import numpy as np


# hyperbolic tangent on numpy array input
def tanh(x):
    a = np.exp(x)
    b = 1. / np.exp(x)
    return (a - b) / (a + b)


class HyperbolicTan:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return tanh(input_tensor)

    def backward(self, error_tensor):
        return (1. - tanh(self.input_tensor)) * error_tensor
