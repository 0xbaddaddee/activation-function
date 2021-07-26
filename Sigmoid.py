import numpy as np


# apply sigmoid function on a numpy array
def sigmoid(z):
    return 1. / (1. + np.exp(-z))


class Sigmoid:

    def __init__(self):
        self.input_tensor = None  # keep track of the current input tensor to use in the backward pass

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return sigmoid(input_tensor)

    def backward(self, error_tensor):
        return sigmoid(self.input_tensor) * (1. - sigmoid(self.input_tensor)) * error_tensor
