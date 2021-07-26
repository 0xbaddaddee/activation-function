import numpy as np


class ReLU:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.fmax(self.input_tensor, 0.0)

    def backward(self, error_tensor):
        return np.where(self.input_tensor <= 0, 0, error_tensor)
