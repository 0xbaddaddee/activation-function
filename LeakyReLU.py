import numpy as np


class LeakyReLU:

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.where(input_tensor > 0, input_tensor, self.alpha * input_tensor)

    def backward(self, error_tensor):
        return np.where(self.input_tensor > 0, error_tensor, self.alpha * error_tensor
