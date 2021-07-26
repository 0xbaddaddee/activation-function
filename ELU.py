import numpy as np


class ELU:

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.where(input_tensor > 0, input_tensor, self.alpha * (np.exp(input_tensor) - 1))

    def backward(self, error_tensor):
        return np.where(self.input_tensor > 0, error_tensor, self.alpha * np.exp(self.input_tensor) * error_tensor)
