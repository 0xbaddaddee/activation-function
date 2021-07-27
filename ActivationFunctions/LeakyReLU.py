import numpy as np


class LeakyReLU:

    def __init__(self, alpha = 0.001):
        self.alpha = alpha
        self.input_tensor = None

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha):
        self._alpha = new_alpha

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.where(input_tensor > 0, input_tensor, self.alpha * input_tensor)

    def backward(self, error_tensor):
        return np.where(self.input_tensor > 0, error_tensor, self.alpha * error_tensor)
