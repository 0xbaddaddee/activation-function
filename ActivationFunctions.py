import numpy as np


class ReLU:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.fmax(self.input_tensor, 0.0)

    def backward(self, error_tensor):
        return np.where(self.input_tensor <= 0, 0, error_tensor)


class LeakyReLU:

    def __init__(self, alpha=0.001):
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


class HyperbolicTan:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        a = np.exp(self.input_tensor)
        b = 1. / a
        tanh_x = (a - b) / (a + b)
        return tanh_x

    def backward(self, error_tensor):
        a = np.exp(self.input_tensor)
        b = 1. / a
        tanh_x = (a - b) / (a + b)
        return (1. - np.square(tanh_x)) * error_tensor


class ELU:

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.where(input_tensor > 0, input_tensor, self.alpha * (np.exp(input_tensor) - 1))

    def backward(self, error_tensor):
        return np.where(self.input_tensor > 0, error_tensor, self.alpha * np.exp(self.input_tensor) * error_tensor)


class Sigmoid:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return 1. / (1. + np.exp(-input_tensor))

    def backward(self, error_tensor):
        z = 1. / (1. + np.exp(-self.input_tensor))
        return z * (1. - z) * error_tensor


class SELU:

    def __init__(self, lambda_param=1.0507, alpha_param=1.67326):
        self.lambda_param = lambda_param
        self.alpha_param = alpha_param
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return self.lambda_param * np.where(input_tensor < 0,
                                            self.alpha_param * (np.exp(input_tensor) - 1), input_tensor)

    def backward(self, error_tensor):
        return self.lambda_param * error_tensor * np.where(self.input_tensor < 0,
                                                           self.alpha_param * np.exp(self.input_tensor), 1)


class Softplus:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.log(1. + np.exp(input_tensor))

    def backward(self, error_tensor):
        return error_tensor / (1. + np.exp(-self.input_tensor))
