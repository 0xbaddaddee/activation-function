import ActivationFunctions

import numpy as np

if __name__ == '__main__':
    relu = ActivationFunctions.ReLU()
    input_tensor = np.random.randint(-10, 10, 16).reshape((4, 4))
    forward = relu.forward(input_tensor)
    error_tensor = relu.backward(forward)
