import numpy as np
from ..models import Activation


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def softmax_derivative(z):
    # Softmax + cross entropy loss trick (No need to compute derivative explicitly)
    pass


class Softmax(Activation):
    def __init__(self):
        super().__init__(softmax, softmax_derivative, name="Softmax")
