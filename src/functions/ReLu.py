import numpy as np
from ..models import Activation


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x <= 0, 0, 1)


class ReLu(Activation):
    def __init__(self):
        super().__init__(relu, relu_derivative, name="ReLu")