import numpy as np
from ..models import Activation


# Singleton

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_der(x):
    sigma = sigmoid(x)
    return sigma*(1-sigma)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_der, name="Sigmoid")