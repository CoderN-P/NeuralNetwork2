from typing import overload

from . import Layer
import numpy as np


class Activation(Layer):
    def __init__(self, activation, activation_der, name):
        self.input_data = None
        self.activation = activation # Activation function (e.g., sigmoid, relu)
        self.activation_der = activation_der # Derivative of the activation function
        self.name = name
 
    def forward(self, x):
        self.input_data = x
        return self.activation(x)

    def backward(self, grad, lr):
        return np.multiply(grad, self.activation_der(self.input_data))
    
    def __str__(self):
        return self.name