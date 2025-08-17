import numpy as np
from .Layer import Layer


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((output_size, 1))  # Bias vector
        self.input_data = None  # To store input data for backpropagation
        

    def forward(self, x):
        self.input_data = x
        
        # Compute the output of the dense layer
        # This multiplies the input vector by the weights and adds the biases
        # In matrix multiplication, this basically returns a vector of the dot product of the input and weights for each output neuron plus the biases
        return np.dot(self.weights, x) + self.biases

    def backward(self, grad, lr):     
        grad_weights = np.dot(grad, self.input_data.T)
        grad_biases = np.sum(grad, axis=1, keepdims=True)
        
        # Update weights and biases
        self.weights -= lr * grad_weights
        self.biases -= lr * grad_biases
        
        # Return gradient for the previous layer
        return np.dot(self.weights.T, grad)
