import numpy as np
import json
from .models.Network import Network
from . import Activation, Dense
from .functions.mapping import loss_funcs, activation_funcs


class NeuralNetwork(Network):
    def __init__(self, layers, loss_function):
        """
        Initialize the neural network with layers and a loss function.
        
        :param layers: List of layers in the network.
        :param loss_function: Loss function to be used for training.
        """
        self.layers = layers
        self.loss_function = loss_function
        
    def predict(self, x):
        """
        Make predictions using the network.
        
        :param x: Input data.
        :return: Predicted output.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def train(self, x, y, epochs, lr):
        """
        Train the network on the provided data.
        
        :param x: Input data.
        :param y: Target labels.
        :param epochs: Number of training epochs.
        :param lr: Learning rate for training.
        """
        
        if len(x) != len(y):
            raise ValueError("Input and target data must have the same number of samples.")
        
        for epoch in range(epochs):
            loss = 0
          
            for x_train, y_train in zip(x, y):
                # Forward pass
                output = self.predict(x_train)
                
                
                expected = np.array([[int(i == y_train[0][0])] for i in range(0, 10)])

                
                # Compute loss
                loss += self.loss_function.compute(expected, output)

                # Cross Entropy + Softmax trick
                grad = self.loss_function.gradient(expected, output)

                
                if not (str(self.loss_function) == "CrossEntropy" and str(self.layers[-1]) == "Softmax"):
                    grad = self.layers[-1].backward(grad, lr)

                
                # Backpropagation
                for i in range(len(self.layers)-2, -1, -1):
                    layer = self.layers[i]
                    grad = layer.backward(grad, lr)
                    
            # Average loss for the epoch
            loss /= len(x)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")
            
           
    def save(self, path):
        parameters = {
            "loss": str(self.loss_function)
        }
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Activation):
                parameters[f"A{i}"] = str(layer)
            else:
                parameters[f"W{i}"] = layer.weights.tolist()
                parameters[f"B{i}"] = layer.biases.tolist()

        with open(path, "w") as json_file:
            json.dump(parameters, json_file, indent=4)
            
    @classmethod
    def load(cls, path):
        file = open(path, "r")
        parameters = json.load(file)
        
        layers = []
        for k, v in parameters.items():
            if k == "loss":
                continue
            if "W" in k: # Weights
                weights = np.array(v)
                shape = weights.shape
                l = Dense(shape[0], shape[1])
                l.weights = weights
                layers.append(l)
            elif "B" in k: # Biases
                biases = np.array(v)
                layers[-1].biases = biases
            else: # Activation func
                activation = activation_funcs.get(v, None)

                if activation:
                    layers.append(activation())
                    
        return cls(layers, loss_funcs[parameters["loss"]]())
                
                