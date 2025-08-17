from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        """
        Forward pass through the layer.
        
        :param x: Input data.
        :return: Output data after applying the layer's transformation.
        """
        pass

    @abstractmethod
    def backward(self, grad, lr):
        """
        Backward pass through the layer to compute gradients.
        
        :param grad: Gradient of the loss with respect to the output of this layer.
        :param lr: Learning rate for updating parameters.
        
        :return: Gradient of the loss with respect to the input of this layer.
        """
        pass
        