from abc import ABC, abstractmethod


class Network(ABC):
    @abstractmethod
    def predict(self, x):
        """
        Make predictions using the network.
        
        :param x: Input data.
        :return: Predicted output.
        """
        
        pass

    @abstractmethod
    def train(self, x, y, epochs, lr) -> None:
        """
        Train the network on the provided data.
        :param x: Input data.
        :param y: Target labels.
        :param epochs: Number of training epochs.
        :param lr: Learning rate for training.
        """
        
        pass
    
    @abstractmethod
    def save(self, path) -> None:
        """
        Save the network to a file.
        :param path: File path where the network should be saved.
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """
        Load the network from a file.
        :param path: 
        :return: Network with parameters loaded from the file.
        """
        pass