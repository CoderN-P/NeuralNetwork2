from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def compute(self, y_true, y_pred):
        """
        Compute the loss between true labels and predicted labels.
        
        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :return: Computed loss value.
        """
        pass

    @abstractmethod
    def gradient(self, y_true, y_pred):
        """
        Compute the gradient of the loss with respect to the predicted labels.
        
        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :return: Gradient of the loss with respect to the predicted labels.
        """
        pass