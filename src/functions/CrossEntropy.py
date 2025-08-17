from ..models import Loss
import numpy as np

class CrossEntropy(Loss):
    def compute(self, y_true, y_pred):
        """
        Compute the Cross-Entropy loss.
        
        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :return: Computed MSE loss value.
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true * np.log(y_pred)).sum()

    def gradient(self, y_true, y_pred):
        return y_pred - y_true # Gradient of cross-entropy with softmax is y_pred - y_true
    
    def __str__(self):
        return "CrossEntropy"