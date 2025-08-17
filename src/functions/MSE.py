from ..models import Loss

class MSE(Loss):
    def compute(self, y_true, y_pred):
        """
        Compute the Mean Squared Error (MSE) loss.
        
        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :return: Computed MSE loss value.
        """
        return ((y_true - y_pred) ** 2).mean()

    def gradient(self, y_true, y_pred):
        """
        Compute the gradient of the MSE loss with respect to the predicted labels.
        
        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :return: Gradient of the MSE loss with respect to the predicted labels.
        """
        
    
        return 2 * (y_true - y_pred) / len(y_true)  # Normalized by number of samples
    
    def __str__(self):
        return "MSE"