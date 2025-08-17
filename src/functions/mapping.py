from .MSE import MSE
from .ReLu import ReLu
from .Sigmoid import Sigmoid
from .Softmax import Softmax
from .CrossEntropy import CrossEntropy



loss_funcs = {
    "MSE": MSE,
    "CrossEntropy": CrossEntropy,
}

activation_funcs = {
    "ReLu": ReLu,
    "Sigmoid": Sigmoid,
    "Softmax": Softmax,
}