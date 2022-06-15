import numpy as np


class MSE():

    def loss(y, t):

        return np.power(t - y, 2)

    def derivative(y, t):

        return -(t - y)


class BinaryCrossEntropy():

    def loss(y, t):

        return -t * np.log(y) + (1 - t) * np.log(1 - y),

    def derivative(y, t):

        return -t / y + (1 - t) / (1 - y)


class CategoricalCrossEntropy():

    def loss(y, t):

        return t * np.log(y)

    def derivative(y, t):

        return t / y


loss_functions = {
    
    "mse": MSE,
    "binary crossentropy": BinaryCrossEntropy,
    "categorical crossentropy": CategoricalCrossEntropy,

}
