import numpy as np


class MSE():

    def loss(self, y, t):

        return np.power(t - y, 2)

    def derivative(self, y, t):

        return -(t - y)


class BinaryCrossEntropy():

    def loss(self, y, t):

        return -(t * np.log(y + 1e-8) + (1 - t) * np.log(1 - y + 1e-8))

    def derivative(self, y, t):

        return -t / (y + 1e-8) + (1 - t) / (1 - (y + 1e-8))


class CategoricalCrossEntropy():

    def loss(self, y, t):

        return - t * np.log(y)

    def derivative(self, y, t):

        return -t / y


class MiniMaxCrossEntropy():

    def generator_loss(self, y_real, t = None): # not used, implemented in gan.py
        return -np.log(y_real)

    def discriminator_real_loss(self, y_real, t = None):# not used, implemented in gan.py
        return -np.log(y_real)

    def discriminator_fake_loss(self, y_fake, t = None):# not used, implemented in gan.py
        return -np.log(1 - y_fake)

    def generator_derivative(self, y_fake, t = None): # -log(D(G(z)))
        return -1 / (y_fake + 1e-8)

    def discriminator_real_derivative(self, y_real, t = None): # min -log(D(x))
        return -1 / (y_real + 1e-8)

    def discriminator_fake_derivative(self, y_fake, t = None): # max log(1 - D(G(z)))
        return  1 / (1 - y_fake + 1e-8)

loss_functions = {
    
    "mse": MSE(),
    "binary_crossentropy": BinaryCrossEntropy(),
    "categorical_crossentropy": CategoricalCrossEntropy(),
    "minimax_crossentropy": MiniMaxCrossEntropy(),

}
