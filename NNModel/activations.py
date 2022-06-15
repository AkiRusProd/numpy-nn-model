import numpy as np




class Sigmoid():

    def function(self, x):

        return 1 / (1 + np.exp(-x))

    def derivative(self, x):

        return self.function(x) * (1.0 - self.function(x))


class Tanh():

    def function(self, x):

        return np.tanh(x)

    def derivative(self, x):

        return 1.0 - np.power(self.function(x), 2)


class ReLU():

    def function(self, x):

        return np.maximum(0, x)

    def derivative(self, x):

        return np.where(x <= 0, 0, 1)


class LeakyReLU():

    def __init__(self, alpha = 0.01):
        self.alpha = alpha

    def function(self, x):

        return np.where(x <= 0, self.alpha * x, x)

    def derivative(self, x):

        return np.where(x <= 0, self.alpha, 1)


class ELU():

    def __init__(self, alpha = 0.1):
        self.alpha = alpha 

    def function(self, x):

        return np.where(x <= 0, self.alpha * (np.exp(x) - 1), x)

    def derivative(self, x):

        return np.where(x <= 0, self.alpha + self.function(x), 1)


class GELU():

    def function(self, x):

        return (
                0.5
                * x
                * (
                    1
                    + np.tanh(
                        np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))
                    )
                )
            )

    def derivative(self, x):
        sech = lambda z: 2 / (np.exp(z) + np.exp(-z))

        return (
            0.5 * np.tanh(0.0356774 * np.power(x, 3) + 0.797885 * x)
            + (0.0535161 * np.power(x, 3) + 0.398942 * x)
            * np.power(sech(0.0356774 * np.power(x, 3) + 0.797885 * x), 2)
            + 0.5
        )

    
activations= {
    
    "sigmoid": Sigmoid(),
    "tanh": Tanh(),
    "relu": ReLU(),
    "leaky relu": LeakyReLU(),
    "elu": ELU(),
    "gelu": GELU(),
    
}
