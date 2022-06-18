import numpy as np


#References: https://mlfromscratch.com/activation-functions-explained/
class ActivationFunction():

    def __init__(self):
        pass

    def parent_name(self): 
        
        return "ActivationFunction"



class Sigmoid(ActivationFunction):

    def function(self, x):

        return 1 / (1 + np.exp(-x))

    def derivative(self, x):

        return self.function(x) * (1.0 - self.function(x))


class Tanh(ActivationFunction):

    def function(self, x):

        return np.tanh(x)

    def derivative(self, x):

        return 1.0 - np.power(self.function(x), 2)


class Softmax(ActivationFunction):

    def function(self, x):
        e_x = np.exp(x - np.max(x, axis = -1, keepdims=True))

        return e_x / np.sum(e_x, axis = -1, keepdims=True)

    def derivative(self, x):
        
        return self.function(x) * (1.0 - self.function(x))


class Softplus(ActivationFunction):

    def function(self, x):
        
        return np.log(1 + np.exp(x))

    def derivative(self, x):
        
        return 1 / (1 + np.exp(-x))


class ReLU(ActivationFunction):

    def function(self, x):

        return np.maximum(0, x)

    def derivative(self, x):

        return np.where(x <= 0, 0, 1)


class LeakyReLU(ActivationFunction):

    def __init__(self, alpha = 0.01):
        self.alpha = alpha

    def function(self, x):

        return np.where(x <= 0, self.alpha * x, x)

    def derivative(self, x):

        return np.where(x <= 0, self.alpha, 1)


class ELU(ActivationFunction):

    def __init__(self, alpha = 0.1):
        self.alpha = alpha 

    def function(self, x):

        return np.where(x <= 0, self.alpha * (np.exp(x) - 1), x)

    def derivative(self, x):

        return np.where(x <= 0, self.alpha + self.function(x), 1)


class SELU(ActivationFunction):

    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.lmbda = 1.0507009873554804934193349852946 

    def function(self, x):
        return self.lmbda * np.where(x > 0, x, self.alpha*(np.exp(x)-1))

    def derivative(self, x):
        return self.lmbda * np.where(x > 0, 1, self.alpha * np.exp(x))


class GELU(ActivationFunction):

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

class Identity(ActivationFunction):

    def function(self, x):

            return x

    def derivative(self, x):

            return np.ones(x.shape)

    
activations= {
    
    "sigmoid": Sigmoid(),
    "tanh": Tanh(),
    "softmax": Softmax(),
    "softplus": Softplus(),
    "relu": ReLU(),
    "leaky relu": LeakyReLU(),
    "elu": ELU(),
    "selu": SELU(),
    "gelu": GELU(),
    None: Identity()
    
}
