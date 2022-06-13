import numpy as np


class BatchNormalization():

    def __init__(self, momentum = 0.99, epsilon = 0.001):

        self.momentum = momentum
        self.epsilon = epsilon

        self.gamma = None
        self.beta = None

        self.mean = None
        self.var = None

        self.moving_mean = None
        self.moving_var = None
        

    def forward_prop(self, X):
        self.batch_size, self.input_size = X.shape

        if self.gamma == None: self.gamma = np.ones(self.input_size)
        if self.beta == None: self.beta = np.zeros(self.input_size)

        if self.moving_mean is None: self.moving_mean = np.mean(X, axis = 0)
        if self.moving_var is None: self.moving_var = np.mean(X, axis = 0)

        self.mean = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)

        moving_mean = self.momentum * moving_mean + (1.0 - self.momentum) * self.mean
        moving_var = self.momentum * moving_var + (1.0 - self.momentum) * self.var


        self.X_centered = (X - self.mean)
        self.stddev_inv = 1 / np.sqrt(self.var + self.eps)

        X_hat = self.X_centered * self.stddev_inv

        Y = self.gamma * X_hat + self.beta

        return Y

        

    def backward_prop(self, error):
        
        output_error = (1 / self.batch_size) * self.gamma * self.stddev_inv * (
            self.batch_size * error
            - np.sum(error, axis = 0)
            - self.X_centered * np.pow(self.stddev_inv, 2) * np.sum(error * self.X_centered, axis = 0)
            )

        X_hat = self.X_centered * self.stddev_inv
        grad_gamma = np.sum(error * X_hat, axi = 0)
        grad_beta = np.sum(error, axis = 0)

        self.gamma -= grad_gamma
        self.beta -= grad_beta

        return output_error