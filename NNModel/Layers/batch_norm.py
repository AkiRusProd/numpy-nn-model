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

        self.input_shape = None
    

    def build(self, optimizer):
        self.optimizer = optimizer

        self.gamma = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)


        self.vg, self.mg         = np.zeros_like(self.gamma), np.zeros_like(self.gamma) # optimizers params
        self.vg_hat, self.mg_hat = np.zeros_like(self.gamma), np.zeros_like(self.gamma) # optimizers params

        self.vb, self.mb         = np.zeros_like(self.gamma), np.zeros_like(self.gamma) # optimizers params
        self.vb_hat, self.mb_hat = np.zeros_like(self.gamma), np.zeros_like(self.gamma) # optimizers params

        self.output_shape = self.input_shape


    def forward_prop(self, X, training):
        self.batch_size, self.input_size = X.shape

        if self.moving_mean is None: self.moving_mean = np.mean(X, axis = 0)
        if self.moving_var is None: self.moving_var = np.mean(X, axis = 0)
        

        if training == True:
            self.mean = np.mean(X, axis=0)
            self.var = np.var(X, axis=0)

            self.moving_mean = self.momentum * self.moving_mean + (1.0 - self.momentum) * self.mean
            self.moving_var = self.momentum * self.moving_var + (1.0 - self.momentum) * self.var
        else:
            self.mean = self.moving_mean
            self.var = self.moving_var


        self.X_centered = (X - self.mean)
        self.stddev_inv = 1 / np.sqrt(self.var + self.epsilon)

        X_hat = self.X_centered * self.stddev_inv

        Y = self.gamma * X_hat + self.beta
        
        return Y

        

    def backward_prop(self, error):
        
        output_error = (1 / self.batch_size) * self.gamma * self.stddev_inv * (
            self.batch_size * error
            - np.sum(error, axis = 0)
            - self.X_centered * np.power(self.stddev_inv, 2) * np.sum(error * self.X_centered, axis = 0)
            )

        X_hat = self.X_centered * self.stddev_inv
        self.grad_gamma = np.sum(error * X_hat, axis = 0)
        self.grad_beta = np.sum(error, axis = 0)

        # self.gamma -= grad_gamma
        # self.beta -= grad_beta

        return output_error

    def update_weights(self, layer_num):
        self.gamma, self.vg, self.mg, self.vg_hat, self.mg_hat  = self.optimizer.update(self.grad_gamma, self.gamma, self.vg, self.mg, self.vg_hat, self.mg_hat, layer_num)
        self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat = self.optimizer.update(self.grad_beta, self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat, layer_num)