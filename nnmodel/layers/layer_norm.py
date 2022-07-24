import numpy as np
from nnmodel.exceptions.values_checker import ValuesChecker

class LayerNormalization():
    """
    Applies layer normalization to the input data
    ---------------------------------------------
        Args:
            `momentum` (float): the momentum parameter of the moving mean
            `epsilon` (float): the epsilon parameter of the algorithm
        Returns:
            output: the normalized input data with same shape
    """

    def __init__(self, epsilon = 0.001, input_shape = None):

        self.epsilon  = ValuesChecker.check_float_variable(epsilon, "epsilon")

        self.gamma = None
        self.beta = None

        self.mean = None
        self.var = None

        self.moving_mean = None #Not using
        self.moving_var = None #Not using

        self.optimizer = None

        self.axis = None
        self.input_shape = ValuesChecker.check_input_dim(input_shape, input_dim = None)
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    

    def build(self):
        self.gamma = np.ones((1))
        self.beta = np.zeros((1))


        self.vg, self.mg         = np.zeros_like(self.gamma), np.zeros_like(self.gamma)
        self.vg_hat, self.mg_hat = np.zeros_like(self.gamma), np.zeros_like(self.gamma)

        self.vb, self.mb         = np.zeros_like(self.gamma), np.zeros_like(self.gamma)
        self.vb_hat, self.mb_hat = np.zeros_like(self.gamma), np.zeros_like(self.gamma)

        

        self.output_shape = self.input_shape


    def forward_prop(self, X, training):
        self.input_data = X
        self.batch_size = X.shape[0]

        if self.axis is None: self.axis = tuple(np.arange(len(self.input_data.shape))[1:])
        
        self.mean = np.mean(self.input_data, axis = self.axis, keepdims = True)
        self.var = np.var(self.input_data, axis = self.axis, keepdims = True)
        
    
        self.X_centered = (self.input_data - self.mean)
        self.stddev_inv = 1 / np.sqrt(self.var + self.epsilon)

        X_hat = self.X_centered * self.stddev_inv

        self.output_data = self.gamma * X_hat + self.beta

        return self.output_data

        

    def backward_prop(self, error):
        
        output_error = (1 / self.batch_size) * self.gamma * self.stddev_inv * (
            self.batch_size * error
            - np.sum(error, axis = self.axis, keepdims = True)
            - self.X_centered * np.power(self.stddev_inv, 2) * np.sum(error * self.X_centered, axis = self.axis, keepdims = True)
            )

        X_hat = self.X_centered * self.stddev_inv
        self.grad_gamma = np.sum(error * X_hat)
        self.grad_beta = np.sum(error)

        
        return output_error

    def update_weights(self, layer_num):
        self.gamma, self.vg, self.mg, self.vg_hat, self.mg_hat  = self.optimizer.update(self.grad_gamma, self.gamma, self.vg, self.mg, self.vg_hat, self.mg_hat, layer_num)
        self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat = self.optimizer.update(self.grad_beta, self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat, layer_num)

    def get_grads(self):
        return self.grad_gamma, self.grad_beta

    def set_grads(self, grads):
        self.grad_gamma, self.grad_beta = grads