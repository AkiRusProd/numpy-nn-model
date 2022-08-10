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

    def __init__(self, normalized_shape = None, epsilon = 0.001, input_shape = None):

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
        self.normalized_shape = ValuesChecker.check_input_dim(normalized_shape, input_dim = None)
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    

    def build(self):
        if self.normalized_shape is None:
            self.normalized_shape = self.input_shape

        self.gamma = np.ones(self.normalized_shape)
        self.beta = np.zeros(self.normalized_shape)


        self.vg, self.mg         = np.zeros_like(self.gamma), np.zeros_like(self.gamma)
        self.vg_hat, self.mg_hat = np.zeros_like(self.gamma), np.zeros_like(self.gamma)

        self.vb, self.mb         = np.zeros_like(self.gamma), np.zeros_like(self.gamma)
        self.vb_hat, self.mb_hat = np.zeros_like(self.gamma), np.zeros_like(self.gamma)

        

        self.output_shape = self.input_shape


    def forward_prop(self, X, training):
        self.input_data = X
        x_T = self.input_data.T
        # self.feature_size = np.prod(x_T.shape[:-1]) #x_T.shape[0]
        self.normalized_axis = tuple(np.arange(self.input_data.ndim - self.gamma.ndim)) 
        self.feature_size = self.gamma.size

        
        self.mean = np.mean(x_T, axis = 0)
        self.var = np.var(x_T,axis = 0)
        
    
        self.X_centered = (x_T - self.mean)
        self.stddev_inv = 1 / np.sqrt(self.var + self.epsilon)

        self.X_hat_T = self.X_centered * self.stddev_inv
        self.X_hat = self.X_hat_T.T
        
        self.output_data = self.gamma * self.X_hat + self.beta

        return self.output_data

        

    def backward_prop(self, error):
        error_T = error.T


        #first variant
        output_error = (1 / self.feature_size) * np.expand_dims(self.gamma, axis = self.normalized_axis).T * self.stddev_inv * ( #self.gamma[np.newaxis, :].T
            self.feature_size * error_T
            - np.sum(error_T, axis = 0)
            - self.X_centered * np.power(self.stddev_inv, 2) * np.sum(error_T * self.X_centered, axis = 0)
            )

        #second variant
        # dX_hat = error_T * self.gamma[np.newaxis, :].T
        # output_error = (1 / self.feature_size) * self.stddev_inv * (
        #     self.feature_size * dX_hat
        #     - np.sum(dX_hat, axis = 0)
        #     - self.X_hat_T * np.sum(dX_hat * self.X_hat_T, axis = 0)
        # )

        #third (naive slow) variant
        # dX_hat = error_T * self.gamma[np.newaxis, :].T
        # dvar = np.sum(dX_hat * self.X_centered, axis=0) * -.5 * self.stddev_inv**3
        # dmu = np.sum(dX_hat * -self.stddev_inv, axis=0) + dvar * np.mean(-2. * self.X_centered, axis=0)

        # output_error = (dX_hat * self.stddev_inv) + (dvar * 2 * self.X_centered / self.feature_size) + (dmu / self.feature_size)

        output_error = output_error.T

        
        self.grad_gamma = np.sum(error * self.X_hat, axis = self.normalized_axis) #axis = 0
        self.grad_beta = np.sum(error, axis = self.normalized_axis) #axis = 0

        
        return output_error

    def update_weights(self, layer_num):
        self.gamma, self.vg, self.mg, self.vg_hat, self.mg_hat  = self.optimizer.update(self.grad_gamma, self.gamma, self.vg, self.mg, self.vg_hat, self.mg_hat, layer_num)
        self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat = self.optimizer.update(self.grad_beta, self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat, layer_num)

    def get_grads(self):
        return self.grad_gamma, self.grad_beta

    def set_grads(self, grads):
        self.grad_gamma, self.grad_beta = grads