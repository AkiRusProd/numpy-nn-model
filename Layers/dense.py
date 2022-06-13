import numpy as np


class Dense():
    #TODO
    #add accumulate gradients for batch backprop
    #maybe remove batch_size; only propagate one sample, not batch
    #add bias

    def __init__(self, units_num, activation = None, input_shape = None):
        self.units_num = units_num
        self.activation = NotImplemented
        self.activation_der = NotImplemented
        self.input_shape = input_shape

        self.w  = None

    def init_weights(self):
        batch_size, input_size = self.input_shape
        
        self.w = np.random.normal(0, 1, (input_size, self.units_num))

    def forward_prop(self, X):
        self.input_data = X
        self.output_data = self.activation(np.dot(self.input_data, self.w))
        
        return self.output_data

    def backward_prop(self, error):
        error *= self.activation_der(self.output_data)
        
        self.grad_w = np.dot(self.input_data.T, error)

        output_error = np.dot(error, self.w.T) #* self.activation_der(self.input_data) #TODO FIX: ACT FUNC OF PREV LAYER| unverified solution

        return output_error

