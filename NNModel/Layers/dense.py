import numpy as np
from NNModel.activations import activations


class Dense():
    #TODO
    #add bias

    def __init__(self, units_num, activation = None, input_shape = None):
        self.units_num = units_num
        self.input_shape = input_shape

        if type(activation) is str:
            self.activation = activations[activation]
        else:
            self.activation = activation

        
        self.w  = None

    def build(self, optimizer):
        self.optimizer = optimizer
        self.input_size = self.input_shape[-1]
        
        self.w = np.random.normal(0, pow(self.input_size, -0.5), (self.input_size, self.units_num))

        self.v, self.m         = np.zeros_like(self.w), np.zeros_like(self.w) # optimizers params
        self.v_hat, self.m_hat = np.zeros_like(self.w), np.zeros_like(self.w) # optimizers params

        self.output_shape = (1, self.units_num)

    def forward_prop(self, X, training):
        self.input_data = X
       
        self.batch_size = len(self.input_data)

        self.output_data = np.dot(self.input_data, self.w)
        
        return self.activation.function(self.output_data)

    def backward_prop(self, error):
        error *= self.activation.derivative(self.output_data)
        
        self.grad_w = np.dot(self.input_data.T, error)

        output_error = np.dot(error, self.w.T) #* self.activation_der(self.input_data) #TODO FIX: ACT FUNC OF PREV LAYER| unverified solution

        return output_error

    def update_weights(self, layer_num):
  
        self.w, self.v, self.m, self.v_hat, self.m_hat = self.optimizer.update(self.grad_w, self.w, self.v, self.m, self.v_hat, self.m_hat, layer_num)
        

