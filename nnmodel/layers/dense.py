import numpy as np
from nnmodel.activations import activations
from nnmodel.exceptions.values_checker import ValuesChecker

class Dense():
    #TODO
    #add bias

    def __init__(self, units_num, activation = None, input_shape = None, use_bias = True):
        self.units_num   = ValuesChecker.check_integer_variable(units_num, "units_num")
        self.input_shape = ValuesChecker.check_input_dim(input_shape, input_dim = 1)
        self.activation  = ValuesChecker.check_activation(activation, activations)
        self.use_bias = use_bias

        
        self.w = None
        self.b = None

    def build(self, optimizer):
        self.optimizer = optimizer
        self.input_size = self.input_shape[-1]
        
        self.w = np.random.normal(0, pow(self.input_size, -0.5), (self.input_size, self.units_num))
        self.b = np.zeros(self.units_num)

        self.v, self.m         = np.zeros_like(self.w), np.zeros_like(self.w) # optimizers params
        self.v_hat, self.m_hat = np.zeros_like(self.w), np.zeros_like(self.w) # optimizers params

        self.vb, self.mb         = np.zeros_like(self.b), np.zeros_like(self.b) # optimizers params
        self.vb_hat, self.mb_hat = np.zeros_like(self.b), np.zeros_like(self.b) # optimizers params

        self.output_shape = (1, self.units_num)

    def forward_prop(self, X, training):
        self.input_data = X
       
        self.batch_size = len(self.input_data)

        self.output_data = np.dot(self.input_data, self.w)# + self.b
        
        return self.activation.function(self.output_data)

    def backward_prop(self, error):
        error *= self.activation.derivative(self.output_data)
        
        self.grad_w = np.dot(self.input_data.T, error)
        self.grad_b = np.sum(error, axis = 0)

        output_error = np.dot(error, self.w.T) #* self.activation_der(self.input_data) #TODO FIX: ACT FUNC OF PREV LAYER| unverified solution

        return output_error

    def update_weights(self, layer_num):
        self.w, self.v, self.m, self.v_hat, self.m_hat  = self.optimizer.update(self.grad_w, self.w, self.v, self.m, self.v_hat, self.m_hat, layer_num)
        if self.use_bias == True:
            self.b, self.vb, self.mb, self.vb_hat, self.mb_hat  = self.optimizer.update(self.grad_b, self.b, self.vb, self.mb, self.vb_hat, self.mb_hat, layer_num)
        

