import numpy as np
from nnmodel.activations import activations
from nnmodel.exceptions.values_checker import ValuesChecker

class RNN():
    """
    Add Vanilla RNN layer
    ---------------------
        Args:
            `units_num` (int): number of neurons in the layer
            `activation` (str) or (`ActivationFunction` class): activation function
            `return_sequences` (bool): if `True`, the output of the layer is a sequence of vectors, else the output is a last vector
            `use_bias` (bool):  `True` if used. `False` if not used
            `cycled_states` (bool): `True` if future iteration init state equals previous iteration last state. `False` if future iteration init state equals 0
        Returns:
            output: data with shape (batch_size, timesteps, units_num)
    """

    def __init__(self, units_num, activation = 'tanh', input_shape = None, return_sequences = False, use_bias = True, cycled_states = True):
        self.units_num   = ValuesChecker.check_integer_variable(units_num, "units_num")
        self.input_shape = ValuesChecker.check_input_dim(input_shape, input_dim = 2)
        self.activation  = ValuesChecker.check_activation(activation, activations)
        self.use_bias    = ValuesChecker.check_boolean_type(use_bias, "use_bias")

        self.return_sequences = ValuesChecker.check_boolean_type(return_sequences, "cycled_states")
        self.cycled_states = ValuesChecker.check_boolean_type(cycled_states, "return_sequences")
        self.w = None
        self.wh = None
        self.b = None

        self.optimizer = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


    def build(self):
        
        self.timesteps, self.input_size = self.input_shape

        self.w = np.random.normal(0, pow(self.input_size, -0.5), (self.input_size, self.units_num))
        self.wh = np.random.normal(0, pow(self.units_num, -0.5), (self.units_num, self.units_num))
        self.b = np.zeros(self.units_num)

        self.v, self.m         = np.zeros_like(self.w), np.zeros_like(self.w) # optimizers params
        self.v_hat, self.m_hat = np.zeros_like(self.w), np.zeros_like(self.w) # optimizers params

        self.vh, self.mh         = np.zeros_like(self.wh), np.zeros_like(self.wh) # optimizers params
        self.vh_hat, self.mh_hat = np.zeros_like(self.wh), np.zeros_like(self.wh) # optimizers params

        self.vb, self.mb         = np.zeros_like(self.b), np.zeros_like(self.b) # optimizers params
        self.vb_hat, self.mb_hat = np.zeros_like(self.b), np.zeros_like(self.b) # optimizers params
        self.hprev = None

        self.output_shape = (1, self.units_num) if self.return_sequences == False else (self.timesteps, self.units_num)


    def forward_prop(self, X, training):
        self.input_data = X
        self.batch_size = len(self.input_data)  
        if len(self.input_data.shape) == 2:
            self.input_data = self.input_data[:, np.newaxis, :]
       
        # self.batch_size, self.timesteps = self.input_data.shape[0], self.input_data.shape[1]
        
        self.states = np.zeros((self.batch_size, self.timesteps + 1, self.units_num))
        self.unactivated_states = np.zeros_like(self.states)
        
        if self.hprev is None: self.hprev = np.zeros_like(self.states[:, 0, :])
        if self.cycled_states == True and training == True and self.states[:, -1, :].shape == self.hprev.shape:
            self.states[:, -1, :] = self.hprev.copy()
        
        for t in range(self.timesteps):
            self.unactivated_states[:, t, :] = np.dot(self.input_data[:, t, :], self.w) + np.dot(self.states[:, t-1, :], self.wh) + self.b
            self.states[:, t, :] =  self.activation.function(self.unactivated_states[:, t, :])

        
        self.hprev = self.states[:, self.timesteps - 1, :].copy()
        if self.return_sequences == False:
           
            return self.states[:, -2, :]
        else:
            
            return self.states[:, 0 : -1, :]

 

    def backward_prop(self, error):

        if self.return_sequences == False:
            temp = np.zeros_like((self.states))
            temp[:, -2, :] = error
            error = temp

        # if len(error.shape) == 2:
        #     error = error[:, np.newaxis, :]
        # self.batch_size, self.timesteps, self.input_size = error.shape

        self.grad_w = np.zeros_like(self.w)
        self.grad_wh = np.zeros_like(self.wh)
        self.grad_b = np.zeros_like(self.b)

        next_hidden_delta = np.zeros((self.units_num))

        output_error = np.zeros((self.input_data.shape))

        for t in reversed(range(self.timesteps)):
            hidden_delta = (next_hidden_delta + error[:, t, :]) *  self.activation.derivative(self.unactivated_states[:, t, :])

            self.grad_w  += np.dot(self.input_data[:, t, :].T, hidden_delta)
            self.grad_wh += np.dot(self.states[:, t - 1, :].T, hidden_delta)
            self.grad_b += np.sum(error[:, t, :], axis = 0)

            next_hidden_delta = np.dot(hidden_delta, self.wh.T)

            output_error[:, t, :] = np.dot(np.dot(error[:, t, :], self.wh.T), self.w.T)

   
        return output_error


    def update_weights(self, layer_num):
        self.w, self.v, self.m, self.v_hat, self.m_hat  = self.optimizer.update(self.grad_w, self.w, self.v, self.m, self.v_hat, self.m_hat, layer_num)
        self.wh, self.vh, self.mh, self.vh_hat, self.mh_hat = self.optimizer.update(self.grad_wh, self.wh, self.vh, self.mh, self.vh_hat, self.mh_hat, layer_num)
        if self.use_bias == True:
            self.b, self.vb, self.mb, self.vb_hat, self.mb_hat  = self.optimizer.update(self.grad_b, self.b, self.vb, self.mb, self.vb_hat, self.mb_hat, layer_num)

    def get_grads(self):
        return self.grad_w, self.grad_wh, self.grad_b

    def set_grads(self, grads):
        self.grad_w, self.grad_wh, self.grad_b = grads
        