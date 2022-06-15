import numpy as np
from NNModel.activations import activations


class RNN():
    #TODO
    #add accumulate gradients for batch backprop
    #maybe remove batch_size; only propagate one sample, not batch
    #add bias

    def __init__(self, units_num, activation = 'tanh', input_shape = None, return_sequences = False):
        self.units_num = units_num
        self.input_shape = input_shape

        if type(activation) is str:
            self.activation = activations[activation]
        else:
            self.activation = activation
    

        self.return_sequences = return_sequences

        self.w = None
        self.wh = None


    def build(self, optimizer):
        self.optimizer = optimizer
        self.timesteps, self.input_size = self.input_shape

        self.w = np.random.normal(0, pow(self.input_size, -0.5), (self.input_size, self.units_num))
        self.wh = np.random.normal(0, pow(self.units_num, -0.5), (self.units_num, self.units_num))

        self.v, self.m         = np.zeros_like(self.w), np.zeros_like(self.w) # optimizers params
        self.v_hat, self.m_hat = np.zeros_like(self.w), np.zeros_like(self.w) # optimizers params

        self.vh, self.mh         = np.zeros_like(self.w), np.zeros_like(self.w) # optimizers params
        self.vh_hat, self.mh_hat = np.zeros_like(self.w), np.zeros_like(self.w) # optimizers params

        self.output_shape = (1, self.units_num) if self.return_sequences == False else (self.timesteps, self.units_num)


    def forward_prop(self, X):
        self.input_data = X
        self.batch_size = len(self.input_data)
        # self.batch_size, self.timesteps, self.input_size = self.input_data.shape

        self.states = np.zeros((self.batch_size, self.timesteps + 1, self.units_num))
        
        for t in range(len(self.timesteps)):
            self.states[t, :] =  self.activation.function(np.dot(self.input_data[t, :], self.w) + np.dot(self.states[t-1, :], self.wh))

        if self.return_sequences == False:
            return self.states[-1]
        else:
            return self.states[:-1]

 

    def backward_prop(self, error):
        # batch_size, timesteps, input_size = error.shape

        if self.return_sequences == False:
            temp = np.zeros_like((self.states))
            temp[-1 :] = error
            error = temp

        error *= self.activation.derivative(self.states)

        self.grad_w = np.zeros_like(self.w)
        self.grad_wh = np.zeros_like(self.wh)

        next_hidden_delta = np.zeros((self.units_num))

        output_error = np.zeros((self.input_data.shape))

        for t in reversed(range(self.timesteps)):
            hidden_delta = (np.dot(next_hidden_delta, self.wh.T) + error[t, :]) *  self.activation.derivative(self.states[t, :])

            self.grad_w  -= np.dot(self.input_data[t, :].T, hidden_delta)
            self.grad_wh -= np.dot(self.states[ t - 1, :].T, hidden_delta)

            next_hidden_delta = hidden_delta

            output_error[t, :] = np.dot(np.dot(error[t, :], self.wh.T), self.w.T) #* self.activation_der(self.states[:, t]) #TODO FIX: ACT FUNC OF PREV LAYER| unverified solution

   
        return output_error


    def update_weights(self, layer_num):
        self.w  = self.optimizer.update(self.grad_w, self.w, self.v, self.m, self.v_hat, self.m_hat, layer_num)
        self.wh = self.optimizer.update(self.grad_wh, self.wh, self.vh, self.mh, self.vh_hat, self.mh_hat, layer_num)