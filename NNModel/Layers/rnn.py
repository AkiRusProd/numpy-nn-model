import numpy as np


class RNN():
    #TODO
    #add accumulate gradients for batch backprop
    #maybe remove batch_size; only propagate one sample, not batch
    #add bias

    def __init__(self, units_num, activation = 'tanh', input_shape = None, return_sequences = False):
        self.units_num = units_num
        self.activation = NotImplemented
        self.activation_der = NotImplemented
        self.input_shape = input_shape
        self.return_sequences = return_sequences

        self.w = None
        self.wh = None


    def init_weights(self):
        timesteps, input_size = self.input_shape

        self.w = np.random.normal(0, 1, (input_size, self.units_num))
        self.wh = np.random.normal(0, 1, (self.units_num, self.units_num))


    def forward_prop(self, X):
        self.input_data = X
        self.batch_size, self.timesteps, self.input_size = self.input_data.shape

        self.states = np.zeros((self.batch_size, self.timesteps + 1, self.units_num))
        
        for t in range(len(self.timesteps)):
            self.states[t, :] =  self.activation(np.dot(self.input_data[t, :], self.w) + np.dot(self.states[t-1, :], self.wh))

        if self.return_sequences == False:
            return self.states[-1]
        else:
            return self.states

 

    def backward_prop(self, error):
        # batch_size, timesteps, input_size = error.shape

        if self.return_sequences == False:
            temp = np.zeros_like((self.states))
            temp[-1 :] = error
            error = temp

        error *= self.activation_der(self.states)

        grad_w = np.zeros_like(self.w)
        grad_wh = np.zeros_like(self.wh)

        next_hidden_delta = np.zeros((self.units_num))

        output_error = np.zeros((self.input_data.shape))

        for t in reversed(range(self.timesteps)):
            hidden_delta = (np.dot(next_hidden_delta, self.wh.T) + error[t, :]) *  self.act_func_der(self.states[t, :])

            grad_w  -= np.dot(self.input_data[t, :].T, hidden_delta)
            grad_wh -= np.dot(self.states[ t - 1, :].T, hidden_delta)

            next_hidden_delta = hidden_delta

            output_error[t, :] = np.dot(np.dot(error[t, :], self.wh.T), self.w.T) #* self.activation_der(self.states[:, t]) #TODO FIX: ACT FUNC OF PREV LAYER| unverified solution

   
        return output_error