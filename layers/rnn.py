import numpy as np
from autograd import Tensor





class _RNNTensor(Tensor):
    def __init__(self, data, args, op):
        self.data = data
        self.args = args
        self.op = op

    def backward(self, grad=1):
        X, weight, weight_h, bias, states, unactivated_states, input_size, hidden_size, timesteps, nonlinearity, X_data = self.args

        if self.data.shape != states[:, 0 : -1, :].shape: # if return_sequences == False
            temp = np.zeros_like((states))
            temp[:, -2, :] = grad
            grad = temp

        next_grad_states= np.zeros((hidden_size))

        grad_weight = np.zeros_like(weight.data)
        grad_weight_h = np.zeros_like(weight_h.data)
        grad_bias = np.zeros_like(bias.data)

        grad_X = np.zeros_like(X_data)

        for t in reversed(range(timesteps)):
            grad_states = (next_grad_states + grad[:, t, :]) *  nonlinearity.derivative(unactivated_states[:, t, :])

            grad_weight += np.dot(X_data[:, t, :].T, grad_states)
            grad_weight_h += np.dot(states[:, t - 1, :].T, grad_states)
            grad_bias += np.sum(grad_states, axis = 0)

            grad_X[:, t, :] = np.dot(grad_states, weight.data.T)
            next_grad_states = np.dot(grad_states, weight_h.data.T)


        X.backward(grad_X.reshape(X.shape))

        weight.backward(grad_weight)
        weight_h.backward(grad_weight_h)
        if bias is not None:
            bias.backward(grad_bias)



class RNN():
    """
    Add Vanilla RNN layer
    ---------------------
        Args:
            `hidden_size` (int): number of neurons in the layer
            `nonlinearity` (str) or (`Activation Function` class): activation function
            `use_bias` (bool):  `True` if used. `False` if not used
            `cycled_states` (bool): `True` future iteration init state equals previous iteration last state. `False` future iteration init state equals 0
        Returns:
            output: data with shape (batch_size, timesteps, hidden_size)
    """

    def __init__(self, input_size, hidden_size, nonlinearity = 'tanh', input_shape = None, bias = True, cycled_states = False, return_sequences = True):
        self.input_size  = input_size
        self.hidden_size   = hidden_size
        self.input_shape = input_shape
        self.nonlinearity  = nonlinearities.get(nonlinearity)
        self.cycled_states = cycled_states
        self.return_sequences = return_sequences

        stdv = 1. / np.sqrt(self.input_size)
        self.weight = Tensor(np.random.uniform(-stdv, stdv, (self.input_size, self.hidden_size)))
        self.weight_h = Tensor(np.random.uniform(-stdv, stdv, (self.hidden_size, self.hidden_size)))
        
        if bias == True:
            self.bias = Tensor(np.zeros(self.hidden_size))
        else:
            self.bias = None

        self.train = True
        self.hprev = None



    def forward(self, X, hprev = None):
        self.X_data = X.data
        
        if len(self.X_data.shape) == 2:
            self.X_data = self.X_data[:, np.newaxis, :]
       
        batch_size, timesteps, input_size = self.X_data.shape

        self.states = np.zeros((batch_size, timesteps + 1, self.hidden_size))
        self.unactivated_states = np.zeros_like(self.states)
        
        if self.cycled_states == False:
            self.hprev = hprev
            
        assert self.hprev is None or self.hprev.shape == self.states[:, -1, :].shape, "hprev shape must be equal to (batch_size, 1, hidden_size)"
        assert self.input_size == input_size, "input_size must be equal to input shape[2]"
        
        if self.hprev is None: 
            self.hprev = np.zeros_like(self.states[:, 0, :])

        self.states[:, -1, :] = self.hprev.copy()

        
        for t in range(timesteps):
            self.unactivated_states[:, t, :] = np.dot(self.X_data[:, t, :], self.weight.data) + np.dot(self.states[:, t-1, :], self.weight_h.data) + self.bias.data if self.bias is not None else + 0
            self.states[:, t, :] =  self.nonlinearity.function(self.unactivated_states[:, t, :])

        if self.cycled_states == True:
            self.hprev = self.states[:, timesteps - 1, :].copy()

        all_states = self.states[:, 0 : -1, :]
        last_state = self.states[:, -2, :].reshape(batch_size, 1, self.hidden_size)

        return (_RNNTensor(all_states, [X, self.weight, self.weight_h, self.bias, self.states, self.unactivated_states, self.input_size, self.hidden_size, timesteps, self.nonlinearity, self.X_data],  "rnn"),
                _RNNTensor(last_state, [X, self.weight, self.weight_h, self.bias, self.states, self.unactivated_states, self.input_size, self.hidden_size, timesteps, self.nonlinearity, self.X_data],  "rnn"))

    def __call__(self, X, hprev = None):
        return self.forward(X, hprev)
    



class Tanh():
    def function(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1.0 - np.power(self.function(x), 2)

class Sigmoid():
    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        f_x = self.function(x)
        return f_x * (1.0 - f_x)

class ReLU():
    def function(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x <= 0, 0, 1)


nonlinearities = {
    'tanh': Tanh(),
    'sigmoid': Sigmoid(),
    'relu': ReLU()
}