import numpy as np
from neunet.autograd import Tensor





class _RNNTensor(Tensor):
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        X, weight, weight_h, bias, states, unactivated_states, input_size, hidden_size, timesteps, nonlinearity = self.args
        X_data = X.data
        
        if len(X_data.shape) == 2:
            X_data = X_data[np.newaxis, :, :]

        if self.data.shape != states[:, 0 : -1, :].shape: # if return_sequences == "last"
            temp = np.zeros_like((states))
            temp[:, [-2], :] = grad #[-2] saves dims when slicing
            grad = temp

        next_grad_states= np.zeros((hidden_size))

        grad_weight = np.zeros_like(weight.data)
        grad_weight_h = np.zeros_like(weight_h.data)
        grad_bias = np.zeros(hidden_size)

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
            `input_size` (int): number of neurons in the input layer
            `hidden_size` (int): number of neurons in the hidden layer
            `nonlinearity` (str): activation function
            `bias` (bool):  `True` if used. `False` if not used
            `cycled_states` (bool): `True` future iteration init state equals previous iteration last state. `False` future iteration init state equals 0
            `return_sequences` (str): `"all"` return all timesteps. `"last"` return only last timestep. `"both"` return both
        Returns:
            output: data with shape (batch_size, timesteps, hidden_size)
    """

    def __init__(self, input_size, hidden_size, nonlinearity = 'tanh', bias = True, cycled_states = False, return_sequences = "both"):
        self.input_size  = input_size
        self.hidden_size   = hidden_size
        self.nonlinearity  = nonlinearities.get(nonlinearity)
        self.cycled_states = cycled_states
        self.return_sequences = return_sequences

        stdv = 1. / np.sqrt(self.hidden_size)
        self.weight = Tensor(np.random.uniform(-stdv, stdv, (self.input_size, self.hidden_size)), dtype=np.float32)
        self.weight_h = Tensor(np.random.uniform(-stdv, stdv, (self.hidden_size, self.hidden_size)), dtype=np.float32)
        
        if bias == True:
            self.bias = Tensor(np.zeros(self.hidden_size), dtype=np.float32)
        else:
            self.bias = None

        self.train = True
        self.hprev = None

    def named_parameters(self):
        return [
            ("weight", self.weight), 
            ("weight_h", self.weight_h), 
            ("bias", self.bias)
        ]



    def forward(self, X, hprev = None):
        X_data = X.data
        
        if len(X_data.shape) == 2:
            X_data = X_data[np.newaxis, :, :]
       
        batch_size, timesteps, input_size = X_data.shape

        states = np.zeros((batch_size, timesteps + 1, self.hidden_size))
        unactivated_states = np.zeros_like(states)
        
        if self.cycled_states == False:
            self.hprev = hprev
            
        assert self.hprev is None or self.hprev.shape == states[:, -1, :].shape, "hprev shape must be equal to (batch_size, 1, hidden_size)"
        assert self.input_size == input_size, "input_size must be equal to input shape[2]"
        
        if self.hprev is None: 
            self.hprev = np.zeros_like(states[:, 0, :])

        states[:, -1, :] = self.hprev.copy()

        
        for t in range(timesteps):
            unactivated_states[:, t, :] = np.dot(X_data[:, t, :], self.weight.data) + np.dot(states[:, t-1, :], self.weight_h.data) + self.bias.data if self.bias is not None else + 0
            states[:, t, :] =  self.nonlinearity.function(unactivated_states[:, t, :])

        if self.cycled_states == True:
            self.hprev = states[:, timesteps - 1, :].copy()

        all_states = states[:, 0 : -1, :]
        last_state = states[:, -2, :].reshape(batch_size, 1, self.hidden_size)

        cache = [X, self.weight, self.weight_h, self.bias, states, unactivated_states, self.input_size, self.hidden_size, timesteps, self.nonlinearity]

        if self.return_sequences in ["all", True]:
            return _RNNTensor(all_states, cache, "rnn")
        elif self.return_sequences in ["last", False]:
            return _RNNTensor(last_state, cache, "rnn")
        elif self.return_sequences == "both":
            return (_RNNTensor(all_states, cache, "rnn"), _RNNTensor(last_state, cache, "rnn"))

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