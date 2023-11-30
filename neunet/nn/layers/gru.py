import numpy as np
from neunet.autograd import Tensor



class _GRUTensor(Tensor):
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        (X, weight_z, weight_r, weight_h, weight_hz, weight_hr, weight_hh, bias_z, bias_r, bias_h, 
        update_gates, reset_gates, cell_states, hidden_states, unactivated_update_gates, unactivated_reset_gates, unactivated_cell_states, 
        input_size, hidden_size, timesteps, nonlinearity, recurrent_nonlinearity) = self.args
        X_data = X.data
        
        if len(X_data.shape) == 2:
            X_data = X_data[np.newaxis, :, :]

        if self.data.shape != hidden_states[:, 0 : -1, :].shape: # if return_sequences == "last"
            temp = np.zeros_like((hidden_states))
            temp[:, [-2], :] = grad #[-2] saves dims when slicing
            grad = temp

        next_hidden_delta = np.zeros((hidden_size))

        grad_weight_z = np.zeros_like(weight_z.data)
        grad_weight_r = np.zeros_like(weight_r.data)
        grad_weight_h = np.zeros_like(weight_h.data)

        grad_weight_hz = np.zeros_like(weight_hz.data)
        grad_weight_hr = np.zeros_like(weight_hr.data)
        grad_weight_hh = np.zeros_like(weight_hh.data)

        grad_bias_z = np.zeros(hidden_size)
        grad_bias_r = np.zeros(hidden_size)
        grad_bias_h = np.zeros(hidden_size)

        grad_X = np.zeros_like(X_data)

        for t in reversed(range(timesteps)):
            
            hidden_delta = grad[:, t, :] + next_hidden_delta
            
            cell_gates_delta = hidden_delta * (1 - update_gates[:, t, :]) * nonlinearity.derivative(unactivated_cell_states[:, t, :])
            grad_weight_h   += np.dot(X_data[:, t, :].T,  cell_gates_delta)
            grad_weight_hh  += np.dot(hidden_states[:, t - 1, :].T * reset_gates[:, t, :].T, cell_gates_delta)
            grad_bias_h   += cell_gates_delta.sum(axis=0)
        
            reset_gates_delta = np.dot(cell_gates_delta, weight_hh.data.T) * hidden_states[:, t - 1, :] * recurrent_nonlinearity.derivative(unactivated_reset_gates[:, t, :])
            
            grad_weight_r   += np.dot(X_data[:, t, :].T,  reset_gates_delta)
            grad_weight_hr  += np.dot(hidden_states[:, t - 1, :].T, reset_gates_delta)
            grad_bias_r   += reset_gates_delta.sum(axis=0)

            update_gates_delta = hidden_delta * (hidden_states[:, t - 1, :] - cell_states[:, t, :]) * recurrent_nonlinearity.derivative(unactivated_update_gates[:, t, :])
            grad_weight_z   += np.dot(X_data[:, t, :].T,  update_gates_delta)
            grad_weight_hz  += np.dot(hidden_states[:, t - 1, :].T,  update_gates_delta)
            grad_bias_z   += update_gates_delta.sum(axis=0)


            next_hidden_delta =  np.dot(update_gates_delta, weight_hz.data.T) + np.dot(reset_gates_delta, weight_hr.data.T) + np.dot(cell_gates_delta, weight_hh.data.T) * reset_gates[:, t, :] + hidden_delta * update_gates[:, t, :] 
    
            grad_X[:, t, :] = np.dot(cell_gates_delta, weight_h.data.T) + np.dot(update_gates_delta, weight_z.data.T) + np.dot(reset_gates_delta, weight_r.data.T)

        X.backward(grad_X)
        weight_z.backward(grad_weight_z)
        weight_r.backward(grad_weight_r)
        weight_h.backward(grad_weight_h)
        weight_hz.backward(grad_weight_hz)
        weight_hr.backward(grad_weight_hr)
        weight_hh.backward(grad_weight_hh)
        if all([bias_z, bias_r, bias_h]):
            bias_z.backward(grad_bias_z)
            bias_r.backward(grad_bias_r)
            bias_h.backward(grad_bias_h)












class GRU():
    """
    Add GRU layer
    --------------
        Args:
            `input_size` (int): number of neurons in the input layer
            `hidden_size` (int): number of neurons in the hidden layer
            `nonlinearity` (str): activation function
            `recurrent_nonlinearity`: activation function
            `return_sequences` (bool): if `True`, the output of the layer is a sequence of vectors, else the output is a last vector
            `bias` (bool):  `True` if used. `False` if not used
            `cycled_states` (bool): `True` if future iteration init state equals previous iteration last state. `False` if future iteration init state equals 0
        Returns:
            output: data with shape (batch_size, timesteps, hidden_size)
        References:
            https://arxiv.org/pdf/1612.07778.pdf
            
            https://github.com/erikvdplas/gru-rnn/blob/master/main.py

            https://shuby.de/blog/post/comp_graphs/
        
    """

    def __init__(self, input_size, hidden_size, nonlinearity = 'tanh', recurrent_nonlinearity = 'sigmoid',  return_sequences = "both", bias = True, cycled_states = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity  = nonlinearities.get(nonlinearity)
        self.recurrent_nonlinearity  = nonlinearities.get(recurrent_nonlinearity)
       

        self.return_sequences = return_sequences
        self.cycled_states = cycled_states


        stdv = 1. / np.sqrt(self.hidden_size)
        self.weight_z = Tensor(np.random.uniform(-stdv, stdv, (self.input_size, self.hidden_size)), dtype=np.float32)
        self.weight_r = Tensor(np.random.uniform(-stdv, stdv, (self.input_size, self.hidden_size)), dtype=np.float32)
        self.weight_h = Tensor(np.random.uniform(-stdv, stdv, (self.input_size, self.hidden_size)), dtype=np.float32)

        self.weight_hz = Tensor(np.random.uniform(-stdv, stdv, (self.hidden_size, self.hidden_size)), dtype=np.float32)
        self.weight_hr = Tensor(np.random.uniform(-stdv, stdv, (self.hidden_size, self.hidden_size)), dtype=np.float32)
        self.weight_hh = Tensor(np.random.uniform(-stdv, stdv, (self.hidden_size, self.hidden_size)), dtype=np.float32)

        if bias:
            self.bias_z = Tensor(np.zeros(self.hidden_size), dtype=np.float32)
            self.bias_r = Tensor(np.zeros(self.hidden_size), dtype=np.float32)
            self.bias_h = Tensor(np.zeros(self.hidden_size), dtype=np.float32)
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None

        self.cprev = None
        self.hprev = None

    def named_parameters(self):
        return [
            ('weight_z', self.weight_z),
            ('weight_r', self.weight_r),
            ('weight_h', self.weight_h),
            ('weight_hz', self.weight_hz),
            ('weight_hr', self.weight_hr),
            ('weight_hh', self.weight_hh),
            ('bias_z', self.bias_z),
            ('bias_r', self.bias_r),
            ('bias_h', self.bias_h),
        ]


    def forward(self, X, hprev = None, cprev = None):
        X_data = X.data

        if len(X_data.shape) == 2:
            X_data = X_data[np.newaxis, :, :]
       
        batch_size, timesteps, input_size = X_data.shape

        update_gates = np.zeros((batch_size, timesteps, self.hidden_size))
        unactivated_update_gates = np.zeros_like(update_gates)

        reset_gates = np.zeros((batch_size, timesteps, self.hidden_size))
        unactivated_reset_gates = np.zeros_like(reset_gates)

        cell_states = np.zeros((batch_size, timesteps + 1, self.hidden_size))
        unactivated_cell_states = np.zeros_like(cell_states)

        hidden_states = np.zeros((batch_size, timesteps + 1, self.hidden_size))

        if self.cycled_states == False:
            self.hprev = hprev
            self.cprev = cprev

        assert self.hprev is None or self.hprev.shape == hidden_states[:, -1, :].shape, "hprev shape must be equal to (batch_size, 1, hidden_size)"
        assert self.cprev is None or self.cprev.shape == cell_states[:, -1, :].shape, "cprev shape must be equal to (batch_size, 1, hidden_size)"
        assert self.input_size == input_size, "input_size must be equal to input shape[2]"

        if self.hprev is None: 
            self.hprev = np.zeros_like(hidden_states[:, 0, :])
        if self.cprev is None: 
            self.cprev = np.zeros_like(cell_states[:, 0, :])


        cell_states[:, -1, :] = self.cprev.copy()
        hidden_states[:, -1, :] = self.hprev.copy()
        #Z_t = recurrent_nonlinearity(x_t * weight_z + h_t-1 * U_z + bias_z)
        #R_t = recurrent_nonlinearity(x_t * weight_r + h_t-1 * U_r + bias_r)
        #C_t = nonlinearity(x_t * weight_h + R_t * h_t-1 * U_h + bias_h)
        #H_t = z_t * H_t-1 + (1 - z_t) * C_t

        
        for t in range(timesteps):
            unactivated_update_gates[:, t, :] = np.dot(X_data[:, t, :], self.weight_z.data) + np.dot(hidden_states[:, t-1, :], self.weight_hz.data) + self.bias_z.data if self.bias_z is not None else 0
            update_gates[:, t, :] = self.recurrent_nonlinearity.function(unactivated_update_gates[:, t, :])

            unactivated_reset_gates[:, t, :] = np.dot(X_data[:, t, :], self.weight_r.data) + np.dot(hidden_states[:, t-1, :], self.weight_hr.data) + self.bias_r.data if self.bias_r is not None else 0
            reset_gates[:, t, :]  = self.recurrent_nonlinearity.function(unactivated_reset_gates[:, t, :])

            unactivated_cell_states[:, t, :] = np.dot(X_data[:, t, :], self.weight_h.data) + np.dot(reset_gates[:, t, :] * hidden_states[:, t - 1, :], self.weight_hh.data) + self.bias_h.data if self.bias_h is not None else 0
            cell_states[:, t, :] = self.nonlinearity.function(unactivated_cell_states[:, t, :])

            hidden_states[:, t, :] = update_gates[:, t, :] * hidden_states[:, t - 1, :] + (1 - update_gates[:, t, :]) * cell_states[:, t, :]

        if self.cycled_states == True:
            self.cprev = cell_states[:, timesteps - 1, :].copy()
            self.hprev = hidden_states[:, timesteps - 1, :].copy()

        all_states = hidden_states[:, 0 : -1, :]
        last_state = hidden_states[:, -2, :].reshape(batch_size, 1, self.hidden_size)

        cache = [X, self.weight_z, self.weight_r, self.weight_h, self.weight_hz, self.weight_hr, self.weight_hh, self.bias_z, self.bias_r, self.bias_h, 
        update_gates, reset_gates, cell_states, hidden_states, unactivated_update_gates, unactivated_reset_gates, unactivated_cell_states, 
        self.input_size, self.hidden_size, timesteps, self.nonlinearity, self.recurrent_nonlinearity]

        if self.return_sequences in ["all", True]:
            return _GRUTensor(all_states, cache, "GRU")
        elif self.return_sequences in ["last", False]:
            return _GRUTensor(last_state, cache, "GRU")
        elif self.return_sequences == "both":
            return (_GRUTensor(all_states, cache, "GRU"), _GRUTensor(last_state, cache, "GRU"))


    def __call__(self, X, hprev = None, cprev = None):
        return self.forward(X, hprev, cprev)





            
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