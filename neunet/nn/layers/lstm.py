import numpy as np
import cupy as cp
import neunet
from neunet.autograd import Tensor
from neunet.nn.parameter import Parameter
from neunet.nn.containers import Module


class _LSTMTensor(Tensor):
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

    def backward(self, grad=1):
        (
            X,
            weight_f,
            weight_i,
            weight_o,
            weight_c,
            weight_hf,
            weight_hi,
            weight_ho,
            weight_hc,
            bias_f,
            bias_i,
            bias_o,
            bias_c,
            forget_gates,
            input_gates,
            output_gates,
            cell_gates,
            cell_states,
            hidden_states,
            unactivated_forget_gates,
            unactivated_input_gates,
            unactivated_output_gates,
            unactivated_cell_gates,
            input_size,
            hidden_size,
            timesteps,
            nonlinearity,
            recurrent_nonlinearity,
        ) = self.args
        X_data = X.data

        if len(X_data.shape) == 2:
            X_data = X_data[self.xp.newaxis, :, :]

        if (
            self.data.shape != hidden_states[:, 0:-1, :].shape
        ):  # if return_sequences == "last"
            temp = self.xp.zeros_like((hidden_states))
            temp[:, [-2], :] = grad  # [-2] saves dims when slicing
            grad = temp

        next_hidden_delta = self.xp.zeros((hidden_size))
        next_cell_delta = self.xp.zeros((hidden_size))

        grad_weight_f = self.xp.zeros_like(weight_f.data)
        grad_weight_i = self.xp.zeros_like(weight_i.data)
        grad_weight_o = self.xp.zeros_like(weight_o.data)
        grad_weight_c = self.xp.zeros_like(weight_c.data)

        grad_weight_hf = self.xp.zeros_like(weight_hf.data)
        grad_weight_hi = self.xp.zeros_like(weight_hi.data)
        grad_weight_ho = self.xp.zeros_like(weight_ho.data)
        grad_weight_hc = self.xp.zeros_like(weight_hf.data)

        grad_bias_f = self.xp.zeros(hidden_size)
        grad_bias_i = self.xp.zeros(hidden_size)
        grad_bias_o = self.xp.zeros(hidden_size)
        grad_bias_c = self.xp.zeros(hidden_size)

        grad_X = self.xp.zeros_like(X_data)

        for t in reversed(range(timesteps)):
            hidden_delta = grad[:, t, :] + next_hidden_delta
            cell_delta = (
                hidden_delta
                * output_gates[:, t, :]
                * nonlinearity.derivative(cell_states[:, t, :])
                + next_cell_delta
            )

            output_gates_delta = (
                hidden_delta
                * nonlinearity.function(cell_states[:, t, :])
                * recurrent_nonlinearity.derivative(unactivated_output_gates[:, t, :])
            )
            grad_weight_o += self.xp.dot(X_data[:, t, :].T, output_gates_delta)
            grad_weight_ho += self.xp.dot(
                hidden_states[:, t - 1, :].T, output_gates_delta
            )
            grad_bias_o += output_gates_delta.sum(axis=0)

            forget_gates_delta = (
                cell_delta * cell_states[:, t - 1, :]
            ) * recurrent_nonlinearity.derivative(unactivated_forget_gates[:, t, :])
            grad_weight_f += self.xp.dot(X_data[:, t, :].T, forget_gates_delta)
            grad_weight_hf += self.xp.dot(
                hidden_states[:, t - 1, :].T, forget_gates_delta
            )
            grad_bias_f += forget_gates_delta.sum(axis=0)

            input_gates_delta = (
                cell_delta * cell_gates[:, t, :]
            ) * recurrent_nonlinearity.derivative(unactivated_input_gates[:, t, :])
            grad_weight_i += self.xp.dot(X_data[:, t, :].T, input_gates_delta)
            grad_weight_hi += self.xp.dot(
                hidden_states[:, t - 1, :].T, input_gates_delta
            )
            grad_bias_i += input_gates_delta.sum(axis=0)

            cell_gates_delta = (
                cell_delta * input_gates[:, t, :]
            ) * nonlinearity.derivative(unactivated_cell_gates[:, t, :])
            grad_weight_c += self.xp.dot(X_data[:, t, :].T, cell_gates_delta)
            grad_weight_hc += self.xp.dot(
                hidden_states[:, t - 1, :].T, cell_gates_delta
            )
            grad_bias_c += cell_gates_delta.sum(axis=0)

            next_hidden_delta = (
                self.xp.dot(cell_gates_delta, weight_hc.data.T)
                + self.xp.dot(input_gates_delta, weight_hi.data.T)
                + self.xp.dot(forget_gates_delta, weight_hf.data.T)
                + self.xp.dot(output_gates_delta, weight_ho.data.T)
            )
            next_cell_delta = cell_delta * forget_gates[:, t, :]

            grad_X[:, t, :] = (
                self.xp.dot(cell_gates_delta, weight_c.data.T)
                + self.xp.dot(input_gates_delta, weight_i.data.T)
                + self.xp.dot(forget_gates_delta, weight_f.data.T)
                + self.xp.dot(output_gates_delta, weight_o.data.T)
            )

        X.backward(grad_X)
        weight_f.backward(grad_weight_f)
        weight_i.backward(grad_weight_i)
        weight_o.backward(grad_weight_o)
        weight_c.backward(grad_weight_c)
        weight_hf.backward(grad_weight_hf)
        weight_hi.backward(grad_weight_hi)
        weight_ho.backward(grad_weight_ho)
        weight_hc.backward(grad_weight_hc)
        if all([bias_f, bias_i, bias_o, bias_c]):
            bias_f.backward(grad_bias_f)
            bias_i.backward(grad_bias_i)
            bias_o.backward(grad_bias_o)
            bias_c.backward(grad_bias_c)


class LSTM(Module):
    """
    Add LSTM layer
    --------------
        Args:
            `input_size` (int): number of neurons in the input layer
            `hidden_size` (int): number of neurons in the hidden layer
            `nonlinearity` (str): activation function
            `recurrent_nonlinearity` (str): activation function
            `return_sequences` (bool): if `True`, the output of the layer is a sequence of vectors, else the output is a last vector
            `bias` (bool):  `True` if used. `False` if not used
            `cycled_states` (bool): `True` if future iteration init state equals previous iteration last state. `False` if future iteration init state equals 0
        Returns:
            output: data with shape (batch_size, timesteps, hidden_size)
        References:
            https://en.wikipedia.org/wiki/Long_short-term_memory

            https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/

            https://arunmallya.github.io/writeups/nn/lstm/index.html#/

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        nonlinearity="tanh",
        recurrent_nonlinearity="sigmoid",
        return_sequences="both",
        bias=True,
        cycled_states=False,
        device="cpu",
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearities.get(nonlinearity)
        self.recurrent_nonlinearity = nonlinearities.get(recurrent_nonlinearity)

        self.return_sequences = return_sequences
        self.cycled_states = cycled_states

        stdv = 1.0 / np.sqrt(self.hidden_size)
        self.weight_f = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (self.input_size, self.hidden_size)),
                dtype=np.float32,
            )
        )
        self.weight_i = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (self.input_size, self.hidden_size)),
                dtype=np.float32,
            )
        )
        self.weight_o = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (self.input_size, self.hidden_size)),
                dtype=np.float32,
            )
        )
        self.weight_c = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (self.input_size, self.hidden_size)),
                dtype=np.float32,
            )
        )

        self.weight_hf = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (self.hidden_size, self.hidden_size)),
                dtype=np.float32,
            )
        )
        self.weight_hi = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (self.hidden_size, self.hidden_size)),
                dtype=np.float32,
            )
        )
        self.weight_ho = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (self.hidden_size, self.hidden_size)),
                dtype=np.float32,
            )
        )
        self.weight_hc = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (self.hidden_size, self.hidden_size)),
                dtype=np.float32,
            )
        )

        if bias:
            self.bias_f = Parameter(
                neunet.tensor(np.zeros(self.hidden_size), dtype=np.float32)
            )
            self.bias_i = Parameter(
                neunet.tensor(np.zeros(self.hidden_size), dtype=np.float32)
            )
            self.bias_o = Parameter(
                neunet.tensor(np.zeros(self.hidden_size), dtype=np.float32)
            )
            self.bias_c = Parameter(
                neunet.tensor(np.zeros(self.hidden_size), dtype=np.float32)
            )
        else:
            self.bias_f = None
            self.bias_i = None
            self.bias_o = None
            self.bias_c = None

        self.cprev = None
        self.hprev = None

        self.to(device)

    def named_parameters(self):
        return [
            ("weight_f", self.weight_f),
            ("weight_i", self.weight_i),
            ("weight_o", self.weight_o),
            ("weight_c", self.weight_c),
            ("weight_hf", self.weight_hf),
            ("weight_hi", self.weight_hi),
            ("weight_ho", self.weight_ho),
            ("weight_hc", self.weight_hc),
            ("bias_f", self.bias_f),
            ("bias_i", self.bias_i),
            ("bias_o", self.bias_o),
            ("bias_c", self.bias_c),
        ]

    def forward(self, X, hprev=None, cprev=None):
        assert isinstance(X, Tensor), "Input must be a tensor"
        assert X.device == self.device, "Tensors must be on the same device"
        X_data = X.data

        if len(X_data.shape) == 2:
            X_data = X_data[self.xp.newaxis, :, :]

        batch_size, timesteps, input_size = X_data.shape

        forget_gates = self.xp.zeros((batch_size, timesteps, self.hidden_size))
        unactivated_forget_gates = self.xp.zeros_like(forget_gates)

        input_gates = self.xp.zeros((batch_size, timesteps, self.hidden_size))
        unactivated_input_gates = self.xp.zeros_like(input_gates)

        output_gates = self.xp.zeros((batch_size, timesteps, self.hidden_size))
        unactivated_output_gates = self.xp.zeros_like(output_gates)

        cell_gates = self.xp.zeros((batch_size, timesteps, self.hidden_size))
        unactivated_cell_gates = self.xp.zeros_like(cell_gates)

        cell_states = self.xp.zeros((batch_size, timesteps + 1, self.hidden_size))
        hidden_states = self.xp.zeros((batch_size, timesteps + 1, self.hidden_size))

        if self.cycled_states == False:
            self.hprev = hprev
            self.cprev = cprev

        assert (
            self.hprev is None or self.hprev.shape == hidden_states[:, -1, :].shape
        ), "hprev shape must be equal to (batch_size, 1, hidden_size)"
        assert (
            self.cprev is None or self.cprev.shape == cell_states[:, -1, :].shape
        ), "cprev shape must be equal to (batch_size, 1, hidden_size)"
        assert (
            self.input_size == input_size
        ), "input_size must be equal to input shape[2]"

        if self.hprev is None:
            self.hprev = self.xp.zeros_like(hidden_states[:, 0, :])
        if self.cprev is None:
            self.cprev = self.xp.zeros_like(cell_states[:, 0, :])

        cell_states[:, -1, :] = self.cprev.copy()
        hidden_states[:, -1, :] = self.hprev.copy()

        # f_t = recurrent_nonlinearity(X_data @ self.weight_f + hs_t-1 @ self.weight_hf + self.bias_f)
        # i_t = recurrent_nonlinearity(X_data @ self.weight_i + hs_t-1 @ self.weight_hi + self.bias_i)
        # o_t = recurrent_nonlinearity(X_data @ self.weight_o + hs_t-1 @ self.weight_ho + self.bias_o)
        # c_t =           nonlinearity(X_data @ self.weight_c + hs_t-1 @ self.weight_hc + self.bias_c)
        # cs_t = f_t * cs_t-1 + i_t * c_t
        # hs_t = o_t * nonlinearity(cs_t)

        for t in range(timesteps):
            unactivated_forget_gates[:, t, :] = (
                self.xp.dot(X_data[:, t, :], self.weight_f.data)
                + self.xp.dot(hidden_states[:, t - 1, :], self.weight_hf.data)
                + self.bias_f.data
                if self.bias_f is not None
                else +0
            )
            forget_gates[:, t, :] = self.recurrent_nonlinearity.function(
                unactivated_forget_gates[:, t, :]
            )

            unactivated_input_gates[:, t, :] = (
                self.xp.dot(X_data[:, t, :], self.weight_i.data)
                + self.xp.dot(hidden_states[:, t - 1, :], self.weight_hi.data)
                + self.bias_i.data
                if self.bias_i is not None
                else +0
            )
            input_gates[:, t, :] = self.recurrent_nonlinearity.function(
                unactivated_input_gates[:, t, :]
            )

            unactivated_output_gates[:, t, :] = (
                self.xp.dot(X_data[:, t, :], self.weight_o.data)
                + self.xp.dot(hidden_states[:, t - 1, :], self.weight_ho.data)
                + self.bias_o.data
                if self.bias_o is not None
                else +0
            )
            output_gates[:, t, :] = self.recurrent_nonlinearity.function(
                unactivated_output_gates[:, t, :]
            )

            unactivated_cell_gates[:, t, :] = (
                self.xp.dot(X_data[:, t, :], self.weight_c.data)
                + self.xp.dot(hidden_states[:, t - 1, :], self.weight_hc.data)
                + self.bias_c.data
                if self.bias_c is not None
                else +0
            )
            cell_gates[:, t, :] = self.nonlinearity.function(
                unactivated_cell_gates[:, t, :]
            )

            cell_states[:, t, :] = (
                forget_gates[:, t, :] * cell_states[:, t - 1, :]
                + input_gates[:, t, :] * cell_gates[:, t, :]
            )
            hidden_states[:, t, :] = output_gates[:, t, :] * self.nonlinearity.function(
                cell_states[:, t, :]
            )

        if self.cycled_states == True:
            self.cprev = cell_states[:, timesteps - 1, :].copy()
            self.hprev = hidden_states[:, timesteps - 1, :].copy()

        all_states = hidden_states[:, 0:-1, :]
        last_state = hidden_states[:, -2, :].reshape(batch_size, 1, self.hidden_size)

        cache = [
            X,
            self.weight_f,
            self.weight_i,
            self.weight_o,
            self.weight_c,
            self.weight_hf,
            self.weight_hi,
            self.weight_ho,
            self.weight_hc,
            self.bias_f,
            self.bias_i,
            self.bias_o,
            self.bias_c,
            forget_gates,
            input_gates,
            output_gates,
            cell_gates,
            cell_states,
            hidden_states,
            unactivated_forget_gates,
            unactivated_input_gates,
            unactivated_output_gates,
            unactivated_cell_gates,
            self.input_size,
            self.hidden_size,
            timesteps,
            self.nonlinearity,
            self.recurrent_nonlinearity,
        ]

        if self.return_sequences in ["all", True]:
            return _LSTMTensor(all_states, cache, "lstm", self.device)
        elif self.return_sequences in ["last", False]:
            return _LSTMTensor(last_state, cache, "lstm", self.device)
        elif self.return_sequences == "both":
            return (
                _LSTMTensor(all_states, cache, "lstm", self.device),
                _LSTMTensor(last_state, cache, "lstm", self.device),
            )

    def __call__(self, X, hprev=None, cprev=None):
        return self.forward(X, hprev, cprev)

    def to(self, device):
        assert device in ["cpu", "cuda"], "Device must be 'cpu' or 'cuda'"
        if device == "cpu":
            self.xp = np
        else:
            self.xp = cp

        self.device = device
        for weight in self.named_parameters():
            setattr(self, weight[0], weight[1].to(device))

        return self


class NonLinearity(object):
    def function(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

    def select_lib(self, x):
        if isinstance(x, np.ndarray):
            xp = np
        else:
            xp = cp

        return xp


class Tanh(NonLinearity):
    def function(self, x):
        xp = self.select_lib(x)
        return xp.tanh(x)

    def derivative(self, x):
        xp = self.select_lib(x)
        return 1.0 - xp.power(self.function(x), 2)


class Sigmoid(NonLinearity):
    def function(self, x):
        xp = self.select_lib(x)
        return 1 / (1 + xp.exp(-x))

    def derivative(self, x):
        f_x = self.function(x)
        return f_x * (1.0 - f_x)


class ReLU(NonLinearity):
    def function(self, x):
        xp = self.select_lib(x)
        return xp.maximum(0, x)

    def derivative(self, x):
        xp = self.select_lib(x)
        return xp.where(x <= 0, 0, 1)


nonlinearities = {"tanh": Tanh(), "sigmoid": Sigmoid(), "relu": ReLU()}
