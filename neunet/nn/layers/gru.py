from typing import Any, Union

import cupy as cp
import numpy as np

import neunet
from neunet.autograd import Tensor
from neunet.nn.modules import Module
from neunet.nn.parameter import Parameter


class _GRUTensor(Tensor):
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(
                X: Tensor,
                weight_z: Tensor,
                weight_r: Tensor,
                weight_h: Tensor,
                weight_hz: Tensor,
                weight_hr: Tensor,
                weight_hh: Tensor,
                bias_z: Tensor,
                bias_r: Tensor,
                bias_h: Tensor,
                update_gates,
                reset_gates,
                cell_states,
                hidden_states,
                unactivated_update_gates,
                unactivated_reset_gates,
                unactivated_cell_states,
                hidden_size,
                timesteps,
                nonlinearity,
                recurrent_nonlinearity,
                grad
            ):
            X_data = X.data

            if len(X_data.shape) == 2:
                X_data = X_data[X.xp.newaxis, :, :]

            if self.data.shape != hidden_states[:, 0:-1, :].shape:  # if return_sequences == "last" # NOTE: self is here (potential memory leak)
                temp = X.xp.zeros_like((hidden_states))
                temp[:, [-2], :] = grad  # [-2] saves dims when slicing
                grad = temp

            next_hidden_delta = X.xp.zeros((hidden_size), dtype=grad.dtype)

            grad_weight_z = X.xp.zeros_like(weight_z.data)
            grad_weight_r = X.xp.zeros_like(weight_r.data)
            grad_weight_h = X.xp.zeros_like(weight_h.data)

            grad_weight_hz = X.xp.zeros_like(weight_hz.data)
            grad_weight_hr = X.xp.zeros_like(weight_hr.data)
            grad_weight_hh = X.xp.zeros_like(weight_hh.data)

            grad_bias_z = X.xp.zeros(hidden_size, dtype=grad.dtype)
            grad_bias_r = X.xp.zeros(hidden_size, dtype=grad.dtype)
            grad_bias_h = X.xp.zeros(hidden_size, dtype=grad.dtype)

            grad_X = X.xp.zeros_like(X_data)

            for t in reversed(range(timesteps)):
                hidden_delta = grad[:, t, :] + next_hidden_delta

                cell_gates_delta = (
                    hidden_delta
                    * (1 - update_gates[:, t, :])
                    * nonlinearity.derivative(unactivated_cell_states[:, t, :])
                )
                grad_weight_h += X.xp.dot(X_data[:, t, :].T, cell_gates_delta)
                grad_weight_hh += X.xp.dot(
                    hidden_states[:, t - 1, :].T * reset_gates[:, t, :].T, cell_gates_delta
                )
                grad_bias_h += cell_gates_delta.sum(axis=0)

                reset_gates_delta = (
                    X.xp.dot(cell_gates_delta, weight_hh.data.T)
                    * hidden_states[:, t - 1, :]
                    * recurrent_nonlinearity.derivative(unactivated_reset_gates[:, t, :])
                )

                grad_weight_r += X.xp.dot(X_data[:, t, :].T, reset_gates_delta)
                grad_weight_hr += X.xp.dot(hidden_states[:, t - 1, :].T, reset_gates_delta)
                grad_bias_r += reset_gates_delta.sum(axis=0)

                update_gates_delta = (
                    hidden_delta
                    * (hidden_states[:, t - 1, :] - cell_states[:, t, :])
                    * recurrent_nonlinearity.derivative(unactivated_update_gates[:, t, :])
                )
                grad_weight_z += X.xp.dot(X_data[:, t, :].T, update_gates_delta)
                grad_weight_hz += X.xp.dot(hidden_states[:, t - 1, :].T, update_gates_delta)
                grad_bias_z += update_gates_delta.sum(axis=0)

                next_hidden_delta = (
                    X.xp.dot(update_gates_delta, weight_hz.data.T)
                    + X.xp.dot(reset_gates_delta, weight_hr.data.T)
                    + X.xp.dot(cell_gates_delta, weight_hh.data.T) * reset_gates[:, t, :]
                    + hidden_delta * update_gates[:, t, :]
                )

                grad_X[:, t, :] = (
                    X.xp.dot(cell_gates_delta, weight_h.data.T)
                    + X.xp.dot(update_gates_delta, weight_z.data.T)
                    + X.xp.dot(reset_gates_delta, weight_r.data.T)
                )

            X.apply_grad(grad_X)
            weight_z.apply_grad(grad_weight_z)
            weight_r.apply_grad(grad_weight_r)
            weight_h.apply_grad(grad_weight_h)
            weight_hz.apply_grad(grad_weight_hz)
            weight_hr.apply_grad(grad_weight_hr)
            weight_hh.apply_grad(grad_weight_hh)
            if all([bias_z, bias_r, bias_h]):
                bias_z.apply_grad(grad_bias_z)
                bias_r.apply_grad(grad_bias_r)
                bias_h.apply_grad(grad_bias_h)

        self.grad_fn = grad_fn


class GRU(Module):
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

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nonlinearity: str="tanh",
        recurrent_nonlinearity: str="sigmoid",
        return_sequences: Union[str, bool]="both",
        bias: bool=True,
        cycled_states: bool=False,
        device: str="cpu",
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity: Union[NonLinearity, Any] = nonlinearities.get(nonlinearity) 
        self.recurrent_nonlinearity: Union[NonLinearity, Any] = nonlinearities.get(recurrent_nonlinearity) 

        self.return_sequences = return_sequences
        self.cycled_states = cycled_states

        stdv = 1.0 / np.sqrt(self.hidden_size)
        self.weight_z = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (self.input_size, self.hidden_size)),
                dtype=np.float32,
            )
        )
        self.weight_r = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (self.input_size, self.hidden_size)),
                dtype=np.float32,
            )
        )
        self.weight_h = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (self.input_size, self.hidden_size)),
                dtype=np.float32,
            )
        )

        self.weight_hz = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (self.hidden_size, self.hidden_size)),
                dtype=np.float32,
            )
        )
        self.weight_hr = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (self.hidden_size, self.hidden_size)),
                dtype=np.float32,
            )
        )
        self.weight_hh = Parameter(
            neunet.tensor(
                np.random.uniform(-stdv, stdv, (self.hidden_size, self.hidden_size)),
                dtype=np.float32,
            )
        )

        if bias:
            self.bias_z: Union[Tensor, None]  = Parameter(neunet.tensor(np.zeros(self.hidden_size), dtype=np.float32))
            self.bias_r: Union[Tensor, None]  = Parameter(neunet.tensor(np.zeros(self.hidden_size), dtype=np.float32))
            self.bias_h: Union[Tensor, None]  = Parameter(neunet.tensor(np.zeros(self.hidden_size), dtype=np.float32))
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None

        self.cprev = None
        self.hprev = None

        self.to(device)

    def forward(self, X: Tensor, hprev: Any=None, cprev: Any=None) -> Union[Tensor, tuple[Tensor, Tensor]]:
        if not isinstance(X, Tensor):
            raise TypeError("Input must be a tensor")
        if X.device != self.device:
            raise ValueError("Tensors must be on the same device")

        X_data = X.data

        if len(X_data.shape) == 2:
            X_data = X_data[self.xp.newaxis, :, :]

        batch_size, timesteps, input_size = X_data.shape

        update_gates = self.xp.zeros((batch_size, timesteps, self.hidden_size), dtype=X_data.dtype)
        unactivated_update_gates = self.xp.zeros_like(update_gates)

        reset_gates = self.xp.zeros((batch_size, timesteps, self.hidden_size), dtype=X_data.dtype)
        unactivated_reset_gates = self.xp.zeros_like(reset_gates)

        cell_states = self.xp.zeros(
            (batch_size, timesteps + 1, self.hidden_size), dtype=X_data.dtype
        )
        unactivated_cell_states = self.xp.zeros_like(cell_states)

        hidden_states = self.xp.zeros(
            (batch_size, timesteps + 1, self.hidden_size), dtype=X_data.dtype
        )

        if self.cycled_states == False:
            self.hprev = hprev
            self.cprev = cprev

        if self.hprev is not None and self.hprev.shape != hidden_states[:, -1, :].shape:
            raise ValueError("hprev shape must be equal to (batch_size, 1, hidden_size)")
        if self.cprev is not None and self.cprev.shape != cell_states[:, -1, :].shape:
            raise ValueError("cprev shape must be equal to (batch_size, 1, hidden_size)")
        if self.input_size != input_size:
            raise ValueError("input_size must be equal to input shape[2]")

        if self.hprev is None:
            self.hprev = self.xp.zeros_like(hidden_states[:, 0, :])
        if self.cprev is None:
            self.cprev = self.xp.zeros_like(cell_states[:, 0, :])

        cell_states[:, -1, :] = self.cprev.copy() # type: ignore
        hidden_states[:, -1, :] = self.hprev.copy() # type: ignore
        # Z_t = recurrent_nonlinearity(x_t * weight_z + h_t-1 * U_z + bias_z)
        # R_t = recurrent_nonlinearity(x_t * weight_r + h_t-1 * U_r + bias_r)
        # C_t = nonlinearity(x_t * weight_h + R_t * h_t-1 * U_h + bias_h)
        # H_t = z_t * H_t-1 + (1 - z_t) * C_t

        for t in range(timesteps):
            unactivated_update_gates[:, t, :] = (
                self.xp.dot(X_data[:, t, :], self.weight_z.data)
                + self.xp.dot(hidden_states[:, t - 1, :], self.weight_hz.data)
                + self.bias_z.data
                if self.bias_z is not None
                else 0
            )
            update_gates[:, t, :] = self.recurrent_nonlinearity.function(
                unactivated_update_gates[:, t, :]
            )

            unactivated_reset_gates[:, t, :] = (
                self.xp.dot(X_data[:, t, :], self.weight_r.data)
                + self.xp.dot(hidden_states[:, t - 1, :], self.weight_hr.data)
                + self.bias_r.data
                if self.bias_r is not None
                else 0
            )
            reset_gates[:, t, :] = self.recurrent_nonlinearity.function(
                unactivated_reset_gates[:, t, :]
            )

            unactivated_cell_states[:, t, :] = (
                self.xp.dot(X_data[:, t, :], self.weight_h.data)
                + self.xp.dot(
                    reset_gates[:, t, :] * hidden_states[:, t - 1, :],
                    self.weight_hh.data,
                )
                + self.bias_h.data
                if self.bias_h is not None
                else 0
            )
            cell_states[:, t, :] = self.nonlinearity.function(unactivated_cell_states[:, t, :])

            hidden_states[:, t, :] = (
                update_gates[:, t, :] * hidden_states[:, t - 1, :]
                + (1 - update_gates[:, t, :]) * cell_states[:, t, :]
            )

        if self.cycled_states == True:
            self.cprev = cell_states[:, timesteps - 1, :].copy()
            self.hprev = hidden_states[:, timesteps - 1, :].copy()

        all_states = hidden_states[:, 0:-1, :]
        last_state = hidden_states[:, -2, :].reshape(batch_size, 1, self.hidden_size)

        cache = [
            X,
            self.weight_z,
            self.weight_r,
            self.weight_h,
            self.weight_hz,
            self.weight_hr,
            self.weight_hh,
            self.bias_z,
            self.bias_r,
            self.bias_h,
            update_gates,
            reset_gates,
            cell_states,
            hidden_states,
            unactivated_update_gates,
            unactivated_reset_gates,
            unactivated_cell_states,
            self.hidden_size,
            timesteps,
            self.nonlinearity,
            self.recurrent_nonlinearity,
        ]

        if self.return_sequences in ["all", True]:
            return _GRUTensor(all_states, cache, "GRU", self.device)
        elif self.return_sequences in ["last", False]:
            return _GRUTensor(last_state, cache, "GRU", self.device)
       
        return (
            _GRUTensor(all_states, cache, "GRU", self.device),
            _GRUTensor(last_state, cache, "GRU", self.device),
        )

    def __call__(self, X, hprev=None, cprev=None):
        return self.forward(X, hprev, cprev)


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
