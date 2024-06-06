import numpy as np
import cupy as cp

# BUGS
# fix some activations dtype, fix dtype int init; (maybe fix dtype in backwards of some layers (like bnorm2d))

class Tensor:
    def __init__(
        self, data, args=None, op=None, requires_grad=True, dtype=None, device="cpu"
    ):
        assert device in ["cpu", "cuda"], "Device must be 'cpu' or 'cuda'"
        if device == "cpu":
            self.xp = np
        else:
            self.xp = cp

        if isinstance(data, Tensor):
            self.data = self.xp.array(data.data, dtype=dtype)
        else:
            self.data = self.xp.array(data, dtype=dtype)

        self.grad = None
        self.op = op
        self.args = args
        self.requires_grad = requires_grad
        self.device = device

    def tensor(self, t, requires_grad=False):
        if isinstance(t, Tensor):
            assert t.device == self.device, "Tensors must be on the same device"

            return t

        return Tensor(t, requires_grad=requires_grad, device=self.device, dtype=self.data.dtype)

    def to(self, device):
        assert device in ["cpu", "cuda"], "Device must be 'cpu' or 'cuda'"
        if device == "cpu":
            xp = np
        else:
            xp = cp

        data = (
            xp.array(self.data)
            if isinstance(self.data, np.ndarray)
            else xp.array(self.data.get(), dtype=self.data.dtype)
        )

        return Tensor(
            data, requires_grad=self.requires_grad, dtype=self.data.dtype, device=device
        )

    def detach(self):
        return Tensor(
            data=self.data,
            args=self.args,
            op=self.op,
            requires_grad=False,
            dtype=self.data.dtype,
            device=self.device,
        )

    def item(self):
        return self.data.item()

    def add(self, t):
        t = self.tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = [self, t] if requires_grad else None

        return Tensor(
            self.data + t.data,
            args,
            "add",
            requires_grad=requires_grad,
            device=self.device,
        )

    def sub(self, t):
        t = self.tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = [self, t] if requires_grad else None

        return Tensor(
            self.data - t.data,
            args,
            "sub",
            requires_grad=requires_grad,
            device=self.device,
        )

    def mul(self, t):
        t = self.tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = [self, t] if requires_grad else None

        return Tensor(
            self.data * t.data,
            args,
            "mul",
            requires_grad=requires_grad,
            device=self.device,
        )

    def div(self, t):
        t = self.tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = [self, t] if requires_grad else None

        return Tensor(
            self.data / t.data,
            args,
            "div",
            requires_grad=requires_grad,
            device=self.device,
        )

    def matmul(self, n):
        n = self.tensor(n)

        requires_grad = self.requires_grad or n.requires_grad
        args = [self, n] if requires_grad else None

        return Tensor(
            self.xp.matmul(self.data, n.data),
            args,
            "matmul",
            requires_grad=requires_grad,
            device=self.device,
        )

    def sum(self, *args, **kwargs):
        axis = kwargs.get("axis", None) if len(args) == 0 else args[0]
        return Tensor(
            self.data.sum(*args, **kwargs),
            [self, axis],
            "sum",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def mean(self, *args, **kwargs):
        axis = kwargs.get("axis", None) if len(args) == 0 else args[0]
        return Tensor(
            self.data.mean(*args, **kwargs),
            [self, axis],
            "mean",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def var(self, *args, **kwargs):
        axis = kwargs.get("axis", None) if len(args) == 0 else args[0]
        return Tensor(
            self.data.var(*args, **kwargs),
            [self, axis],
            "var",
            requires_grad=self.requires_grad,
            device=self.device,
        )  # ddof = 0;

    def power(self, n):
        n = self.tensor(n)

        requires_grad = self.requires_grad or n.requires_grad
        args = [self, n] if requires_grad else None

        return Tensor(
            self.data**n.data,
            args,
            "power",
            requires_grad=requires_grad,
            device=self.device,
        )

    def sqrt(self):
        return Tensor(
            self.xp.sqrt(self.data),
            [self],
            "sqrt",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def log(self):
        return Tensor(
            self.xp.log(self.data),
            [self],
            "log",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def exp(self):
        return Tensor(
            self.xp.exp(self.data),
            [self],
            "exp",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def tanh(self):
        return Tensor(
            self.xp.tanh(self.data),
            [self],
            "tanh",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def sin(self):
        return Tensor(
            self.xp.sin(self.data),
            [self],
            "sin",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def cos(self):
        return Tensor(
            self.xp.cos(self.data),
            [self],
            "cos",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def maximum(self, t):
        t = self.tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = [self, t] if requires_grad else None

        return Tensor(
            self.xp.maximum(self.data, t.data),
            args,
            "maximum",
            requires_grad=requires_grad,
            device=self.device,
        )

    def minimum(self, t):
        t = self.tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = [self, t] if requires_grad else None

        return Tensor(
            self.xp.minimum(self.data, t.data),
            args,
            "minimum",
            requires_grad=requires_grad,
            device=self.device,
        )

    def max(self, axis=None, keepdims=False):  # equivalent to torch.amax
        return Tensor(
            self.data.max(axis=axis, keepdims=keepdims),
            [self, axis, keepdims],
            "max",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def min(self, axis=None, keepdims=False):  # equivalent to torch.amin
        return Tensor(
            self.data.min(axis=axis, keepdims=keepdims),
            [self, axis, keepdims],
            "min",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def concatenate(self, *tensors, axis=0):
        tensors = [self.tensor(t) for t in tensors]
        return Tensor(
            self.xp.concatenate([self.data] + [t.data for t in tensors], axis=axis),
            [self] + tensors + [axis],
            "concatenate",
            requires_grad=self.requires_grad or any([t.requires_grad for t in tensors]),
            device=self.device,
        )

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 else shape
        return Tensor(
            self.data.reshape(shape),
            [self],
            "reshape",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    # def split(self, n, axis = 0):
    #     # return Tensor(self.xp.split(self.data, n, axis = axis), [self], "split", requires_grad=self.requires_grad)
    #     return [Tensor(t, [self], "split", requires_grad=self.requires_grad) for t in self.xp.split(self.data, n, axis = axis)]

    def abs(self):
        return Tensor(
            self.xp.abs(self.data),
            [self],
            "abs",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def transpose(self, *axes):
        axes = axes[0] if len(axes) == 1 else axes
        if len(axes) == 0:
            axes = range(self.data.ndim)[::-1]
        return Tensor(
            self.data.transpose(axes),
            [self, axes],
            "transpose",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def swapaxes(self, axis1, axis2):
        return Tensor(
            self.xp.swapaxes(self.data, axis1, axis2),
            [self, axis1, axis2],
            "swapaxes",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def flip(self, axis):
        if axis is None:
            axis = range(self.data.ndim)
        return Tensor(
            self.xp.flip(self.data, axis),
            [self, axis],
            "flip",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def __neg__(self):
        return Tensor(
            -self.data,
            [self],
            "neg",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def __pos__(self):
        return Tensor(
            self.data,
            [self],
            "pos",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def __abs__(self):
        return self.abs()

    def __add__(self, t):
        return self.add(t)

    def __sub__(self, t):
        return self.sub(t)

    def __mul__(self, t):
        return self.mul(t)

    def __truediv__(self, t):
        return self.div(t)

    def __matmul__(self, t):
        return self.matmul(t)

    def __pow__(self, t):
        return self.power(t)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, dtype={self.data.dtype}, device={self.device})"

    # def __str__(self):
    #     return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __radd__(self, t):
        t = self.tensor(t)
        return t.add(self)

    def __rsub__(self, t):
        t = self.tensor(t)
        return t.sub(self)

    def __rmul__(self, t):
        t = self.tensor(t)
        return t.mul(self)

    def __rtruediv__(self, t):
        t = self.tensor(t)
        return t.div(self)

    def __rmatmul__(self, t):
        t = self.tensor(t)
        return t.matmul(self)

    def __rpow__(self, t):
        t = self.tensor(t)
        return t.power(self)

    # add unpacking of split tensors
    def __iter__(self):
        return iter(
            Tensor(
                self.data[i],
                [self, i],
                "getitem",
                requires_grad=self.requires_grad,
                device=self.device,
            )
            for i in range(self.data.shape[0])
        )

    def __getitem__(
        self, index
    ):  # problem when use grad array indexes: example y[0].grad; non-leaf tensor; in torch it retain_grad
        return Tensor(
            self.data[index],
            [self, index],
            "getitem",
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def __array__(self, dtype=None):
        return self.data.astype(dtype, copy=False)

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        return self.transpose()

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def backward(
        self, grad=None
    ):  # grad=self.xp.array(1) # TODO: ASSERT GRAD SHAPE == DATA SHAPE, assert grad.device == self.device
        if not self.requires_grad:
            return

        if grad is None:
            grad = self.xp.ones_like(self.data, dtype=self.dtype)
        else:
            grad = self.xp.array(grad, dtype=self.dtype)

        if (
            grad.size != self.data.size
            or grad.ndim != self.data.ndim
            or grad.shape != self.data.shape
        ):  # reverse broadcast; TODO : MAYBE MOVE IT TO ANOTHER PLACE
            if self.data.size == 1:
                grad = grad.sum()
            elif self.data.ndim == grad.ndim:
                grad = grad.sum(
                    axis=tuple(
                        self.xp.where(
                            self.xp.array(self.data.shape) != self.xp.array(grad.shape)
                        )[0].tolist()
                    ),
                    keepdims=True,
                )
            else:
                data_shape = (1,) * (grad.ndim - self.data.ndim) + self.data.shape
                axis = tuple(
                    self.xp.where(
                        self.xp.array(data_shape) != self.xp.array(grad.shape)
                    )[0].tolist()
                )
                grad = grad.sum(axis=axis)

            grad = grad.reshape(self.data.shape)

        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad  # += BUG FIX

        if self.op == "add":
            self.args[0].backward(grad)
            self.args[1].backward(grad)

        elif self.op == "sub":
            self.args[0].backward(grad)
            self.args[1].backward(-grad)

        elif self.op == "mul":
            self.args[0].backward(grad * self.args[1].data)
            self.args[1].backward(self.args[0].data * grad)

        elif self.op == "div":
            self.args[0].backward(grad / self.args[1].data)
            self.args[1].backward(-grad * self.args[0].data / self.args[1].data ** 2)

        elif self.op == "matmul":
            if (
                self.args[0].data.ndim > 1 and self.args[1].data.ndim > 1
            ):  # [matrix x matrix]
                self.args[0].backward(
                    self.xp.matmul(grad, self.args[1].data.swapaxes(-1, -2))
                )
                self.args[1].backward(
                    self.xp.matmul(self.args[0].data.swapaxes(-1, -2), grad)
                )

            elif (
                self.args[0].data.ndim == 1 and self.args[1].data.ndim == 1
            ):  # [vector x vector]
                self.args[0].backward(grad * self.args[1].data)
                self.args[1].backward(grad * self.args[0].data)

            elif (
                self.args[0].data.ndim == 1 and self.args[1].data.ndim > 1
            ):  # [vector x matrix]
                self.args[0].backward(
                    self.xp.matmul(grad, self.args[1].data.swapaxes(-1, -2))
                )
                self.args[1].backward(self.xp.outer(self.args[0].data, grad))

            elif (
                self.args[0].data.ndim > 1 and self.args[1].data.ndim == 1
            ):  # [matrix x vector]
                self.args[0].backward(self.xp.outer(grad, self.args[1].data))
                self.args[1].backward(
                    self.xp.matmul(self.args[0].data.swapaxes(-1, -2), grad)
                )

        elif self.op == "sum":
            axis = self.args[1]
            if grad.ndim != self.args[0].data.ndim and axis is not None:
                grad = self.xp.expand_dims(grad, axis)
            self.args[0].backward(self.xp.ones_like(self.args[0].data) * grad)

        elif self.op == "mean":
            axis = self.args[1]

            if grad.ndim != self.args[0].data.ndim and axis is not None:
                grad = self.xp.expand_dims(grad, axis)

            _axis = list(axis) if isinstance(axis, tuple) else axis
            self.args[0].backward(
                self.xp.ones_like(self.args[0].data)
                * grad
                / self.xp.prod(self.xp.array(self.args[0].data.shape)[_axis])
            )

        elif (
            self.op == "var"
        ):  # axis=None, ddof=0, keepdims=False add params instead args kwargs
            axis = self.args[1]

            if grad.ndim != self.args[0].data.ndim and axis is not None:
                grad = self.xp.expand_dims(grad, axis)

            _axis = list(axis) if isinstance(axis, tuple) else axis
            self.args[0].backward(
                self.xp.ones_like(self.args[0].data)
                * grad
                * 2
                * (self.args[0].data - self.args[0].data.mean(axis=axis, keepdims=True))
                / self.xp.prod(self.xp.array(self.args[0].data.shape)[_axis])
            )

        elif self.op == "power":
            self.args[0].backward(
                grad * self.args[1].data * self.args[0].data ** (self.args[1].data - 1)
            )
            self.args[1].backward(
                grad
                * self.args[0].data ** self.args[1].data
                * self.xp.log(self.args[0].data)
            )

        elif self.op == "sqrt":
            self.args[0].backward(grad * 1 / (2 * self.xp.sqrt(self.args[0].data)))

        elif self.op == "log":
            self.args[0].backward(grad * 1 / self.args[0].data)

        elif self.op == "exp":
            self.args[0].backward(grad * self.xp.exp(self.args[0].data))

        elif self.op == "tanh":
            self.args[0].backward(grad * (1 - self.xp.tanh(self.args[0].data) ** 2))

        elif self.op == "sin":
            self.args[0].backward(grad * self.xp.cos(self.args[0].data))

        elif self.op == "cos":
            self.args[0].backward(grad * -self.xp.sin(self.args[0].data))

        elif self.op == "maximum":
            self.args[0].backward(grad * (self.args[0].data >= self.args[1].data))
            self.args[1].backward(grad * (self.args[0].data <= self.args[1].data))

        elif self.op == "minimum":
            self.args[0].backward(grad * (self.args[0].data <= self.args[1].data))
            self.args[1].backward(grad * (self.args[0].data >= self.args[1].data))

        elif self.op == "max":
            axis, keepdims = self.args[1:]
            if grad.ndim != self.args[0].data.ndim and axis is not None:
                grad = self.xp.expand_dims(grad, axis)
            self.args[0].backward(
                grad
                * (self.args[0].data == self.args[0].data.max(axis=axis, keepdims=True))
            )

        elif self.op == "min":
            axis, keepdims = self.args[1:]
            if grad.ndim != self.args[0].data.ndim and axis is not None:
                grad = self.xp.expand_dims(grad, axis)
            self.args[0].backward(
                grad
                * (self.args[0].data == self.args[0].data.min(axis=axis, keepdims=True))
            )

        elif self.op == "concatenate":
            axis = self.args[-1]
            args = self.args[:-1]
            args_shapes = [arg.data.shape for arg in args]

            grads = self.xp.split(
                grad,
                self.xp.cumsum(
                    self.xp.array([arg_shape[axis] for arg_shape in args_shapes])
                )[:-1].tolist(),
                axis=axis,
            )

            for i, arg in enumerate(args):
                arg.backward(grads[i])

        elif self.op == "reshape":
            self.args[0].backward(grad.reshape(self.args[0].data.shape))

        # elif self.op == "split":
        #     self.args[0].backward(self.xp.concatenate(grad, axis = 0))

        elif self.op == "getitem":
            _grad = self.xp.zeros_like(self.args[0].data)
            _grad[self.args[1]] = grad

            self.args[0].backward(_grad)

        elif self.op == "transpose":
            self.args[0].backward(grad.transpose(self.args[1]))

        elif self.op == "swapaxes":
            self.args[0].backward(grad.swapaxes(self.args[1], self.args[2]))

        elif self.op == "flip":
            self.args[0].backward(self.xp.flip(grad, axis=self.args[1]))

        elif self.op == "neg":
            self.args[0].backward(-grad)

        elif self.op == "pos":
            self.args[0].backward(grad)

        elif self.op == "abs":
            self.args[0].backward(grad * self.xp.sign(self.args[0].data))


# BUGS:
# grad X - mean not correct with pytorch; maybe NOT BUG becase small numbers manipulation (Numerical stability issues)
# softmax not equals grads with pytorch; place: div; maybe NOT BUG becase small numbers manipulation (Numerical stability issues)????


# def repeat_to_match_shape(self, g, shape, dtype, axis, keepdims): same
# https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_vjps.py
#     """Returns the array g repeated along axis to fit vector space vs.
#     Also returns the number of repetitions of the array."""
#     if shape == ():
#         return g, 1
#     axis = list(axis) if isinstance(axis, tuple) else axis
#     new_shape = self.xp.array(shape)
#     new_shape[axis] = 1
#     num_reps = self.xp.prod(self.xp.array(shape)[axis])
#     # Can't use broadcast_to because of numpy bug: https://github.com/numpy/numpy/issues/9165
#     # return aself.xp.broadcast_to(aself.xp.reshape(g, new_shape), shape), num_reps
#     return self.xp.reshape(g, new_shape) + self.xp.zeros(shape, dtype=dtype), num_reps

# elif self.op == "mean":
# shape = self.args[0].data.shape
# axis = self.args[1]
# dtype = self.xp.result_type(self.args[0].data)
# g_repeated, num_reps = self.repeat_to_match_shape(grad, shape, dtype, axis, None)
# print(f"g_repeated {g_repeated}")
# self.args[0].backward(g_repeated / num_reps)
