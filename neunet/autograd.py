from typing import Any, Union

import cupy as cp
import numpy as np



class Tensor:
    def __init__(self, data: Any, args=None, op=None, requires_grad: bool=True, dtype = None, device: str="cpu"):
        if device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'")
        if device == "cpu":
            self.xp = np
        else:
            self.xp = cp

        if isinstance(data, Tensor):
            self.data: Union[np.ndarray, cp.ndarray] = self.xp.array(data.data, dtype=dtype if dtype else np.float32)
        else:
            self.data = self.xp.array(data, dtype=dtype if dtype else np.float32)

        self.grad = None
        self.op = op
        self.args = args
        self.requires_grad = requires_grad
        self.device = device
        self.grad_fn = lambda: None

    def ensure_tensor(self, t: Union[Any, 'Tensor'], requires_grad=False) -> 'Tensor':
        if isinstance(t, Tensor):
            if t.device != self.device:
                raise ValueError("Tensors must be on the same device")

            return t

        return Tensor(t, requires_grad=requires_grad, device=self.device, dtype=self.data.dtype)

    def numpy(self) -> np.ndarray:
        if self.device != "cpu":
            raise ValueError("Tensor must be on the CPU")
        if self.requires_grad:
            raise ValueError("Tensor must not require gradient")

        return self.data

    def to(self, device: str) -> 'Tensor':
        if device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'")
        if device == "cpu":
            xp = np
        else:
            xp = cp

        data = (
            xp.array(self.data)
            if isinstance(self.data, np.ndarray)
            else xp.array(self.data.get(), dtype=self.data.dtype)
        )

        return Tensor(data, requires_grad=self.requires_grad, dtype=self.data.dtype, device=device)

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def detach(self) -> 'Tensor':
        return Tensor(
            data=self.data,
            args=self.args,
            op=self.op,
            requires_grad=False,
            dtype=self.data.dtype,
            device=self.device,
        )

    def item(self) -> float:
        return self.data.item()

    def contiguous(self) -> 'Tensor':
        self.data[...] = self.xp.ascontiguousarray(self.data)
        return self

    def _apply_grad(self, grad):
        if not self.requires_grad:
            return

        grad = self._reverse_broadcast(grad)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad  # += BUG FIX


    def add(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = (self, t) if requires_grad else None
        
        out = Tensor(
            self.data + t.data,
            args,
            "add",
            requires_grad=requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', t: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad)
            if t.requires_grad:
                t._apply_grad(grad)

        out.grad_fn = grad_fn

        return out

    def sub(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = (self, t) if requires_grad else None

        out = Tensor(
            self.data - t.data,
            args,
            "sub",
            requires_grad=requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', t: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad)
            if t.requires_grad:
                t._apply_grad(-grad)

        out.grad_fn = grad_fn

        return out

    def mul(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = (self, t) if requires_grad else None

        out = Tensor(
            self.data * t.data,
            args,
            "mul",
            requires_grad=requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', t: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad * t.data)
            if t.requires_grad:
                t._apply_grad(self.data * grad)

        out.grad_fn = grad_fn

        return out

    def div(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = (self, t) if requires_grad else None

        out = Tensor(
            self.data / t.data,
            args,
            "div",
            requires_grad=requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', t: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad / t.data)
            if t.requires_grad:
                t._apply_grad(-grad  * self.data / (t.data**2))

        out.grad_fn = grad_fn

        return out

    def matmul(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = (self, t) if requires_grad else None

        out = Tensor(
            self.xp.matmul(self.data, t.data),
            args,
            "matmul",
            requires_grad=requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', t: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.data.ndim > 1 and t.data.ndim > 1:  # [matrix x matrix]
                if self.requires_grad:
                    self._apply_grad(self.xp.matmul(grad, t.data.swapaxes(-1, -2)))
                if t.requires_grad:
                    t._apply_grad(self.xp.matmul(self.data.swapaxes(-1, -2), grad))
            elif self.data.ndim == 1 and t.data.ndim == 1:  # [vector x vector]
                if self.requires_grad:
                    self._apply_grad(grad * t.data)
                if t.requires_grad:
                    t._apply_grad(grad * self.data)
            elif self.data.ndim == 1 and t.data.ndim > 1:  # [vector x matrix]
                if self.requires_grad:
                    self._apply_grad(self.xp.matmul(grad, t.data.swapaxes(-1, -2)))
                if t.requires_grad:
                    t._apply_grad(self.xp.outer(self.data, grad))
            elif self.data.ndim > 1 and t.data.ndim == 1:  # [matrix x vector]
                if self.requires_grad:
                    self._apply_grad(self.xp.outer(grad, t.data))
                if t.requires_grad:
                    t._apply_grad(self.xp.matmul(self.data.swapaxes(-1, -2), grad))

        out.grad_fn = grad_fn

        return out
                

    def sum(self, *args: Any, **kwargs: Any) -> 'Tensor':
        axis = kwargs.get("axis", None) if len(args) == 0 else args[0]
        out = Tensor(
            self.data.sum(*args, **kwargs),
            [self, axis],
            "sum",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', axis, grad: Union[np.ndarray, cp.ndarray]):
            if not self.requires_grad:
                return

            if grad.ndim != self.data.ndim and axis is not None:
                grad = self.xp.expand_dims(grad, axis)
            self._apply_grad(self.xp.ones_like(self.data) * grad)

        out.grad_fn = grad_fn

        return out

    def mean(self, *args: Any, **kwargs: Any) -> 'Tensor':
        axis = kwargs.get("axis", None) if len(args) == 0 else args[0]
        out = Tensor(
            self.data.mean(*args, **kwargs),
            [self, axis],
            "mean",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', axis, grad: Union[np.ndarray, cp.ndarray]):
            if not self.requires_grad:
                return

            if grad.ndim != self.data.ndim and axis is not None:
                grad = self.xp.expand_dims(grad, axis)

            _axis = list(axis) if isinstance(axis, tuple) else axis
            self._apply_grad(
                self.xp.ones_like(self.data)
                * grad
                / self.xp.prod(self.xp.array(self.data.shape)[_axis])
            )

        out.grad_fn = grad_fn

        return out

    def var(self, *args: Any, **kwargs: Any) -> 'Tensor':
        axis = kwargs.get("axis", None) if len(args) == 0 else args[0]
        out = Tensor(
            self.data.var(*args, **kwargs),
            [self, axis],
            "var",
            requires_grad=self.requires_grad,
            device=self.device,
        )  # ddof = 0;

        def grad_fn(self: 'Tensor', axis, grad: Union[np.ndarray, cp.ndarray]):
            if not self.requires_grad:
                return

            if grad.ndim != self.data.ndim and axis is not None:
                grad = self.xp.expand_dims(grad, axis)

            _axis = list(axis) if isinstance(axis, tuple) else axis
            self._apply_grad(
                self.xp.ones_like(self.data)
                * grad
                * 2
                * (self.data - self.data.mean(axis=axis, keepdims=True))
                / self.xp.prod(self.xp.array(self.data.shape)[_axis])
            )

        out.grad_fn = grad_fn

        return out


    def power(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = (self, t) if requires_grad else None

        out = Tensor(
            self.data**t.data,
            args,
            "power",
            requires_grad=requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', t: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad * t.data * self.data ** (t.data - 1))
            if t.requires_grad:
                t._apply_grad(grad * self.data ** t.data * self.xp.log(self.data))

        out.grad_fn = grad_fn

        return out
            

    def sqrt(self) -> 'Tensor':
        out = Tensor(
            self.xp.sqrt(self.data),
            [self],
            "sqrt",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad * 1 / (2 * self.xp.sqrt(self.data)))

        out.grad_fn = grad_fn

        return out

    def log(self) -> 'Tensor':
        out = Tensor(
            self.xp.log(self.data),
            [self],
            "log",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad * 1 / self.data)

        out.grad_fn = grad_fn

        return out

    def exp(self) -> 'Tensor':
        out = Tensor(
            self.xp.exp(self.data),
            [self],
            "exp",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad * self.xp.exp(self.data))

        out.grad_fn = grad_fn

        return out

    def tanh(self) -> 'Tensor':
        out = Tensor(
            self.xp.tanh(self.data),
            [self],
            "tanh",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad * (1 - self.xp.tanh(self.data) ** 2))

        out.grad_fn = grad_fn

        return out

    def sin(self):
        out = Tensor(
            self.xp.sin(self.data),
            [self],
            "sin",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad * self.xp.cos(self.data))

        out.grad_fn = grad_fn

        return out

    def cos(self) -> 'Tensor':
        out = Tensor(
            self.xp.cos(self.data),
            [self],
            "cos",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad * -self.xp.sin(self.data))

        out.grad_fn = grad_fn

        return out

    def maximum(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = (self, t) if requires_grad else None

        out = Tensor(
            self.xp.maximum(self.data, t.data),
            args,
            "maximum",
            requires_grad=requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', t: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad * (self.data >= t.data))
            if t.requires_grad:
                t._apply_grad(grad * (self.data <= t.data))

        out.grad_fn = grad_fn

        return out

    def minimum(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = (self, t) if requires_grad else None

        out = Tensor(
            self.xp.minimum(self.data, t.data),
            args,
            "minimum",
            requires_grad=requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', t: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad * (self.data <= t.data))
            if t.requires_grad:
                t._apply_grad(grad * (self.data >= t.data))

        out.grad_fn = grad_fn

        return out

    def max(self, axis=None, keepdims=False) -> 'Tensor':  # equivalent to torch.amax
        out = Tensor(
            self.data.max(axis=axis, keepdims=keepdims),
            [self, axis],
            "max",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', axis, grad: Union[np.ndarray, cp.ndarray]):
            if not self.requires_grad:
                return

            if grad.ndim != self.data.ndim and axis is not None:
                grad = self.xp.expand_dims(grad, axis)
            self._apply_grad(grad * (self.data == self.data.max(axis=axis, keepdims=True)))

        out.grad_fn = grad_fn

        return out
            

    def min(self, axis=None, keepdims=False) -> 'Tensor':  # equivalent to torch.amin
        out = Tensor(
            self.data.min(axis=axis, keepdims=keepdims),
            [self, axis],
            "min",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', axis, grad: Union[np.ndarray, cp.ndarray]):
            if not self.requires_grad:
                return

            if grad.ndim != self.data.ndim and axis is not None:
                grad = self.xp.expand_dims(grad, axis)
            self._apply_grad(grad * (self.data == self.data.min(axis=axis, keepdims=True)))

        out.grad_fn = grad_fn

        return out

    def concatenate(self, *tensors: 'Tensor', axis=0) -> 'Tensor':
        tensors = tuple(self.ensure_tensor(t) for t in tensors)
        args = (self, *tensors, axis)
        out = Tensor(
            self.xp.concatenate([self.data] + [t.data for t in tensors], axis=axis),
            args,
            "concatenate",
            requires_grad=self.requires_grad or any(t.requires_grad for t in tensors),
            device=self.device,
        )

        def grad_fn(*args, grad: Union[np.ndarray, cp.ndarray]):
            args_shapes: list[Tensor] = [arg.data.shape for arg in args[:-1]]
            axis = args[-1]

            grads = self.xp.split(
                grad,
                self.xp.cumsum(self.xp.array([arg_shape[axis] for arg_shape in args_shapes]))[
                    :-1
                ].tolist(),
                axis=axis,
            )

            for i, arg in enumerate(args[:-1]):
                if arg.requires_grad:
                    arg._apply_grad(grads[i])

        out.grad_fn = grad_fn

        return out

    def reshape(self, *shape: Any) -> 'Tensor':
        shape = shape[0] if len(shape) == 1 else shape
        out = Tensor(
            self.data.reshape(shape),
            [self],
            "reshape",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad.reshape(self.data.shape))

        out.grad_fn = grad_fn

        return out


    # def split(self, n, axis = 0):
    #     # return Tensor(self.xp.split(self.data, n, axis = axis), [self], "split", requires_grad=self.requires_grad)
    #     return [Tensor(t, [self], "split", requires_grad=self.requires_grad) for t in self.xp.split(self.data, n, axis = axis)]

    def abs(self) -> 'Tensor':
        out = Tensor(
            self.xp.abs(self.data),
            [self],
            "abs",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad * self.xp.sign(self.data))

        out.grad_fn = grad_fn

        return out

    def transpose(self, *axes: Any) -> 'Tensor': 
        axes = axes[0] if len(axes) == 1 else axes
        if len(axes) == 0:
            axes = tuple(range(self.data.ndim)[::-1])
        out = Tensor(
            self.data.transpose(axes),
            [self, axes],
            "transpose",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', axes: Any, grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad.transpose(axes))

        out.grad_fn = grad_fn

        return out

    def swapaxes(self, axis1: int, axis2: int) -> 'Tensor':
        out = Tensor(
            self.xp.swapaxes(self.data, axis1, axis2),
            [self, axis1, axis2],
            "swapaxes",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', axis1: int, axis2: int, grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad.swapaxes(axis1, axis2))

        out.grad_fn = grad_fn

        return out

    def flip(self, axis: Any) -> 'Tensor':
        if axis is None:
            axis = range(self.data.ndim)
        out = Tensor(
            self.xp.flip(self.data, axis),
            [self, axis],
            "flip",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', axis: Any, grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(self.xp.flip(grad, axis=axis))

        out.grad_fn = grad_fn

        return out

    def where(self, condition: 'Tensor', t: Union[Any, 'Tensor']) -> 'Tensor':
        condition = self.ensure_tensor(condition)
        t = self.ensure_tensor(t)

        requires_grad = self.requires_grad or t.requires_grad
        args = [self, condition, t] if requires_grad else None

        out = Tensor(
            np.where(condition.data, self.data, t.data),
            args,
            "where",
            requires_grad=requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', condition: 'Tensor', t: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad * self.xp.where(condition.data, grad, self.xp.zeros_like(grad)))
            if t.requires_grad:
                t._apply_grad(grad * self.xp.where(condition.data, self.xp.zeros_like(grad), grad))

        out.grad_fn = grad_fn

        return out

    def equal(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        return Tensor(
            self.xp.equal(self.data, t.data),
            None,
            "equal",
            requires_grad=False,
            device=self.device,
        )

    def not_equal(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        return Tensor(
            self.xp.not_equal(self.data, t.data),
            None,
            "not_equal",
            requires_grad=False,
            device=self.device,
        )

    def greater(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        return Tensor(
            self.xp.greater(self.data, t.data),
            None,
            "greater",
            requires_grad=False,
            device=self.device,
        )

    def greater_equal(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        return Tensor(
            self.xp.greater_equal(self.data, t.data),
            None,
            "greater_equal",
            requires_grad=False,
            device=self.device,
        )

    def less(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        return Tensor(
            self.xp.less(self.data, t.data),
            None,
            "less",
            requires_grad=False,
            device=self.device,
        )

    def less_equal(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        return Tensor(
            self.xp.less_equal(self.data, t.data),
            None,
            "less_equal",
            requires_grad=False,
            device=self.device,
        )

    def logical_and(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        return Tensor(
            self.xp.logical_and(self.data, t.data),
            None,
            "logical_and",
            requires_grad=False,
            device=self.device,
        )

    def logical_or(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)

        return Tensor(
            self.xp.logical_or(self.data, t.data),
            None,
            "logical_or",
            requires_grad=False,
            device=self.device,
        )

    def logical_not(self) -> 'Tensor':
        return Tensor(
            self.xp.logical_not(self.data),
            None,
            "logical_not",
            requires_grad=False,
            device=self.device,
        )

    def __eq__(self, t: Union[Any, 'Tensor']) -> 'Tensor': # type: ignore[override]
        return self.equal(t)

    def __ne__(self, t: Union[Any, 'Tensor']) -> 'Tensor': # type: ignore[override]
        return self.not_equal(t)

    def __gt__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        return self.greater(t)

    def __ge__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        return self.greater_equal(t)

    def __lt__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        return self.less(t)

    def __le__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        return self.less_equal(t)

    def __and__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        return self.logical_and(t)

    def __or__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        return self.logical_or(t)

    def __invert__(self) -> 'Tensor':
        return self.logical_not()

    def __neg__(self) -> 'Tensor':
        out = Tensor(
            -self.data,
            [self],
            "neg",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(-grad)

        out.grad_fn = grad_fn

        return out

    def __pos__(self) -> 'Tensor':
        out = Tensor(
            self.data,
            [self],
            "pos",
            requires_grad=self.requires_grad,
            device=self.device,
        )

        def grad_fn(self: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                self._apply_grad(grad)

        out.grad_fn = grad_fn

        return out

    def __abs__(self) -> 'Tensor':
        return self.abs()

    def __add__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        return self.add(t)

    def __sub__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        return self.sub(t)

    def __mul__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        return self.mul(t)

    def __truediv__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        return self.div(t)

    def __matmul__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        return self.matmul(t)

    def __pow__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        return self.power(t)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, dtype={self.data.dtype}, device={self.device})"

    # def __str__(self):
    #     return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __radd__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)
        return t.add(self)

    def __rsub__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)
        return t.sub(self)

    def __rmul__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)
        return t.mul(self)

    def __rtruediv__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)
        return t.div(self)

    def __rmatmul__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)
        return t.matmul(self)

    def __rpow__(self, t: Union[Any, 'Tensor']) -> 'Tensor':
        t = self.ensure_tensor(t)
        return t.power(self)

    # add unpacking of split tensors
    # def __iter__(self) -> Iterator['Tensor']:
    #     out = iter(
    #         Tensor(
    #             self.data[i],
    #             [self,],
    #             "getitem",
    #             requires_grad=self.requires_grad,
    #             device=self.device,
    #         )
    #         for i in range(self.data.shape[0])
    #     )

    #     def grad_fn():
    #         if self.requires_grad:
    #             grad = self.xp.zeros_like(self.data)
    #             grad[i] = out.grad

    #             self._apply_grad(grad)

    #     for i, o in enumerate(out):
    #         o.grad_fn = grad_fn

    #     return out
    def __getitem__(
        self, index: Any
    ) -> 'Tensor':  # problem when use grad array indexes: example y[0].grad; non-leaf tensor; in torch it retain_grad
        out = Tensor(
            self.data[index],
            [self, index],
            "getitem",
            requires_grad=self.requires_grad,
            device=self.device,
            dtype=self.dtype
        )

        def grad_fn(self: 'Tensor', index, grad: Union[np.ndarray, cp.ndarray]):
            if self.requires_grad:
                _grad = self.xp.zeros_like(self.data)
                _grad[index] = grad

                self._apply_grad(_grad)

        out.grad_fn = grad_fn

        return out


    def __setitem__(self, key, value: Union[Any, 'Tensor']):
        if self.requires_grad:
            raise RuntimeError("Cannot assign values to a tensor with requires_grad=True")
        value = self.ensure_tensor(value)
        self.data[key] = value.data

    def __array__(self, dtype: Any=None) -> np.ndarray:
        return self.data.astype(dtype, copy=False)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def T(self) -> 'Tensor':
        return self.transpose()

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def _reverse_broadcast(self, grad):
        if (
            grad.size != self.data.size
            or grad.ndim != self.data.ndim
            or grad.shape != self.data.shape
        ):
            if self.data.size == 1:
                grad = grad.sum()
            elif self.data.ndim == grad.ndim:
                grad = grad.sum(
                    axis=tuple(
                        self.xp.where(self.xp.array(self.data.shape) != self.xp.array(grad.shape))[
                            0
                        ].tolist()
                    ),
                    keepdims=True,
                )
            else:
                data_shape = (1,) * (grad.ndim - self.data.ndim) + self.data.shape
                axis = tuple(
                    self.xp.where(self.xp.array(data_shape) != self.xp.array(grad.shape))[
                        0
                    ].tolist()
                )
                grad = grad.sum(axis=axis)

            grad = grad.reshape(self.data.shape)

        return grad


    def backward(
        self, grad=None
    ):  # grad=self.xp.array(1) # TODO: ASSERT GRAD SHAPE == DATA SHAPE, assert grad.device == self.device
        if not self.requires_grad:
            return

        # TODO: think about this logic
        if grad is None:
            grad = self.xp.ones_like(self.data, dtype=self.dtype)
        else:
            grad = self.xp.array(grad, dtype=self.dtype)

        self._apply_grad(grad)
        # Perform a topological sort to ensure gradients are calculated in the correct order
        def toposort(v):
            tape = []
            visited_ids = set()
            stack = [v]

            while stack:
                node: Tensor = stack.pop()
                node_id = id(node)
                
                if node_id in visited_ids:
                    continue
                
                visited_ids.add(node_id)
                
                if node.args is not None:
                    for child in node.args:
                        if not isinstance(child, Tensor):
                            continue
                        if child.requires_grad is False:
                            continue
                        stack.append(child)
                
                    tape.append(node)

            return tape
        
        # Apply the backward function in reverse order
        for v in toposort(self):
            v.grad_fn(*v.args, grad=v.grad)

# BUGS:
# grad X - mean not correct with pytorch; maybe NOT BUG becase small numbers manipulation (Numerical stability issues)
# softmax not equals grads with pytorch; place: div; maybe NOT BUG becase small numbers manipulation (Numerical stability issues)????