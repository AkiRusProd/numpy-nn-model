import pickle
from pathlib import Path

import numpy as np

from neunet import nn as nn
from neunet.autograd import Tensor

int16 = np.int16
int32 = np.int32
int64 = np.int64
float16 = np.float16
float32 = np.float32
float64 = np.float64


def save(obj: object, f, pickle_protocol: int = 2):
    path = Path(f)
    with path.open("wb") as file:
        pickle.dump(obj, file, protocol=pickle_protocol)

def load(f):
    
    path = Path(f)
    with path.open("rb") as file:
        return pickle.load(file)

# references to the original Tensor functions


def tensor(data, requires_grad=False, dtype=float32, device="cpu"):
    return Tensor(data, requires_grad=requires_grad, dtype=dtype, device=device)


def ones(*shape, dtype=None, requires_grad=False, device="cpu"):
    shape = tuple(*shape) if all(isinstance(arg, (list, tuple)) for arg in shape) else shape

    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad, device=device)


def zeros(*shape, dtype=None, requires_grad=False, device="cpu"):
    shape = tuple(*shape) if all(isinstance(arg, (list, tuple)) for arg in shape) else shape

    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad, device=device)


def rand(*shape, dtype=None, requires_grad=False, device="cpu"):
    shape = tuple(*shape) if all(isinstance(arg, (list, tuple)) for arg in shape) else shape

    return Tensor(np.random.rand(*shape).astype(dtype), requires_grad=requires_grad, device=device)


def randn(*shape, dtype=None, requires_grad=False, device="cpu"):
    shape = tuple(*shape) if all(isinstance(arg, (list, tuple)) for arg in shape) else shape

    return Tensor(
        np.random.randn(*shape).astype(dtype),
        requires_grad=requires_grad,
        device=device,
    )


def arange(start=0, end=None, step=1, dtype=None, requires_grad=False, device="cpu"):
    if end is None:
        start, end = 0, start
    return Tensor(
        np.arange(start, end, step, dtype=dtype),
        requires_grad=requires_grad,
        device=device,
    )


def ones_like(tensor, dtype=None, requires_grad=False, device="cpu"):
    return Tensor(np.ones_like(tensor.data, dtype), requires_grad=requires_grad, device=device)


def zeros_like(tensor, dtype=None, requires_grad=False, device="cpu"):
    return Tensor(np.zeros_like(tensor.data, dtype), requires_grad=requires_grad, device=device)


def argmax(x, axis=None, keepdims=False, device="cpu"):
    return Tensor(
        np.argmax(x.data, axis=axis, keepdims=keepdims),
        requires_grad=False,
        device=device,
        dtype=int32
    )


def argmin(x, axis=None, keepdims=False, device="cpu"):
    return Tensor(
        np.argmin(x.data, axis=axis, keepdims=keepdims),
        requires_grad=False,
        device=device,
        dtype = int32
    )


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def mul(x, y):
    return x * y


def div(x, y):
    return x / y


def matmul(x, y):
    return x.matmul(y)


def sum(x, axis=None, keepdims=False):
    return x.sum(axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    return x.mean(axis=axis, keepdims=keepdims)


def var(x, axis=None, keepdims=False):
    return x.var(axis=axis, keepdims=keepdims)


def power(x, y):
    return x**y


def sqrt(x):
    return x.sqrt()


def log(x):
    return x.log()


def exp(x):
    return x.exp()


def tanh(x):
    return x.tanh()


def sin(x):
    return x.sin()


def cos(x):
    return x.cos()


def maximum(x, y):
    return x.maximum(y)


def minimum(x, y):
    return x.minimum(y)


def max(x, axis=None, keepdims=False):
    return x.max(axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    return x.min(axis=axis, keepdims=keepdims)


def concatenate(*tensors, axis=0):
    tensors = tensors[0] if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)) else tensors
    return Tensor.concatenate(*tensors, axis=axis)


def reshape(x, *shape):
    return x.reshape(*shape)


def abs(x):
    return x.abs()


def transpose(x, *axes):
    return x.transpose(*axes)


def swapaxes(x, axis1, axis2):
    return x.swapaxes(axis1, axis2)


def flip(x, axis):
    return x.flip(axis=axis)

def where(condition, x, y):
    x = tensor(x, device=condition.device) if not isinstance(x, Tensor) else x
    return x.where(condition, y)

def equal(x, y):
    return x.equal(y)

def not_equal(x, y):
    return x.not_equal(y)

def greater(x, y):
    return x.greater(y)

def greater_equal(x, y):
    return x.greater_equal(y)

def less(x, y):
    return x.less(y)

def less_equal(x, y):
    return x.less_equal(y)

def logical_and(x, y):
    return x.logical_and(y)

def logical_or(x, y):
    return x.logical_or(y)

def logical_not(x):
    return x.logical_not()