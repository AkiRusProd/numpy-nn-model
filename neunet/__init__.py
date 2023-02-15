from neunet.autograd import Tensor
from neunet import nn



# references to the original Tensor functions

def tensor(data, requires_grad=True):
    return Tensor(data, requires_grad=requires_grad)

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
    return x ** y

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

