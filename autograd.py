import numpy as np


class Tensor:
    def __init__(self, val, args=None, op=None, requires_grad=True):
        if isinstance(val, Tensor): # If val is a tensor, copy its attributes
            self.data = val.data
            self.grad = val.grad
            self.op = val.op
            self.args = val.args
            self.requires_grad = val.requires_grad
            return
            
        self.data = np.array(val)
        self.grad = None
        self.op = op
        self.args = args
        self.requires_grad = requires_grad

    def tensor(self, t):
        return t if isinstance(t, Tensor) else Tensor(t)

    def add(self, t):
        t = self.tensor(t)
        return Tensor(self.data + t.data, [self, t], "add", requires_grad=self.requires_grad or t.requires_grad)

    def sub(self, t):
        t = self.tensor(t)
        return Tensor(self.data - t.data, [self, t], "sub", requires_grad=self.requires_grad or t.requires_grad)

    def mul(self, t):
        t = self.tensor(t)
        return Tensor(self.data * t.data, [self, t], "mul", requires_grad=self.requires_grad or t.requires_grad)

    def div(self, t):
        t = self.tensor(t)
        return Tensor(self.data / t.data, [self, t], "div", requires_grad=self.requires_grad or t.requires_grad)

    def dot(self, v):
        v = self.tensor(v)
        return Tensor(np.dot(self.data, v.data), [self, v], "dot", requires_grad=self.requires_grad or v.requires_grad)

    def mv(self, v):
        v = self.tensor(v)
        return Tensor(np.dot(self.data, v.data), [self, v], "mv", requires_grad=self.requires_grad or v.requires_grad)

    def mm(self, n):
        n = self.tensor(n)
        return Tensor(np.dot(self.data, n.data), [self, n], "mm", requires_grad=self.requires_grad or n.requires_grad)

    def sum(self, axis = None):
        return Tensor(self.data.sum(axis = axis), [self], "sum", requires_grad=self.requires_grad)

    def mean(self, axis = None):
        return Tensor(self.data.mean(axis = axis), [self], "mean", requires_grad=self.requires_grad)
        
    def power(self, n):
        n = self.tensor(n)
        return Tensor(self.data ** n.data, [self, n], "power", requires_grad=self.requires_grad or n.requires_grad)    

    def log(self):
        return Tensor(np.log(self.data), [self], "log", requires_grad=self.requires_grad)

    def exp(self):
        return Tensor(np.exp(self.data), [self], "exp", requires_grad=self.requires_grad)

    def maximum(self, t):
        t = self.tensor(t)
        return Tensor(np.maximum(self.data, t.data), [self, t], "maximum", requires_grad=self.requires_grad or t.requires_grad)

    def minimum(self, t):
        t = self.tensor(t)
        return Tensor(np.minimum(self.data, t.data), [self, t], "minimum", requires_grad=self.requires_grad or t.requires_grad)

    def concatenate(self, *tensors, axis = 0):
        tensors = [self.tensor(t) for t in tensors]
        return Tensor(np.concatenate([self.data] + [t.data for t in tensors], axis = axis), [self] + tensors + [axis], "concatenate", requires_grad=self.requires_grad or any([t.requires_grad for t in tensors]))

    def reshape(self, *shape):
        return Tensor(self.data.reshape(shape), [self], "reshape", requires_grad=self.requires_grad)

    # def split(self, n, axis = 0):
    #     # return Tensor(np.split(self.data, n, axis = axis), [self], "split", requires_grad=self.requires_grad)
    #     return [Tensor(t, [self], "split", requires_grad=self.requires_grad) for t in np.split(self.data, n, axis = axis)]

    def abs(self):
        return Tensor(np.abs(self.data), [self], "abs", requires_grad=self.requires_grad)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = range(self.data.ndim)[::-1]
        return Tensor(self.data.transpose(axes), [self, axes], "transpose", requires_grad=self.requires_grad)

    def __neg__(self):
        return Tensor(-self.data, [self], "neg", requires_grad=self.requires_grad)

    def __pos__(self):
        return Tensor(self.data, [self], "pos", requires_grad=self.requires_grad)

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
        return self.dot(t)

    def __pow__(self, t):
        return self.power(t)

    def __repr__(self):
        return str(self.data)

    def __radd__(self, t):
        return self.add(t)

    def __rsub__(self, t):
        return self.sub(t)

    def __rmul__(self, t):
        return self.mul(t)

    def __rtruediv__(self, t):
        return self.div(t)

    def __rmatmul__(self, t):
        return self.dot(t)

    def __rpow__(self, t):
        return self.power(t)

    # add unpacking of split tensors
    def __iter__(self):
        return iter(Tensor(self.data[i], [self, i], "getitem", requires_grad=self.requires_grad) for i in range(self.data.shape[0]))

    def __getitem__(self, index): # problem when use grad array indexes: example y[0].grad; non-leaf tensor; in torch it retain_grad
        return Tensor(self.data[index], [self, index], "getitem", requires_grad=self.requires_grad)

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

    
    



    def backward(self, grad=1):
        if not self.requires_grad:
            return

        if self.grad is None:
            self.grad = grad # np.ones_like(self.data)
        else:
            self.grad += grad

        if self.op == "add":
            self.args[0].backward(grad)
            self.args[1].backward(grad)

        elif self.op == "sub":
            self.args[0].backward(grad)
            self.args[1].backward(-grad)

        elif self.op in ["dot", "mul"]:
            self.args[0].backward(grad * self.args[1].data)
            self.args[1].backward(self.args[0].data * grad)

        elif self.op == "div":
            self.args[0].backward(grad / self.args[1].data)
            self.args[1].backward(-grad * self.args[0].data / self.args[1].data ** 2)
            

        elif self.op == "mv":
            self.args[0].backward(np.outer(grad, self.args[1].data))
            self.args[1].backward(np.dot(grad, self.args[0].data))

        elif self.op == "mm":
            self.args[0].backward(np.dot(grad, self.args[1].data.T))
            self.args[1].backward(np.dot(self.args[0].data.T, grad))

        elif self.op == "sum":
            self.args[0].backward(np.ones_like(self.args[0].data) * grad)

        elif self.op == "mean":
            self.args[0].backward(np.ones_like(self.args[0].data) * grad / self.args[0].data.size)

        elif self.op == "power":
            self.args[0].backward(grad * self.args[1].data * self.args[0].data ** (self.args[1].data - 1))

        elif self.op == "log":
            self.args[0].backward(grad * 1 / self.args[0].data)

        elif self.op == "exp":
            self.args[0].backward(grad * np.exp(self.args[0].data))

        elif self.op == "maximum":
            self.args[0].backward(grad * (self.args[0].data >= self.args[1].data))
            self.args[1].backward(grad * (self.args[0].data <= self.args[1].data))

        elif self.op == "minimum":
            self.args[0].backward(grad * (self.args[0].data <= self.args[1].data))
            self.args[1].backward(grad * (self.args[0].data >= self.args[1].data))

        elif self.op == "concatenate":
            if type(grad) == int and grad == 1:
                grad = np.ones_like(self.data)

            axis = self.args[-1]
            args = self.args[:-1]
            args_shapes = [arg.data.shape for arg in args]

            grads = np.split(grad, np.cumsum([arg_shape[axis] for arg_shape in args_shapes])[:-1], axis=axis)
        
            for i, arg in enumerate(args):
                arg.backward(grads[i])
                
        elif self.op == "reshape":
            if type(grad) == int and grad == 1:
                grad = np.ones_like(self.data)
            self.args[0].backward(grad.reshape(self.args[0].data.shape))

        # elif self.op == "split":
        #     self.args[0].backward(np.concatenate(grad, axis = 0))

        elif self.op == "getitem":
            self.args[0].backward(np.zeros_like(self.args[0].data))
            self.args[0].grad[self.args[1]] = grad

        elif self.op == "transpose":
            if type(grad) == int and grad == 1:
                grad = np.ones_like(self.data)
            self.args[0].backward(grad.transpose(self.args[1]))

        elif self.op == "neg":
            self.args[0].backward(-grad)

        elif self.op == "pos":
            self.args[0].backward(grad)

        elif self.op == "abs":
            self.args[0].backward(grad * np.sign(self.args[0].data))







