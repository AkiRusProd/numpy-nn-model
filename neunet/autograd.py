import numpy as np


class Tensor:
    def __init__(self, val, args=None, op=None, requires_grad=True, dtype = None):
        if isinstance(val, Tensor): # If val is a tensor, copy its attributes
            self.data = val.data
            self.grad = val.grad
            self.op = val.op
            self.args = val.args
            self.requires_grad = val.requires_grad
            return
            
        self.data = np.array(val, dtype=dtype)
        self.grad = None
        self.op = op
        self.args = args
        self.requires_grad = requires_grad

    def tensor(self, t, requires_grad=False):
        return t if isinstance(t, Tensor) else Tensor(t, requires_grad)

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

    def matmul(self, n):
        n = self.tensor(n)
        return Tensor(np.matmul(self.data, n.data), [self, n], "matmul", requires_grad=self.requires_grad or n.requires_grad)

    def sum(self, *args, **kwargs):
        axis = kwargs.get("axis", None) if len(args) == 0 else args[0]
        return Tensor(self.data.sum(*args, **kwargs), [self, axis], "sum", requires_grad=self.requires_grad)

    def mean(self, *args, **kwargs):
        axis = kwargs.get("axis", None) if len(args) == 0 else args[0]
        return Tensor(self.data.mean(*args, **kwargs), [self, axis], "mean", requires_grad=self.requires_grad)

    def var(self, *args, **kwargs):
        axis = kwargs.get("axis", None) if len(args) == 0 else args[0]
        return Tensor(self.data.var(*args, **kwargs), [self, axis], "var", requires_grad=self.requires_grad) #ddof = 0;
        
    def power(self, n):
        n = self.tensor(n)
        return Tensor(self.data ** n.data, [self, n], "power", requires_grad=self.requires_grad or n.requires_grad)    

    def sqrt(self):
        return Tensor(np.sqrt(self.data), [self], "sqrt", requires_grad=self.requires_grad)

    def log(self):
        return Tensor(np.log(self.data), [self], "log", requires_grad=self.requires_grad)

    def exp(self):
        return Tensor(np.exp(self.data), [self], "exp", requires_grad=self.requires_grad)

    def tanh(self):
        return Tensor(np.tanh(self.data), [self], "tanh", requires_grad=self.requires_grad)

    def sin(self):
        return Tensor(np.sin(self.data), [self], "sin", requires_grad=self.requires_grad)

    def cos(self):
        return Tensor(np.cos(self.data), [self], "cos", requires_grad=self.requires_grad)

    def maximum(self, t):
        t = self.tensor(t)
        return Tensor(np.maximum(self.data, t.data), [self, t], "maximum", requires_grad=self.requires_grad or t.requires_grad)

    def minimum(self, t):
        t = self.tensor(t)
        return Tensor(np.minimum(self.data, t.data), [self, t], "minimum", requires_grad=self.requires_grad or t.requires_grad)

    def max(self, axis = None, keepdims = False): #equivalent to torch.amax
        return Tensor(self.data.max(axis = axis, keepdims = keepdims), [self, axis, keepdims], "max", requires_grad=self.requires_grad)

    def min(self, axis = None, keepdims = False): #equivalent to torch.amin
        return Tensor(self.data.min(axis = axis, keepdims = keepdims), [self, axis, keepdims], "min", requires_grad=self.requires_grad)

    def concatenate(self, *tensors, axis = 0):
        tensors = [self.tensor(t) for t in tensors]
        return Tensor(np.concatenate([self.data] + [t.data for t in tensors], axis = axis), [self] + tensors + [axis], "concatenate", requires_grad=self.requires_grad or any([t.requires_grad for t in tensors]))

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 else shape
        return Tensor(self.data.reshape(shape), [self], "reshape", requires_grad=self.requires_grad)

    # def split(self, n, axis = 0):
    #     # return Tensor(np.split(self.data, n, axis = axis), [self], "split", requires_grad=self.requires_grad)
    #     return [Tensor(t, [self], "split", requires_grad=self.requires_grad) for t in np.split(self.data, n, axis = axis)]

    def abs(self):
        return Tensor(np.abs(self.data), [self], "abs", requires_grad=self.requires_grad)

    def transpose(self, *axes):
        axes = axes[0] if len(axes) == 1 else axes
        if len(axes) == 0:
            axes = range(self.data.ndim)[::-1]
        return Tensor(self.data.transpose(axes), [self, axes], "transpose", requires_grad=self.requires_grad)
    
    def swapaxes(self, axis1, axis2):
        return Tensor(np.swapaxes(self.data, axis1, axis2), [self, axis1, axis2], "swapaxes", requires_grad=self.requires_grad)

    def flip(self, axis):
        if axis is None:
            axis = range(self.data.ndim)
        return Tensor(np.flip(self.data, axis), [self, axis], "flip", requires_grad=self.requires_grad)

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
        return self.matmul(t)

    def __pow__(self, t):
        return self.power(t)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, dtype={self.data.dtype})"

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
        return iter(Tensor(self.data[i], [self, i], "getitem", requires_grad=self.requires_grad) for i in range(self.data.shape[0]))

    def __getitem__(self, index): # problem when use grad array indexes: example y[0].grad; non-leaf tensor; in torch it retain_grad
        return Tensor(self.data[index], [self, index], "getitem", requires_grad=self.requires_grad)
    
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

    
    def backward(self, grad = None):#grad=np.array(1) # TODO: ASSERT GRAD SHAPE == DATA SHAPE
        if not self.requires_grad:
            return
        
        if grad is None:
            grad = np.ones_like(self.data, dtype = self.dtype)
        else:
            grad = np.array(grad, dtype = self.dtype)

        if grad.size != self.data.size or grad.ndim != self.data.ndim or grad.shape != self.data.shape: # reverse broadcast; TODO : MAYBE MOVE IT TO ANOTHER PLACE
            if self.data.size == 1:
                grad = grad.sum()
            elif self.data.ndim == grad.ndim:
                grad = grad.sum(axis=tuple(np.where(np.array(self.data.shape) != np.array(grad.shape))[0]), keepdims=True)
            else:
                data_shape = (1,) * (grad.ndim - self.data.ndim) + self.data.shape
                axis = tuple(np.where(np.array(data_shape) != np.array(grad.shape))[0])
                grad = grad.sum(axis=axis).reshape(self.data.shape)
           
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad # += BUG FIX

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

            if self.args[0].data.ndim > 1 and self.args[1].data.ndim > 1: # [matrix x matrix]
                self.args[0].backward(np.matmul(grad, self.args[1].data.swapaxes(-1, -2)))
                self.args[1].backward(np.matmul(self.args[0].data.swapaxes(-1, -2), grad))

            elif self.args[0].data.ndim == 1 and self.args[1].data.ndim == 1: # [vector x vector]
                self.args[0].backward(grad * self.args[1].data)
                self.args[1].backward(grad * self.args[0].data)

            elif self.args[0].data.ndim == 1 and self.args[1].data.ndim > 1: # [vector x matrix]
                self.args[0].backward(grad * self.args[1].data)
                self.args[1].backward(np.outer(grad, self.args[0].data))

            elif self.args[0].data.ndim > 1 and self.args[1].data.ndim == 1: # [matrix x vector]
                self.args[0].backward(np.outer(grad, self.args[1].data))
                self.args[1].backward(grad * self.args[0].data)


        elif self.op == "sum":
            axis = self.args[1]
            if grad.ndim != self.args[0].data.ndim and axis is not None:
                grad = np.expand_dims(grad, axis)
            self.args[0].backward(np.ones_like(self.args[0].data) * grad)

        elif self.op == "mean":
            axis = self.args[1]

            if grad.ndim != self.args[0].data.ndim  and axis is not None:
                grad = np.expand_dims(grad, axis)

            # if axis is None:
            #     self.args[0].backward(np.ones_like(self.args[0].data) * grad / self.args[0].data.size)
            # elif type(axis) is int:
            #     self.args[0].backward(np.ones_like(self.args[0].data) * grad / self.args[0].data.shape[axis])
            # else:
            #     self.args[0].backward(np.ones_like(self.args[0].data) * grad / np.prod([self.args[0].data.shape[i] for i in axis]))
            _axis = list(axis) if isinstance(axis, tuple) else axis
            self.args[0].backward(np.ones_like(self.args[0].data) * grad / np.prod(np.array(self.args[0].data.shape)[_axis]))
            
        elif self.op == "var": #axis=None, ddof=0, keepdims=False add params instead args kwargs
            axis = self.args[1]
          
            if grad.ndim != self.args[0].data.ndim and axis is not None:
                grad = np.expand_dims(grad, axis)
   
            # if axis is None:
            #     self.args[0].backward(np.ones_like(self.args[0].data) * grad * 2 * (self.args[0].data - self.args[0].data.mean()) / self.args[0].data.size)
            # elif type(axis) is int:
            #     self.args[0].backward(np.ones_like(self.args[0].data) * grad * 2 * (self.args[0].data - self.args[0].data.mean(axis=axis, keepdims=True)) / self.args[0].data.shape[axis])
            # else:
            #     self.args[0].backward(np.ones_like(self.args[0].data) * grad * 2 * (self.args[0].data - self.args[0].data.mean(axis=axis, keepdims=True)) / np.prod([self.args[0].data.shape[i] for i in axis]))
            _axis = list(axis) if isinstance(axis, tuple) else axis
            self.args[0].backward(np.ones_like(self.args[0].data) * grad * 2 * (self.args[0].data - self.args[0].data.mean(axis=axis, keepdims=True)) / np.prod(np.array(self.args[0].data.shape)[_axis]))

            

        elif self.op == "power":
            self.args[0].backward(grad * self.args[1].data * self.args[0].data ** (self.args[1].data - 1))
            self.args[1].backward(grad * self.args[0].data ** self.args[1].data * np.log(self.args[0].data))

        elif self.op == "sqrt":
            self.args[0].backward(grad * 1 / (2 * np.sqrt(self.args[0].data)))

        elif self.op == "log":
            self.args[0].backward(grad * 1 / self.args[0].data)

        elif self.op == "exp":
            self.args[0].backward(grad * np.exp(self.args[0].data))

        elif self.op == "tanh":
            self.args[0].backward(grad * (1 - np.tanh(self.args[0].data) ** 2))

        elif self.op == "sin":
            self.args[0].backward(grad * np.cos(self.args[0].data))

        elif self.op == "cos":
            self.args[0].backward(grad * -np.sin(self.args[0].data))

        elif self.op == "maximum":
            self.args[0].backward(grad * (self.args[0].data >= self.args[1].data))
            self.args[1].backward(grad * (self.args[0].data <= self.args[1].data))

        elif self.op == "minimum":
            self.args[0].backward(grad * (self.args[0].data <= self.args[1].data))
            self.args[1].backward(grad * (self.args[0].data >= self.args[1].data))

        elif self.op == "max":
            axis, keepdims = self.args[1:]
            if grad.ndim != self.args[0].data.ndim and axis is not None:
                grad = np.expand_dims(grad, axis)
            self.args[0].backward(grad * (self.args[0].data == self.args[0].data.max(axis=axis, keepdims=True)))

        elif self.op == "min":
            axis, keepdims = self.args[1:]
            if grad.ndim != self.args[0].data.ndim and axis is not None:
                grad = np.expand_dims(grad, axis)
            self.args[0].backward(grad * (self.args[0].data == self.args[0].data.min(axis=axis, keepdims=True)))

        elif self.op == "concatenate":
            axis = self.args[-1]
            args = self.args[:-1]
            args_shapes = [arg.data.shape for arg in args]

            grads = np.split(grad, np.cumsum([arg_shape[axis] for arg_shape in args_shapes])[:-1], axis=axis)
        
            for i, arg in enumerate(args):
                arg.backward(grads[i])
                
        elif self.op == "reshape":
            self.args[0].backward(grad.reshape(self.args[0].data.shape))

        # elif self.op == "split":
        #     self.args[0].backward(np.concatenate(grad, axis = 0))

        elif self.op == "getitem":
            _grad = np.zeros_like(self.args[0].data)
            _grad[self.args[1]] = grad

            self.args[0].backward(_grad)

        elif self.op == "transpose":
            self.args[0].backward(grad.transpose(self.args[1]))

        elif self.op == "swapaxes":
            self.args[0].backward(grad.swapaxes(self.args[1], self.args[2]))

        elif self.op == "flip":
            self.args[0].backward(np.flip(grad, axis=self.args[1]))

        elif self.op == "neg":
            self.args[0].backward(-grad)

        elif self.op == "pos":
            self.args[0].backward(grad)

        elif self.op == "abs":
            self.args[0].backward(grad * np.sign(self.args[0].data))







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
    #     new_shape = np.array(shape)
    #     new_shape[axis] = 1
    #     num_reps = np.prod(np.array(shape)[axis])
    #     # Can't use broadcast_to because of numpy bug: https://github.com/numpy/numpy/issues/9165
    #     # return anp.broadcast_to(anp.reshape(g, new_shape), shape), num_reps
    #     return np.reshape(g, new_shape) + np.zeros(shape, dtype=dtype), num_reps

    # elif self.op == "mean":
        # shape = self.args[0].data.shape
        # axis = self.args[1]
        # dtype = np.result_type(self.args[0].data)
        # g_repeated, num_reps = self.repeat_to_match_shape(grad, shape, dtype, axis, None)
        # print(f"g_repeated {g_repeated}")
        # self.args[0].backward(g_repeated / num_reps)



