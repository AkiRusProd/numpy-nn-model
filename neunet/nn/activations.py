from neunet.autograd import Tensor
from neunet.nn.modules import Module

# import numpy as np


class _SigmoidTensor(Tensor):  # Static sigmoid tensor for backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(x: Tensor, f_x, grad):
            x._apply_grad(grad * f_x * (1 - f_x))

        self.grad_fn = grad_fn


class Sigmoid(Module):  # Static sigmoid computation
    def __init__(self):
        pass

    def forward(self, x: Tensor):
        f_x = 1 / (1 + x.xp.exp(-x.data))
        return _SigmoidTensor(f_x, [x, f_x], "sigmoid", device=x.device)

    def __call__(self, x):
        return self.forward(x)


# class Sigmoid(Tensor): #Dynamic sigmoid computation (slower than static)
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.exp().div(x.exp().add(1))


#     def __call__(self, x):
#         return self.forward(x)


class _ReLUTensor(Tensor):  # Static ReLU tensor for backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(t: Tensor, f_x, grad):
            t._apply_grad(grad * (f_x > 0))

        self.grad_fn = grad_fn


class ReLU(Module):  # Static ReLU computation
    def __init__(self):
        pass

    def forward(self, x: Tensor):
        f_x = x.xp.maximum(0, x.data)
        return _ReLUTensor(f_x, [x, f_x], "relu", device=x.device)

    def __call__(self, x):
        return self.forward(x)


# class ReLU(Tensor): #Dynamic ReLU computation (slower than static)
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.maximum(0)

#     def __call__(self, x):
#         return self.forward(x)


class _LeakyReLUTensor(Tensor):  # Static LeakyReLU tensor for backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(t: Tensor, f_x, alpha, grad):
            t._apply_grad(
                grad * t.xp.where(f_x <= 0, alpha, 1).astype(grad.dtype)
            )

        self.grad_fn = grad_fn

class LeakyReLU(Module):  # Static LeakyReLU computation
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x: Tensor):
        f_x = x.xp.where(x.data <= 0, self.alpha * x.data, x.data).astype(x.dtype)
        return _LeakyReLUTensor(f_x, [x, f_x, self.alpha], "leakyrelu", device=x.device)

    def __call__(self, x):
        return self.forward(x)


# class LeakyReLU(Tensor): #Dynamic LeakyReLU computation (slower than static)
#     def __init__(self, alpha = 0.01):
#         self.alpha = alpha

#     def forward(self, x):
#         return x.maximum(0).add(x.minimum(0).mul(self.alpha))

#     def __call__(self, x):
#         return self.forward(x)


class _TanhTensor(Tensor):  # Static Tanh tensor for backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(t: Tensor, f_x, grad):
            t._apply_grad(grad * (1 - f_x ** 2))

        self.grad_fn = grad_fn


class Tanh(Module):  # Static Tanh computation
    def __init__(self):
        pass

    def forward(self, x: Tensor):
        f_x = x.xp.tanh(x.data)
        return _TanhTensor(f_x, [x, f_x], "tanh", device=x.device)

    def __call__(self, x):
        return self.forward(x)


# class Tanh(Tensor): #Dynamic Tanh computation (slower than static)
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.exp().sub(x.mul(-1).exp()).div(x.exp().add(x.mul(-1).exp()))

#     def __call__(self, x):
#         return self.forward(x)


class _SoftplusTensor(Tensor):  # Static Softplus tensor for backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(t: Tensor, grad):
            x = t.data
            t._apply_grad(grad * (1 / (1 + t.xp.exp(-x))))

        self.grad_fn = grad_fn


class Softplus(Module):  # Static Softplus computation
    def __init__(self):
        pass

    def forward(self, x: Tensor):
        f_x = x.xp.log(1 + x.xp.exp(x.data))
        return _SoftplusTensor(f_x, [x], "softplus", device=x.device)

    def __call__(self, x):
        return self.forward(x)


# class Softplus(Tensor):
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.exp().add(1).log()

#     def __call__(self, x):
#         return self.forward(x)


class _SoftsignTensor(Tensor):  # Static Softsign tensor for backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(t: Tensor, grad):
            x = t.data
            t._apply_grad(grad * (1 / (1 + t.xp.abs(x)) ** 2))

        self.grad_fn = grad_fn


class Softsign(Module):  # Static Softsign computation
    def __init__(self):
        pass

    def forward(self, x: Tensor):
        f_x = x.data / (1 + x.xp.abs(x.data))
        return _SoftsignTensor(f_x, [x], "softsign", device=x.device)

    def __call__(self, x):
        return self.forward(x)


# class Softsign(Tensor):
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.div(x.abs().add(1))

#     def __call__(self, x):
#         return self.forward(x)


class _SwishTensorTensor(Tensor):  # Static Swish tensor for backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(t: Tensor, f_x, beta, grad):
            x = t.data
            sigmoid = lambda x: 1 / (1 + t.xp.exp(-x))

            t._apply_grad(grad * (beta * f_x + sigmoid(beta * x) * (1 - beta * f_x)))

        self.grad_fn = grad_fn


class Swish(Module):  # Static Swish computation
    def __init__(self, beta=1):
        self.beta = beta

    def forward(self, x: Tensor):
        xp = x.xp
        sigmoid = lambda x: 1 / (1 + xp.exp(-x))
        f_x = x.data * sigmoid(self.beta * x.data)

        return _SwishTensorTensor(f_x, [x, f_x, self.beta], "swish", device=x.device)

    def __call__(self, x):
        return self.forward(x)


# class Swish(Tensor): #Dynamic Swish computation (slower than static)
#     def __init__(self, beta = 1):
#         self.beta = beta

#     def forward(self, x):
#         z = x.mul(self.beta)
#         sigmoid = z.exp().div(z.exp().add(1))

#         return x.mul(sigmoid)

#     def __call__(self, x):
#         return self.forward(x)


class _MishTensor(Tensor):  # Static Mish tensor for backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(t: Tensor, grad):
            xp = t.xp
            x = t.data

            grad_x = grad * (
                xp.exp(x)
                * (4 * (x + 1) + 4 * xp.exp(2 * x) + xp.exp(3 * x) + xp.exp(x) * (4 * x + 6))
                / xp.power((2 * xp.exp(x) + xp.exp(2 * x) + 2), 2)
            )

            t._apply_grad(grad_x)

        self.grad_fn = grad_fn


class Mish(Module):  # Static Mish computation
    def __init__(self):
        pass

    def forward(self, x: Tensor):
        f_x = x.data * x.xp.tanh(x.xp.log(1 + x.xp.exp(x.data)))

        return _MishTensor(f_x, [x], "mish", device=x.device)

    def __call__(self, x):
        return self.forward(x)


# class Mish(Tensor): #Dynamic Mish computation (slower than static)
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.mul(x.tanh().mul(x.exp().add(1)).log())

#     def __call__(self, x):
#         return self.forward(x)


class _TanhExpTensor(Tensor):  # Static TanhExp tensor for backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(t: Tensor, grad):
            xp = t.xp
            x = t.data

            grad_x = grad * (xp.tanh(xp.exp(x)) - x * xp.exp(x) * (xp.power(xp.tanh(xp.exp(x)), 2) - 1))

            t._apply_grad(grad_x)

        self.grad_fn = grad_fn

class TanhExp(Module):  # Static TanhExp computation
    def __init__(self):
        pass

    def forward(self, x: Tensor):
        f_x = x.data * x.xp.tanh(x.xp.exp(x.data))

        return _TanhExpTensor(f_x, [x], "tanh_exp", device=x.device)

    def __call__(self, x):
        return self.forward(x)


# class TanhExp(Tensor): #Dynamic TanhExp computation (slower than static)
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.mul(x.exp().tanh())

#     def __call__(self, x):
#         return self.forward(x)


class _ELUTensor(Tensor):  # Static ELU tensor for backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(t: Tensor, f_x, alpha, grad):
            x = t.data
            grad_x = grad * (t.xp.where(x <= 0, alpha + f_x, 1).astype(grad.dtype))

            t._apply_grad(grad_x)

        self.grad_fn = grad_fn


class ELU(Module):  # Static ELU computation
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def forward(self, x: Tensor):
        f_x = x.xp.where(x.data <= 0, self.alpha * (x.xp.exp(x.data) - 1), x.data).astype(x.dtype)

        return _ELUTensor(f_x, [x, f_x, self.alpha], "elu", device=x.device)

    def __call__(self, x):
        return self.forward(x)


class _SELUTensor(Tensor):  # Static SELU tensor for backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(t: Tensor, alpha, lmbda, grad):
            x = t.data
            grad_x = grad * (lmbda * t.xp.where(x > 0, 1, alpha * t.xp.exp(x)).astype(grad.dtype))

            t._apply_grad(grad_x)

        self.grad_fn = grad_fn


class SELU(Module):  # Static SELU computation
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.lmbda = 1.0507009873554804934193349852946

    def forward(self, x: Tensor):
        f_x = self.lmbda * x.xp.where(
            x.data > 0, x.data, self.alpha * (x.xp.exp(x.data) - 1).astype(x.dtype)
        )

        return _SELUTensor(f_x, [x, self.alpha, self.lmbda], "selu", device=x.device)

    def __call__(self, x):
        return self.forward(x)


class _GELUTensor(Tensor):  # Static GELU tensor for backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(t: Tensor, grad):
            xp = t.xp
            x = t.data
            # sech = lambda z: 2 / (np.exp(z) + np.exp(-z))
            sech = lambda z: 1 / xp.cosh(z)

            grad_x = grad * (
                0.5 * xp.tanh(0.0356774 * xp.power(x, 3) + 0.797885 * x)
                + (0.0535161 * xp.power(x, 3) + 0.398942 * x)
                * xp.power(sech(0.0356774 * xp.power(x, 3) + 0.797885 * x), 2)
                + 0.5
            )

            t._apply_grad(grad_x)

        self.grad_fn = grad_fn


class GELU(Module):  # Static GELU computation
    def __init__(self):
        pass

    def forward(self, x: Tensor):
        f_x = (
            0.5
            * x.data
            * (1 + x.xp.tanh(x.xp.sqrt(2 / x.xp.pi) * (x.data + 0.044715 * x.xp.power(x.data, 3))))
        )

        return _GELUTensor(f_x, [x], "gelu", device=x.device)

    def __call__(self, x):
        return self.forward(x)


# class Softmax(Module):  # Dynamic Softmax computation
#     def __init__(self, axis=1):
#         self.axis = axis

#     def forward(self, x: Tensor):
#         e_x = x.sub(x.max(axis=self.axis, keepdims=True)).exp()
#         return e_x.div(e_x.sum(axis=self.axis, keepdims=True))

#     def __call__(self, x):
#         return self.forward(x)


class _SoftmaxTensor(Tensor):  # Static Softmax tensor for backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(t: Tensor, f_x, axis, grad):
            grad_x=(grad - (grad * f_x).sum(axis, keepdims=True)) * f_x

            t._apply_grad(grad_x)

        self.grad_fn = grad_fn

class Softmax(Module):  # Static Softmax computation
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x: Tensor):
        e_x = x.xp.exp(x.data - x.xp.max(x.data, axis = self.axis, keepdims=True))
        
        f_x =  e_x / x.xp.sum(e_x, axis = self.axis, keepdims=True)
        return _SoftmaxTensor(f_x, [x, f_x, self.axis], "softmax", device=x.device)

    def __call__(self, x):
        return self.forward(x)


class _LogSoftmax(Tensor):  # Static LogSoftmax tensor for backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(t: Tensor, f_x, axis, grad):
            softmax = t.xp.exp(f_x) # e^(loge_softmax) = softmax

            grad_x = grad - softmax * grad.sum(axis = axis, keepdims=True)

            t._apply_grad(grad_x)

        self.grad_fn = grad_fn


class LogSoftmax(Module):  
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x: Tensor):
        e_x = x.xp.exp(x.data - x.xp.max(x.data, axis = self.axis, keepdims=True))
        
        f_x =  x.xp.log(e_x / x.xp.sum(e_x, axis = self.axis, keepdims=True))
        return _LogSoftmax(f_x, [x, f_x, self.axis], "log_softmax", device=x.device)

    def __call__(self, x):
        return self.forward(x)