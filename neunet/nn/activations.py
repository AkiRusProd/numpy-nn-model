from neunet.autograd import Tensor
import numpy as np



class _SigmoidTensor(Tensor): #Static sigmoid tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
       
        self.args[0].backward(grad * self.data * (1 - self.data))

class Sigmoid(): #Static sigmoid computation
    def __init__(self):
        pass

    def forward(self, x):
        f_x = 1 / (1 + np.exp(-x.data))
        return _SigmoidTensor(f_x, [x], "sigmoid")

    def __call__(self, x):
        return self.forward(x)
        

# class Sigmoid(Tensor): #Dynamic sigmoid computation (slower than static)
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.exp().div(x.exp().add(1))
        
 
#     def __call__(self, x):
#         return self.forward(x)



class _ReLUTensor(Tensor): #Static ReLU tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        self.args[0].backward(grad * (self.data > 0))

class ReLU(): #Static ReLU computation
    def __init__(self):
        pass

    def forward(self, x):
        f_x = np.maximum(0, x.data)
        return _ReLUTensor(f_x, [x], "relu")

    def __call__(self, x):
        return self.forward(x)



# class ReLU(Tensor): #Dynamic ReLU computation (slower than static)
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.maximum(0)

#     def __call__(self, x):
#         return self.forward(x)




class _LeakyReLUTensor(Tensor): #Static LeakyReLU tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        self.args[0].backward(grad * np.where(self.data <= 0, self.args[1], 1))

class LeakyReLU(): #Static LeakyReLU computation
    def __init__(self, alpha = 0.01):
        self.alpha = alpha

    def forward(self, x):
        f_x = np.where(x.data <= 0, self.alpha * x.data, x.data)
        return _LeakyReLUTensor(f_x, [x, self.alpha], "leakyrelu")

    def __call__(self, x):
        return self.forward(x)


# class LeakyReLU(Tensor): #Dynamic LeakyReLU computation (slower than static)
#     def __init__(self, alpha = 0.01):
#         self.alpha = alpha

#     def forward(self, x):
#         return x.maximum(0).add(x.minimum(0).mul(self.alpha))

#     def __call__(self, x):
#         return self.forward(x)



class _TanhTensor(Tensor): #Static Tanh tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        self.args[0].backward(grad * (1 - self.data ** 2))

class Tanh(): #Static Tanh computation
    def __init__(self):
        pass

    def forward(self, x):
        f_x = np.tanh(x.data)
        return _TanhTensor(f_x, [x], "tanh")

    def __call__(self, x):
        return self.forward(x)



# class Tanh(Tensor): #Dynamic Tanh computation (slower than static)
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.exp().sub(x.mul(-1).exp()).div(x.exp().add(x.mul(-1).exp()))

#     def __call__(self, x):
#         return self.forward(x)


class _SoftplusTensor(Tensor): #Static Softplus tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        x = self.args[0].data
        self.args[0].backward(grad * (1 / (1 + np.exp(-x))))

class Softplus(): #Static Softplus computation
    def __init__(self):
        pass

    def forward(self, x):
        f_x = np.log(1 + np.exp(x.data))
        return _SoftplusTensor(f_x, [x], "softplus")

    def __call__(self, x):
        return self.forward(x)

# class Softplus(Tensor):
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.exp().add(1).log()

#     def __call__(self, x):
#         return self.forward(x)


class _SoftsignTensor(Tensor): #Static Softsign tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        x = self.args[0].data
        self.args[0].backward(grad * (1 / (1 + np.abs(x)) ** 2))

class Softsign(): #Static Softsign computation
    def __init__(self):
        pass

    def forward(self, x):
        f_x = x.data / (1 + np.abs(x.data))
        return _SoftsignTensor(f_x, [x], "softsign")

    def __call__(self, x):
        return self.forward(x)

# class Softsign(Tensor):
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.div(x.abs().add(1))

#     def __call__(self, x):
#         return self.forward(x)


class _SwishTensorTensor(Tensor): #Static Swish tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        x, beta = self.args[0].data, self.args[1]
        f_x = self.data

        sigmoid = lambda x: 1 / (1 + np.exp(-x))

        self.args[0].backward(grad * (beta * f_x + sigmoid(beta * x) * (1 - beta * f_x)))

class Swish(): #Static Swish computation
    def __init__(self, beta = 1):
        self.beta = beta

    def forward(self, x):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        f_x = x.data * sigmoid(self.beta * x.data)

        return _SwishTensorTensor(f_x, [x, self.beta], "swish")

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


class _MishTensor(Tensor): #Static Mish tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        x = self.args[0].data

        grad_x = grad * (np.exp(x) * (4 * (x + 1) + 4 * np.exp(2 * x) + np.exp(3 * x) + np.exp(x) * (4 * x + 6)) / np.power((2 * np.exp(x) + np.exp(2 * x) + 2), 2))

        self.args[0].backward(grad_x)

class Mish(): #Static Mish computation
    def __init__(self):
        pass

    def forward(self, x):
        f_x = x.data * np.tanh(np.log(1 + np.exp(x.data)))

        return _MishTensor(f_x, [x], "mish")

    def __call__(self, x):
        return self.forward(x)

# class Mish(Tensor): #Dynamic Mish computation (slower than static)
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.mul(x.tanh().mul(x.exp().add(1)).log())

#     def __call__(self, x):
#         return self.forward(x)


class _TanhExpTensor(Tensor): #Static TanhExp tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        x = self.args[0].data
        grad_x = grad * (np.tanh(np.exp(x)) - x * np.exp(x) * (np.power(np.tanh(np.exp(x)), 2) - 1))

        self.args[0].backward(grad_x)

class TanhExp(): #Static TanhExp computation
    def __init__(self):
        pass

    def forward(self, x):
        f_x = x.data * np.tanh(np.exp(x.data))

        return _TanhExpTensor(f_x, [x], "tanh_exp")

    def __call__(self, x):  
        return self.forward(x)

# class TanhExp(Tensor): #Dynamic TanhExp computation (slower than static)
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.mul(x.exp().tanh())

#     def __call__(self, x):
#         return self.forward(x)


class _ELUTensor(Tensor): #Static ELU tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        x, alpha = self.args[0].data, self.args[1]
        f_x = self.data

        grad_x = grad * (np.where(x <= 0, alpha + f_x, 1))

        self.args[0].backward(grad_x)

class ELU(): #Static ELU computation
    def __init__(self, alpha = 0.1):
        self.alpha = alpha

    def forward(self, x):
        f_x =  np.where(x.data <= 0, self.alpha * (np.exp(x.data) - 1), x.data)

        return _ELUTensor(f_x, [x, self.alpha], "elu")

    def __call__(self, x):
        return self.forward(x)


class _SELUTensor(Tensor): #Static SELU tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        x, alpha, lmbda = self.args[0].data, self.args[1], self.args[2]
        f_x = self.data

        grad_x = grad * (lmbda * np.where(x > 0, 1, alpha * np.exp(x)))

        self.args[0].backward(grad_x)

class SELU(): #Static SELU computation
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.lmbda = 1.0507009873554804934193349852946 

    def forward(self, x):
        f_x = self.lmbda * np.where(x.data > 0, x.data, self.alpha*(np.exp(x.data)-1))

        return _SELUTensor(f_x, [x, self.alpha, self.lmbda], "selu")

    def __call__(self, x):
        return self.forward(x)


class _GELUTensor(Tensor): #Static GELU tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        x = self.args[0].data
        f_x = self.data

        # sech = lambda z: 2 / (np.exp(z) + np.exp(-z))
        sech = lambda z: 1 / np.cosh(z)

        grad_x = grad * (
            0.5 * np.tanh(0.0356774 * np.power(x, 3) + 0.797885 * x)
            + (0.0535161 * np.power(x, 3) + 0.398942 * x)
            * np.power(sech(0.0356774 * np.power(x, 3) + 0.797885 * x), 2)
            + 0.5
        )

        self.args[0].backward(grad_x)

class GELU(): #Static GELU computation
    def __init__(self):
        pass

    def forward(self, x):
        f_x = 0.5 * x.data * (1 + np.tanh(np.sqrt(2 / np.pi) * (x.data + 0.044715 * np.power(x.data, 3))))

        return _GELUTensor(f_x, [x], "gelu")

    def __call__(self, x):
        return self.forward(x)


class Softmax(): #Dynamic Softmax computation
    def __init__(self, axis = 1):
        self.axis = axis

    def forward(self, x):
        e_x = x.sub(x.max(axis=self.axis, keepdims=True)).exp()
        return e_x.div(e_x.sum(axis=self.axis, keepdims=True))

    def __call__(self, x):
        return self.forward(x)






    
activations= {
    "sigmoid": Sigmoid(),
    "tanh": Tanh(),
    "softmax": Softmax(),
    "softplus": Softplus(),
    "softsign": Softsign(),
    "swish": Swish(),
    "mish": Mish(),
    "tanh_exp": TanhExp(),
    "relu": ReLU(),
    "leaky_relu": LeakyReLU(),
    "elu": ELU(),
    "selu": SELU(),
    "gelu": GELU(),    
}