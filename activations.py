from autograd import Tensor
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

    def __call__(self, x, training = True):
        return self.forward(x)
        

# class Sigmoid(Tensor): #Dynamic sigmoid computation (slower than static)
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.exp().div(x.exp().add(1))
        
 
#     def __call__(self, x, training = True):
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

    def __call__(self, x, training = True):
        return self.forward(x)



# class ReLU(Tensor): #Dynamic ReLU computation (slower than static)
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.maximum(0)

#     def __call__(self, x, training = True):
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

    def __call__(self, x, training = True):
        return self.forward(x)


# class LeakyReLU(Tensor): #Dynamic LeakyReLU computation (slower than static)
#     def __init__(self, alpha = 0.01):
#         self.alpha = alpha

#     def forward(self, x):
#         return x.maximum(0).add(x.minimum(0).mul(self.alpha))

#     def __call__(self, x, training = True):
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

    def __call__(self, x, training = True):
        return self.forward(x)



# class Tanh(Tensor): #Dynamic Tanh computation (slower than static)
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.exp().sub(x.mul(-1).exp()).div(x.exp().add(x.mul(-1).exp()))

#     def __call__(self, x, training = True):
#         return self.forward(x)
