from autograd import Tensor
import numpy as np



class SigmoidTensor(Tensor): #Static sigmoid tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
       
        self.args[0].backward(grad * self.data * (1 - self.data))

class Sigmoid(): #Static sigmoid computation
    def __init__(self):
        pass

    def forward(self, x):
        f_x = 1 / (1 + np.exp(-x.data))
        return SigmoidTensor(f_x, [x], "sigmoid")

    def __call__(self, x, training = True):
        return self.forward(x)
        

# class Sigmoid(Tensor):
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x.exp().div(x.exp().add(1))
        
 
#     def __call__(self, x, training = True):
#         return self.forward(x)


class ReLU(Tensor):
    def __init__(self):
        pass

    def forward(self, x):
        return x.maximum(0)

    def __call__(self, x, training = True):
        return self.forward(x)


class LeakyReLU(Tensor):
    def __init__(self, alpha = 0.01):
        self.alpha = alpha

    def forward(self, x):
        return x.maximum(0).add(x.minimum(0).mul(self.alpha))

    def __call__(self, x, training = True):
        return self.forward(x)

class Tanh(Tensor):
    def __init__(self):
        pass

    def forward(self, x):
        return x.exp().sub(x.mul(-1).exp()).div(x.exp().add(x.mul(-1).exp()))

    def __call__(self, x, training = True):
        return self.forward(x)
