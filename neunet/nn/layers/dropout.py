from neunet.autograd import Tensor
import numpy as np



class _DropoutTensor(Tensor): # tensor for static backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        self.args[0].backward(grad * self.args[1])

class Dropout(): # layer with static backpropagation
    def __init__(self, p = 0.5):
        self.p = p
        self.scale = 1 / (1 - p)
        self.mask = None
        self.training = True

    def forward(self, X):
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, size = X.data.shape) * self.scale
        else:
            self.mask = 1

        self.O = X.data * self.mask

        return _DropoutTensor(self.O, [X, self.mask], "dropout")

    def __call__(self, X):
        return self.forward(X)

    def train(self, mode = True):
        self.training = mode

    def eval(self):
        self.training = False


# class Dropout(): # layer with dynamic backpropagation
#     def __init__(self, p = 0.5):
#         self.p = p
#         self.scale = 1 / (1 - p)
#         self.training = True

#     def forward(self, X):
#         if self.training:
#             mask = np.random.binomial(1, 1 - self.p, size = X.data.shape) * self.scale
#         else:
#             mask = 1

#         return X * mask

#     def __call__(self, X):
#         return self.forward(X)

    # def train(self, mode = True):
    #     self.training = mode

    # def eval(self):
    #     self.training = False