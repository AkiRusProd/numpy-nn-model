from autograd import Tensor
import numpy as np



class _DropoutTensor(Tensor):
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        self.args[0].backward(grad * self.args[1])

class Dropout():
    def __init__(self, p = 0.5):
        self.p = p
        self.mask = None
        self.train = True

    def forward(self, X):
        if self.train:
            self.mask = np.random.binomial(1, 1 - self.p, size = X.data.shape)
        else:
            self.mask = 1

        self.O = X.data * self.mask

        return _DropoutTensor(self.O, [X, self.mask], "dropout")

    def __call__(self, X):
        return self.forward(X)
