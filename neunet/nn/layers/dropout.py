from neunet.autograd import Tensor
from neunet.nn.modules import Module


class _DropoutTensor(Tensor):  # tensor for static backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(X: Tensor, mask, grad):
            X._apply_grad(grad * mask)

        self.grad_fn = grad_fn




class Dropout(Module):  # layer with static backpropagation
    def __init__(self, p: float=0.5):
        self.p = p
        self.scale = 1 / (1 - p)
        self.mask = 1
        self.training = True

    def forward(self, X: Tensor) -> Tensor:
        if not isinstance(X, Tensor):
            raise TypeError("Input must be a tensor")

        if self.training:
            self.mask = (
                X.xp.random.binomial(1, 1 - self.p, size=X.data.shape).astype(X.data.dtype)
                * self.scale
            )
        else:
            self.mask = 1

        self.O = X.data * self.mask

        return _DropoutTensor(self.O, [X, self.mask], "dropout", device=X.device)

    def __call__(self, X):
        return self.forward(X)

    def train(self, mode=True):
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
#             mask = X.xp.random.binomial(1, 1 - self.p, size = X.data.shape) * self.scale
#         else:
#             mask = 1

#         return X * mask

#     def __call__(self, X):
#         return self.forward(X)

# def train(self, mode = True):
#     self.training = mode

# def eval(self):
#     self.training = False
