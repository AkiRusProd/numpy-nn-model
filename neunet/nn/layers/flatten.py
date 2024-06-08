from neunet.autograd import Tensor
from neunet.nn.modules import Module


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, X: Tensor):
        if not isinstance(X, Tensor):
            raise TypeError("Input must be a tensor")

        start = X.ndim + self.start_dim if self.start_dim < 0 else self.start_dim
        end = X.ndim + self.end_dim if self.end_dim < 0 else self.end_dim
        new_shape = X.shape[:start] + (X.xp.prod(X.shape[start : end + 1]),) + X.shape[end + 1 :]

        return X.reshape(*new_shape)

    def __call__(self, X):
        return self.forward(X)
