import numpy as np
from autograd import Tensor


class Flatten():
    def __init__(self, start_dim = 1, end_dim = -1):
        self.start_dim = start_dim
        self.end_dim = end_dim


    def forward(self, x):
        start = x.ndim + self.start_dim if self.start_dim < 0 else self.start_dim
        end = x.ndim + self.end_dim if self.end_dim < 0 else self.end_dim
        new_shape = x.shape[:start] + (np.prod(x.shape[start:end + 1]),) + x.shape[end + 1:]
        
        return x.reshape(*new_shape)


    def __call__(self, x):
        return self.forward(x)
