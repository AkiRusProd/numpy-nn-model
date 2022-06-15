import numpy as np


class Reshape():

    def __init__(self, shape) -> None:
        self.shape = shape

    def forward_prop(self, X, training):
        self.prev_shape = X.shape
        
        return X.reshape(self.prev_shape[0], *self.shape)

    def backward_prop(self, error):
        
        return error.reshape(self.prev_shape)