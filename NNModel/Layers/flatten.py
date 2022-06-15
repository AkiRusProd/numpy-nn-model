import numpy as np


class Flatten():

    def __init__(self) -> None:
        pass

    def forward_prop(self, X, training):
        self.prev_shape = X.shape

        return X.reshape(self.prev_shape[0], np.prod(self.prev_shape[1:]))

    def backward_prop(self, error):
        
        return error.reshape(self.prev_shape)