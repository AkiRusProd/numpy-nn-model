import numpy as np


class Flatten:
    def __init__(self) -> None:
        self.input_shape = None

    def build(self):
        self.output_shape = (1, int(np.prod(self.input_shape)))  # [1:]

    def forward_prop(self, X, training):
        self.prev_shape = X.shape

        return X.reshape(self.prev_shape[0], int(np.prod(self.prev_shape[1:])))

    def backward_prop(self, error):

        return error.reshape(self.prev_shape)
