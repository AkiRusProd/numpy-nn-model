from matplotlib.pyplot import axis
import numpy as np


class Flatten():

    def __init__(self) -> None:
        pass

    def forward_prop(self, X):
        self.prev_shape = X.shape
        
        return X.flatten(axis = 0)

    def backward_prop(self, error):
        
        return error.reshape(self.prev_shape)