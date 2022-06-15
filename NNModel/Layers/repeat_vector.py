import numpy as np


class RepeatVector():

    def __init__(self, num) -> None:
        self.num = num

    def forward_prop(self, X, training):
        
        return np.tile(X, (self.num, 1))

    def backward_prop(self, error):
        
        return np.sum(error, axis = 0)