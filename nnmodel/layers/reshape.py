import numpy as np
from nnmodel.exceptions.values_checker import ValuesChecker

class Reshape():

    def __init__(self, shape) -> None:
        self.shape = ValuesChecker.check_shape(shape)
        self.input_shape = None

    def build(self):
        self.output_shape = self.shape


    def forward_prop(self, X, training):
        self.prev_shape = X.shape
        
        return X.reshape(self.prev_shape[0], *self.shape)

    def backward_prop(self, error):
        
        return error.reshape(self.prev_shape)