import numpy as np
from nnmodel.exceptions.values_checker import ValuesChecker

class Reshape():
    """
    Reshape the input data to the desired shape
    -------------------------------------------
        Args:
            `shape` (tuple) or (list): desired shape of the input data 
            without touching the batch size dimension
        Returns:
            output: the reshaped input data with shape: (batchsize, *shape)
    """

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