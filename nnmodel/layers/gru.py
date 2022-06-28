import numpy as np
from nnmodel.activations import activations
from nnmodel.exceptions.values_checker import ValuesChecker

class GRU():

    def __init__(self):
        pass

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def build(self):
        pass

    def forward_prop(self, X, training):
        self.input_data = X
        self.batch_size = len(self.input_data)

        if len(self.input_data.shape) == 2:
            self.input_data = self.input_data[:, np.newaxis, :]
            
        pass

    def backward_prop(self,):
        pass