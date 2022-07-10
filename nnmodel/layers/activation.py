import numpy as np
from nnmodel.activations import activations
from nnmodel.exceptions.values_checker import ValuesChecker

class Activation():

    def __init__(self, activation = None):
        self.input_shape = None

        self.activation = ValuesChecker.check_activation(activation, activations)

    def build(self):
        self.output_shape = self.input_shape

    def forward_prop(self, X, training):
        self.input_data = X
        return self.activation.function(X)

    def backward_prop(self, error):
        return error * self.activation.derivative(self.input_data)
