import numpy as np
from NNModel.activations import activations


class Activation():

    def __init__(self, activation = None):
        self.input_shape = None

        if type(activation) is str or activation is None:
            self.activation = activations[activation]
        else:
            self.activation = activation

    def build(self, optimizer):
        self.output_shape = self.input_shape

    def forward_prop(self, X, training):
        self.layer_input = X
        return self.activation.function(X)

    def backward_prop(self, error):
        return error * self.activation.derivative(self.layer_input)
