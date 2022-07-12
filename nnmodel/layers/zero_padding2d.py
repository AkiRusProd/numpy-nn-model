import numpy as np
from numba import njit
from nnmodel.exceptions.values_checker import ValuesChecker

class ZeroPadding2D():
    """
    Add padding with zeros to the input data
    ----------------------------------------
        Args:
            `padding` (tuple), (list) of size 2 or (int): padding size for 2d input data with shape: (batchsize, channels, H, W)
        Returns:
            output: the padded input data with shape: (batchsize, channels, H + 2 * padding[0], W + 2 * padding[1])
    """

    def __init__(self, padding = (0, 0), input_shape = None) -> None:
        self.padding      = ValuesChecker.check_size2_variable(padding, variable_name = "padding", min_acceptable_value = 0)
        self.input_shape = ValuesChecker.check_input_dim(input_shape, input_dim = 3)

    def build(self):
        self.padding = (self.padding[0], self.padding[0], self.padding[1], self.padding[1])
        self.output_shape = (self.input_shape[0], self.input_shape[1] + 2 * self.padding[0], self.input_shape[2] + 2 * self.padding[1])

    def forward_prop(self, X, training):
        
        return self.set_padding(X, self.padding)

    def backward_prop(self, error):
        
        return self.remove_padding(error, self.padding)


    @staticmethod
    @njit
    def set_padding(layer, padding):
        # padded_layer = np.pad(layer, ((0, 0), (0, 0), (padding[0], padding[1]), (padding[1], padding[0])), constant_values = 0)
        padded_layer = np.zeros(
            (   
                layer.shape[0],
                layer.shape[1],
                layer.shape[2] + padding[0] + padding[1],
                layer.shape[3] + padding[2] + padding[3],
            )
        )

        padded_layer[
                    :,
                    :,
                    padding[0] :padded_layer.shape[2] - padding[1],
                    padding[2] :padded_layer.shape[3] - padding[3],
                ] = layer

        return padded_layer

    @staticmethod
    @njit
    def remove_padding(layer, padding):
        # unpadded_layer = unpadded_layer[:, :, padding[0]:-padding[1], padding[1]:-padding[0]]
        unpadded_layer = np.zeros(
            (
                layer.shape[0],
                layer.shape[1],
                layer.shape[2] - padding[0] - padding[1],
                layer.shape[3] - padding[2] - padding[3],
            )
        )
        
        unpadded_layer = layer[
                    :,
                    :,
                    padding[0] :layer.shape[2] - padding[1],
                    padding[2] :layer.shape[3] - padding[3],
                ]

        return unpadded_layer