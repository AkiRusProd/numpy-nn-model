import numpy as np
from numba import njit
from nnmodel.values_checker import ValuesChecker

class Pooling2D():
    # add strides, padding= same, valid, custom
    def __init__(self,  pool_size=(2, 2), input_shape = None):
        self.pool_size = ValuesChecker.check_size2_variable(pool_size, variable_name = "pool_size")
        self.input_shape = input_shape

    def build(self, optimizer):
        self.block_size = self.input_shape[1] // self.pool_size[0], self.input_shape[2] // self.pool_size[1]
        self.output_shape = (self.input_shape[0], *self.block_size)
        


    @staticmethod
    @njit
    def pooling(conv_layer, conv_shape, pool_size, block_size, pooling_type):

        pooling_layer = np.zeros((conv_shape[0], conv_shape[1], pool_size[0], pool_size[1]))
        pooling_layer_ind = np.zeros((conv_shape[0], conv_shape[1], conv_shape[2], conv_shape[3]))

        for b in range(conv_shape[0]):
            for s in range(conv_shape[1]):
                for h in range(pool_size[0]):
                    for w in range(pool_size[1]):

                        pool_part = conv_layer[
                            b,
                            s,
                            h * block_size[0] : h * block_size[0] + block_size[0],
                            w * block_size[1] : w * block_size[1] + block_size[1],
                        ]  # .astype(np.float64)

                        if pooling_type == "MaxPooling":
                            pooling_layer[b, s, h, w] = pool_part.max()
                        elif pooling_type == "AvgPooling":
                            pooling_layer[b, s, h, w] = pool_part.mean()
                        # pooling_layer[b, s, h, w] = pooling_type(pool_part)

                        for i in range(block_size[0]):
                            for j in range(block_size[1]):
                                if pool_part[i, j] == pooling_layer[b, s, h, w]:
                                    I = int(i + h * block_size[0])

                                    J = int(j + w * block_size[1])

                        pooling_layer_ind[b, s, I, J] = 1

        return pooling_layer, pooling_layer_ind

    @staticmethod
    @njit
    def pooling_error_backward_prop(
        pooling_layer_ind, error, conv_shape
    ):  
        
        error = error.reshape((error.shape[0], error.shape[1], error.shape[2] * error.shape[3]))

        output_layer_error = np.zeros((conv_shape[0], conv_shape[1], conv_shape[2], conv_shape[3]))

        for b in range(conv_shape[0]):
            for s in range(conv_shape[1]):
                i = 0
                for h in range(conv_shape[2]):
                    for w in range(conv_shape[3]):
                        if pooling_layer_ind[b, s, h, w] == 1:
                            output_layer_error[b, s, h, w] = error[b, s, i]
                            i += 1

        return output_layer_error


class MaxPooling2D(Pooling2D):

    def __init__(self,  pool_size=(2, 2)):
        self.pool_size = pool_size

    def forward_prop(self, X, training):
        self.input_data = X
        # self.block_size = self.input_data.shape[2] // self.pool_size[0], self.input_data.shape[3], self.pool_size[1]

        self.pooling_layer, self.pooling_layer_ind = self.pooling(self.input_data, self.input_data.shape, self.block_size, self.pool_size, "MaxPooling")

        return self.pooling_layer

    def backward_prop(self, error):
        error = error.copy()

        return self.pooling_error_backward_prop(self.pooling_layer_ind, error, self.input_data.shape)



class AveragePooling2D(Pooling2D):
    def __init__(self,  pool_size=(2, 2)):
        self.pool_size = pool_size

    def forward_prop(self, X, training):
        self.input_data = X
        # self.block_size = self.input_data.shape[1] // self.pool_size[0], self.input_data.shape[2], self.pool_size[1]

        self.pooling_layer, self.pooling_layer_ind = self.pooling(self.input_data, self.input_data.shape, self.block_size, self.pool_size, "AvgPooling")

        return self.pooling_layer

    def backward_prop(self, error):

        return self.pooling_error_backward_prop(self.pooling_layer_ind, error, self.input_data.shape)





    