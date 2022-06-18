import numpy as np
from numba import njit


class UpSampling2D():
    #TODO
     #verify speed of METHODS/unpadding; maybe native numpy is faster than numba
     #add other interpolations, "bicubic", "bilinear", now only nearest 

    def __init__(self,  size=(2, 2)) -> None:
        self.size = size
        self.input_shape = None

    def build(self, optimizer):
        self.output_shape = (self.input_shape[0], self.input_shape[1] * self.size[0], self.input_shape[2] * self.size[1])

    def forward_prop(self, X, training):

        return self.make_upsampling(X, self.size)

    def backward_prop(self, error):
        
        return self.make_downsampling(error, self.size)


    @staticmethod
    @njit
    def make_upsampling(layer, scale_factor):  # njit doesnt work
        # upsampled_layer = layer.repeat(scale_factor, axis=1).repeat(scale_factor, axis=2)
        upsampled_layer = np.zeros(
            (   
                layer.shape[0],
                layer.shape[1],
                layer.shape[2] * scale_factor[0],
                layer.shape[3] * scale_factor[1],
            )
        )

        ic, jc = 0, 0

        for b in range(layer.shape[0]):
            for k in range(layer.shape[1]):
                for i in range(layer.shape[2]):
                    for j in range(layer.shape[3]):
                        upsampled_layer[
                            b,
                            k,
                            i + ic : (i + 1) * scale_factor[0],
                            j + jc : (j + 1) * scale_factor[1],
                        ] = layer[b, k, i, j]

                        jc += scale_factor[1] - 1
                    ic += scale_factor[0] - 1

                    jc = 0

                ic = 0

        return upsampled_layer

    @staticmethod
    @njit
    def make_downsampling(layer, scale_factor):
        # downsampled_layer = layer[..., :layer.shape[1], : layer.shape[1]].reshape(layer.shape[0], layer.shape[1]//scale_factor, scale_factor, layer.shape[1]//scale_factor, scale_factor).mean(axis=(2,4))

        downsampled_layer = np.zeros(
            (
                layer.shape[0],
                layer.shape[1],
                layer.shape[2] // scale_factor[0],
                layer.shape[3] // scale_factor[1],
            )
        )

        ic, jc = 0, 0

        for b in range(downsampled_layer.shape[0]):
            for k in range(downsampled_layer.shape[1]):
                for i in range(downsampled_layer.shape[2]):
                    for j in range(downsampled_layer.shape[3]):
                        downsampled_layer[b, k, i, j] = layer[
                            b,
                            k,
                            i + ic : (i + 1) * scale_factor[0],
                            j + jc : (j + 1) * scale_factor[1],
                        ].mean()

                        jc += scale_factor[1] - 1
                    ic += scale_factor[0] - 1

                    jc = 0

                ic = 0

        return downsampled_layer