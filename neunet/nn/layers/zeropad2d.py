import numpy as np
from neunet.autograd import Tensor


class ZeroPad2dTensor(Tensor):
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        X, padding = self.args
        
        if X.data.ndim == 3:
            unpadded_grad = remove_padding(grad.reshape(1, *grad.shape),padding)[0]
        else:
            unpadded_grad = remove_padding(grad, padding)

        X.backward(unpadded_grad)

class ZeroPad2d():
    def __init__(self, padding):
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.padding = (self.padding[0], self.padding[0], self.padding[1], self.padding[1]) if len(self.padding) == 2 else self.padding

    def forward(self, X):
        assert X.ndim == 3 or X.ndim == 4, "X must be 3D or 4D tensor"
        X.data = X.data
        
        if X.data.ndim == 3:
            padded_data = set_padding(X.data.reshape(1, *X.data.shape), self.padding)[0]
        else:
            padded_data = set_padding(X.data, self.padding)

        return ZeroPad2dTensor(padded_data, [X, self.padding], "zeropad2d")

    def __call__(self, X):
        return self.forward(X)




def set_padding(layer, padding):
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

def remove_padding(layer, padding):
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