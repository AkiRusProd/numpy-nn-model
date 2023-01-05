from autograd import Tensor
import numpy as np

class Module:
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        params = []
        for name, item in self.__dict__.items():
            if hasattr(item, 'parameters'):
                params.extend(item.parameters())
            # elif isinstance(item, Tensor):
            #     params.append(item)


        return params


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, X, training = True):
        for layer in self.layers:
            X = layer(X, training)
        return X

    def __call__(self, X, training = True):
        return self.forward(X, training)

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "weight"):
                if layer.weight.requires_grad:
                    params.append(layer.weight)
            if hasattr(layer, "bias"):
                if layer.bias.requires_grad:
                    params.append(layer.bias)
        return params
