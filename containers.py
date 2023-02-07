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
            if hasattr(item, "parameters"):
                params.extend(item.parameters())
            if hasattr(item, "weight"):
                if hasattr(item.weight, "requires_grad"):
                    if item.weight.requires_grad:
                        params.append(item.weight)
            if hasattr(item, "bias"):
                if hasattr(item.bias, "requires_grad"):
                    if item.bias.requires_grad:
                        params.append(item.bias)
            if hasattr(item, "named_parameters"):
                for name, param in item.named_parameters():
                    if param.requires_grad:
                        if param not in params:
                            params.append(param)


        return params


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def __call__(self, X):
        return self.forward(X)

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
            if hasattr(layer, "weight"):
                if hasattr(layer.weight, "requires_grad"):
                    if layer.weight.requires_grad:
                        params.append(layer.weight)
            if hasattr(layer, "bias"):
                if hasattr(layer.bias, "requires_grad"):
                    if layer.bias.requires_grad:
                        params.append(layer.bias)
            if hasattr(layer, "named_parameters"):
                for name, param in layer.named_parameters():
                    if param.requires_grad:
                        if param not in params:
                            params.append(param)
        return params
