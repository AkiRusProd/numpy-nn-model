import cupy as cp
import numpy as np

from neunet.autograd import Tensor


class Module:
    def __init__(self):
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        params = []
        for _, item in self.__dict__.items():
            if isinstance(item, Tensor):
                if (
                    item.requires_grad
                    and item.__class__.__name__ == "Parameter"
                    and item not in params
                ):
                    params.append(item)
            if hasattr(item, "parameters"):
                params.extend(item.parameters())

        return params

    def eval(self):
        self.training = False
        for _, item in self.__dict__.items():
            if hasattr(item, "eval"):
                item.eval()

    def train(self, mode=True):
        self.training = mode
        for _, item in self.__dict__.items():
            if hasattr(item, "train"):
                item.train(mode)

    def to(self, device):
        if device == "cpu":
            self.xp = np
        else:
            self.xp = cp

        self.device = device
        for name, item in self.__dict__.items():
            if hasattr(item, "to"):
                self.__dict__[name] = item.to(device)

        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")


class Sequential:
    def __init__(self, *modules: Module):
        self.modules = list(modules)
        self.training = True

    def forward(self, X):
        for module in self.modules:
            X = module(X)
        return X

    def __call__(self, X):
        return self.forward(X)

    def parameters(self):
        params = []
        for module in self.modules:
            if hasattr(module, "parameters"):
                params.extend(module.parameters())

        return params

    def eval(self):
        self.training = False
        for module in self.modules:
            if hasattr(module, "eval"):
                module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.modules:
            if hasattr(module, "train"):
                module.train(mode)

    def to(self, device):
        for i, module in enumerate(self.modules):
            if hasattr(module, "to"):
                self.modules[i] = module.to(device)

        return self


class ModuleList:
    def __init__(self, modules: list[Module]):
        self.modules = list(modules)

    def __getitem__(self, index):
        return self.modules[index]

    def __setitem__(self, index, module):
        self.modules[index] = module

    def __delitem__(self, index):
        del self.modules[index]

    def __len__(self):
        return len(self.modules)

    def append(self, module):
        self.modules.append(module)

    def extend(self, modules):
        self.modules.extend(modules)

    def insert(self, index, module):
        self.modules.insert(index, module)

    def forward(self, X):
        for module in self.modules:
            X = module(X)
        return X

    def __call__(self, X):
        return self.forward(X)

    def parameters(self):
        params = []
        for module in self.modules:
            if hasattr(module, "parameters"):
                params.extend(module.parameters())

        return params

    def eval(self):
        for module in self.modules:
            if hasattr(module, "eval"):
                module.eval()

    def train(self, mode=True):
        for module in self.modules:
            if hasattr(module, "train"):
                module.train(mode)

    def to(self, device):
        for i, module in enumerate(self.modules):
            if hasattr(module, "to"):
                self.modules[i] = module.to(device)
        return self
