from collections import OrderedDict
from typing import Literal

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
        seen_ids = set()
        for _, item in self.__dict__.items():
            if isinstance(item, Tensor):
                item_id = id(item)
                if (
                    item.requires_grad
                    and item.__class__.__name__ == "Parameter"
                    and item_id not in seen_ids
                ):
                    params.append(item)
                    seen_ids.add(item_id)
            if hasattr(item, "parameters"):
                params.extend(item.parameters())

        return params

    def eval(self):
        self.training = False
        for _, item in self.__dict__.items():
            if hasattr(item, "eval"):
                item.eval()

    def train(self, mode: bool=True):
        self.training = mode
        for _, item in self.__dict__.items():
            if hasattr(item, "train"):
                item.train(mode)

    def to(self, device: Literal["cpu", "cuda"]):
        if device == "cpu":
            self.xp = np
        else:
            self.xp = cp

        self.device = device
        # TODO: Preserve shared Tensor/Parameter references during device move.
        # Current behavior calls `.to(device)` per attribute and can break aliasing
        # (e.g. tied weights like `lm_head.weight = wte.weight` become different objects).
        # Fix idea: use memoization by `id(item)` and reuse already-moved tensors.
        for name, item in self.__dict__.items():
            if hasattr(item, "to"):
                self.__dict__[name] = item.to(device)

        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def state_dict(self):
        # NOTE: state_dict() also includes tensors with requires_grad = False (for example, bnorm running_mean and running_var); parameters() - are not!
        state_dict = OrderedDict()
        for name, item in self.__dict__.items():
            if isinstance(item, Tensor) and item.__class__.__name__ == "Parameter":
                state_dict[name] = item.data
            elif hasattr(item, "state_dict"):
                for key, value in item.state_dict().items():
                    state_dict[name + "." + key] = value

        return state_dict

    def load_state_dict(self, state_dict):
        for name, item in self.__dict__.items():
            if isinstance(item, Tensor) and item.__class__.__name__ == "Parameter":
                if name in state_dict:
                    # item.data = state_dict[name]
                    item.data = self._cast_data(state_dict[name], type(item.data))
            elif hasattr(item, "load_state_dict"):
                # Create a sub-dictionary for the submodule
                sub_dict = {k.split(".", 1)[1]: v for k, v in state_dict.items() if k.startswith(name + ".")}
                item.load_state_dict(sub_dict)

    def _cast_data(self, data, data_type):
        if isinstance(data, data_type):
            return data
        elif isinstance(data, np.ndarray):
            return cp.asarray(data)
        elif isinstance(data, cp.ndarray):
            return data.get()
        else:
            raise ValueError


class Sequential(Module):
    def __init__(self, *modules: Module):
        self.modules = list(modules)
        self.training = True

    def forward(self, X: Tensor) -> Tensor:
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

    def train(self, mode: bool=True):
        self.training = mode
        for module in self.modules:
            if hasattr(module, "train"):
                module.train(mode)

    def to(self, device: Literal["cpu", "cuda"]):
        for i, module in enumerate(self.modules):
            if hasattr(module, "to"):
                self.modules[i] = module.to(device)

        return self

    def state_dict(self):
        state_dict = OrderedDict()
        for i, module in enumerate(self.modules):
            if isinstance(module, Tensor) and module.__class__.__name__ == "Parameter":
                state_dict[str(i)] = module.data
            elif hasattr(module, "state_dict"):
                for key, value in module.state_dict().items():
                    state_dict[str(i) + "." + key] = value

        return state_dict

    def load_state_dict(self, state_dict):
        for i, module in enumerate(self.modules):
            if isinstance(module, Tensor) and module.__class__.__name__ == "Parameter":
                if str(i) in state_dict:
                    # module.data = state_dict[str(i)]
                    module.data = self._cast_data(state_dict[str(i)], type(module.data))
            elif hasattr(module, "load_state_dict"):
                # Create a sub-dictionary for the submodule
                sub_dict = {k.split(".", 1)[1]: v for k, v in state_dict.items() if k.startswith(str(i) + ".")}
                module.load_state_dict(sub_dict)


class ModuleList(Module):
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

    def forward(self, X: Tensor) -> Tensor:
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

    def train(self, mode: bool=True):
        for module in self.modules:
            if hasattr(module, "train"):
                module.train(mode)

    def to(self, device: Literal["cpu", "cuda"]):
        for i, module in enumerate(self.modules):
            if hasattr(module, "to"):
                self.modules[i] = module.to(device)
        return self

    def state_dict(self):
        state_dict = OrderedDict()
        for i, module in enumerate(self.modules):
            if isinstance(module, Tensor) and module.__class__.__name__ == "Parameter":
                state_dict[str(i)] = module.data
            elif hasattr(module, "state_dict"):
                for key, value in module.state_dict().items():
                    state_dict[str(i) + "." + key] = value

        return state_dict

    def load_state_dict(self, state_dict):
        for i, module in enumerate(self.modules):
            if isinstance(module, Tensor) and module.__class__.__name__ == "Parameter":
                if str(i) in state_dict:
                    # module.data = state_dict[str(i)]
                    module.data = self._cast_data(state_dict[str(i)], type(module.data))
            elif hasattr(module, "load_state_dict"):
                # Create a sub-dictionary for the submodule
                sub_dict = {k.split(".", 1)[1]: v for k, v in state_dict.items() if k.startswith(str(i) + ".")}
                module.load_state_dict(sub_dict)
