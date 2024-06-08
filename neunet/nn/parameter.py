import cupy as cp
import numpy as np

from neunet.autograd import Tensor

# class Parameter(Tensor):
#     def __new__(cls, data: Tensor = None, requires_grad = True):
#         assert isinstance(data, Tensor), "Data must be a tensor"

#         t = data.detach()
#         t.requires_grad = requires_grad

#         return t


class Parameter(Tensor):
    def __init__(self, data: Tensor, requires_grad=True):
        if not isinstance(data, Tensor):
            raise TypeError("Data must be a tensor")

        super().__init__(
            data=data.data,
            requires_grad=requires_grad,
            device=data.device,
            dtype=data.dtype,
        )

    def to(self, device):
        if device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'")
        if device == "cpu":
            xp = np
        else:
            xp = cp

        data = (
            xp.array(self.data)
            if isinstance(self.data, np.ndarray)
            else xp.array(self.data.get(), dtype=self.data.dtype)
        )

        return Parameter(
            Tensor(
                data,
                dtype=self.data.dtype,
                device=device,
            ),
            requires_grad=self.requires_grad
        )
