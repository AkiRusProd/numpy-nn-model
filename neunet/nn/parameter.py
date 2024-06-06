from neunet.autograd import Tensor
import numpy as np
import cupy as cp
# class Parameter(Tensor):
#     def __new__(cls, data: Tensor = None, requires_grad = True):
#         assert isinstance(data, Tensor), "Data must be a tensor"

#         t = data.detach()
#         t.requires_grad = requires_grad

#         return t


class Parameter(Tensor):
    def __init__(self, data: Tensor = None, requires_grad=True):
        assert isinstance(data, Tensor), "Data must be a tensor"

        super().__init__(
            data=data.data,
            requires_grad=requires_grad,
            device=data.device,
            dtype=data.dtype,
        )

    def to(self, device):
        assert device in ["cpu", "cuda"], "Device must be 'cpu' or 'cuda'"
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
                requires_grad=self.requires_grad,
                dtype=self.data.dtype,
                device=device,
            )
        )
