import numpy as np

import neunet as nnet
from neunet.autograd import Tensor
from neunet.nn.activations import LogSoftmax
from neunet.nn.modules import Module


class MSELoss(Module):
    def __init__(self):
        pass

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        if not isinstance(y_pred, Tensor) or not isinstance(y_true, Tensor):
            raise TypeError("Input values must be tensors")
        if y_pred.device != y_true.device:
            raise ValueError("Tensors must be on the same device")

        return (y_pred.sub(y_true)).power(2).sum().div(np.prod(y_pred.data.shape))

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class BCELoss(Module):
    def __init__(self, weight=None, reduction="mean"):
        self.weight = weight
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        if not isinstance(y_pred, Tensor) or not isinstance(y_true, Tensor):
            raise TypeError("Input values must be tensors")
        if y_pred.device != y_true.device:
            raise ValueError("Tensors must be on the same device")

        loss = y_true.mul(y_pred.log()).add((1.0 - y_true).mul((1.0 - y_pred).log()))

        if self.weight is None:
            self.weight = y_pred.xp.ones((1))

        if (self.weight * y_pred.data).shape != y_pred.data.shape:
            raise ValueError(
                "Product shape of multiplication weight and y_pred must be equal to y_pred shape"
            )

        loss = loss.mul(self.weight)

        if self.reduction == "mean":
            return loss.mul(-1).mean()
        elif self.reduction == "sum":
            return loss.mul(-1).sum()
        else:
            return loss.mul(-1)

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class CrossEntropyLoss(Module):
    def __init__(self, weight=None, ignore_index=-100, reduction="mean"):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

        # self.softmax = Softmax(axis=1)
        self.log_softmax = LogSoftmax(axis=1)
        self.nll_loss = NLLLoss(weight, ignore_index, reduction)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        if not isinstance(y_pred, Tensor) or not isinstance(y_true, Tensor):
            raise TypeError("Input values must be tensors")
        if y_pred.device != y_true.device:
            raise ValueError("Tensors must be on the same device")

        # y_pred = self.softmax(y_pred).log()
        y_pred = self.log_softmax(y_pred)
        return self.nll_loss(y_pred, y_true)

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class NLLLoss(Module):
    def __init__(self, weight=None, ignore_index=-100, reduction="mean"):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        if not isinstance(y_pred, Tensor) or not isinstance(y_true, Tensor):
            raise TypeError("Input values must be tensors")
        if y_pred.device != y_true.device:
            raise ValueError("Tensors must be on the same device")

        if self.weight is None:
            self.weight = y_pred.xp.ones((y_pred.data.shape[1]), dtype=y_pred.dtype)

        if self.weight.shape != (y_pred.data.shape[1],):
            raise ValueError("Weight shape must be equal to number of classes")
        if y_true.dtype not in (np.int16, np.int32, np.int64):
            raise TypeError("Target must be of int dtype")

        if y_pred.data.ndim == 2:
            y_pred = y_pred[..., None]
        if y_true.data.ndim == 1:
            y_true = y_true[..., None]

        # TODO: if neg value in y_true != ignore_index, raise error, fix, negative ids in weight

        ignore_mask = y_true.data != self.ignore_index

        idx = np.indices(y_true.data.shape, sparse=True)
        criterion = (idx[0], y_true.data, *idx[1:])
        # criterion = (self.xp.arange(y_true.data.shape[0]), y_true.data.flatten())
        loss = -y_pred[criterion] * self.weight[y_true.data] * ignore_mask

        if self.reduction == "mean":
            return nnet.sum(loss / nnet.sum(self.weight[y_true.data] * ignore_mask))

        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class L1Loss(Module):
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        if not isinstance(y_pred, Tensor) or not isinstance(y_true, Tensor):
            raise TypeError("Input values must be tensors")
        if y_pred.device != y_true.device:
            raise ValueError("Tensors must be on the same device")

        loss = y_pred.sub(y_true).abs()

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class KLDivLoss(Module):
    def __init__(self, reduction="mean", log_target=False):
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        if not isinstance(y_pred, Tensor) or not isinstance(y_true, Tensor):
            raise TypeError("Input values must be tensors")
        if y_pred.device != y_true.device:
            raise ValueError("Tensors must be on the same device")

        if not self.log_target:
            loss = y_true.mul(y_true.log().sub(y_pred))
        else:
            loss = y_true.exp().mul(y_true.sub(y_pred))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "batchmean":
            return loss.sum().div(y_pred.data.shape[0])
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
