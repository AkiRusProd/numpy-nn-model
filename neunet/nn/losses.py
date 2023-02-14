from neunet.autograd import Tensor
from neunet.nn.activations import Softmax
import neunet as nnet
import numpy as np


class MSELoss(Tensor):
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        return (y_pred.sub(y_true)).power(2).sum().div(np.prod(y_pred.data.shape))

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)



class BCELoss(Tensor):
    def __init__(self, weight = None, reduction = "mean"):
        self.weight = weight
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        loss = y_true.mul(y_pred.log()).add((Tensor(1) - y_true).mul((Tensor(1) - y_pred).log()))

        if self.weight is None:
            self.weight = np.ones((1))

        assert (self.weight * y_pred.data).shape == y_pred.data.shape, "Product shape of multiplication weight and y_pred must be equal to y_pred shape"

        loss = loss.mul(self.weight)

        if self.reduction == "mean":
            return loss.mul(-1).mean()
        elif self.reduction == "sum":
            return loss.mul(-1).sum()
        else:
            return loss.mul(-1)

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)




class CrossEntropyLoss(Tensor):
    def __init__(self, weight = None, ignore_index = -100, reduction = "mean"):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.softmax = Softmax(axis=1)
        self.nll_loss = NLLLoss(weight, ignore_index, reduction)

    def forward(self, y_pred, y_true):
        y_pred = self.softmax(y_pred).log()
        return self.nll_loss(y_pred, y_true)

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class NLLLoss(Tensor):
    def __init__(self, weight = None, ignore_index = -100, reduction = "mean"):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        if self.weight is None:
            self.weight = np.ones((y_pred.data.shape[1]))

        assert self.weight.shape == (y_pred.data.shape[1], ), "Weight shape must be equal to number of classes"

        if y_pred.data.ndim == 2:
            y_pred = y_pred[..., None]
        if y_true.data.ndim == 1:
            y_true = y_true[..., None]

        ignore_mask = y_true.data != self.ignore_index

        idx = np.indices(y_true.data.shape, sparse=True)
        criterion = (idx[0], y_true.data, *idx[1:])
        loss = -y_pred[criterion] * self.weight[y_true.data] * ignore_mask

        if self.reduction == "mean":
            return nnet.sum(loss / nnet.sum(self.weight[y_true.data] * ignore_mask))
            
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)



class L1Loss(Tensor):
    def __init__(self, reduction = "mean"):
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        loss = y_pred.sub(y_true).abs()

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)