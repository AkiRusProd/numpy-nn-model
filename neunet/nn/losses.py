from neunet.autograd import Tensor
import numpy as np


class MSELoss(Tensor):
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        return (y_pred.sub(y_true)).power(2).sum().div(np.prod(y_pred.data.shape))

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)



class BCELoss(Tensor):
    def __init__(self, reduction = "mean"):
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        loss = y_true.mul(y_pred.log()).add((Tensor(1) - y_true).mul((Tensor(1) - y_pred).log()))
        if self.reduction == "mean":
            return loss.mul(-1).mean()
        elif self.reduction == "sum":
            return loss.mul(-1).sum()
        else:
            return loss.mul(-1)

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)



