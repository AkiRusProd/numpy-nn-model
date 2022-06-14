import numpy as np


class Dropout():

    def __init__(self, rate = 0.1) -> None:
        self.rate = rate
        
    def forward_prop(self, X):
        self.mask = np.random.binomial(
                        n = 1,
                        p = 1 - self.rate,
                        size = X.shape,
                    )

        return X * self.mask

    def backward_prop(self, error):

        return error * self.mask