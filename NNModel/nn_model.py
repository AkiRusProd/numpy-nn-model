import numpy as np


class Model():

    def __init__(self):
        pass

    def compile(self,): pass

    def forward_prop(self, input):
        pass

    def backward_prop(self, error):
        pass

    def fit(self, inputs, targets, batch_size, epochs, loss):
        inputs = np.asarray(inputs)
        batch_num = len(inputs) // batch_size

        batches = np.array_split(inputs, batch_num)
        batches_targets = np.array_split(targets, batch_num)

        for i in range(epochs):
            for j in range(batch_num):
                pass
        

    def predict(self, X, Y,):
        pass