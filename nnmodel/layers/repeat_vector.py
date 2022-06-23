import numpy as np


class RepeatVector():

    def __init__(self, num) -> None:
        self.num = num
        self.input_shape = None

    def build(self):
        self.output_shape = (self.num, *self.input_shape[1 :])

    def forward_prop(self, X, training):
        self.input_data = X
        if len(self.input_data.shape) == 2:
            self.input_data = self.input_data[:, np.newaxis, :]
    
        if len(self.input_data.shape) == 3:
            return np.tile(self.input_data, (self.num, 1))

        return np.tile(self.input_data, (self.num,1, 1))

    def backward_prop(self, error):
        output_error = np.array(np.split(error, self.num, axis=1)).sum(axis=0)

        if self.input_data.shape[1] == 1:
            output_error = np.squeeze(output_error, axis = 1)
        
        return output_error
        