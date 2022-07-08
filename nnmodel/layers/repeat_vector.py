import numpy as np
from nnmodel.exceptions.values_checker import ValuesChecker

class RepeatVector():

    def __init__(self, num) -> None:
        self.num = ValuesChecker.check_integer_variable(num, "num")
        self.input_shape = None

    def build(self):
        if len(self.input_shape) == 2:
            self.output_shape = (self.num * self.input_shape[0], *self.input_shape[1 :])
        else:
            self.output_shape = (self.num, *self.input_shape)
        

    #temporary solution
    def forward_prop(self, X, training):
        self.input_data = X
        if len(self.input_data.shape) == 2: #if 1d layer with 1 timestep: (batchsize, units_num)
            self.input_data = self.input_data[:, np.newaxis, :]
    
        if len(self.input_data.shape) == 3:
            return np.tile(self.input_data, (self.num, 1)) #if 1d layer with many timesteps: (batchsize, timesteps, units_num)

        if len(self.input_data.shape) == 4: #if 2d layer with 1 timestep: (batchsize, channels, H, W)
            self.input_data = self.input_data[:, np.newaxis, ...]        

        if len(self.input_data.shape) == 5: #if 2d layer with many timesteps: (batchsize, timesteps, channels, H, W)
            return np.tile(self.input_data, (self.num, 1, 1, 1)) 
        


    def backward_prop(self, error):
        output_error = np.array(np.split(error, self.num, axis=1)).sum(axis=0)

        if self.input_data.shape[1] == 1:
            output_error = np.squeeze(output_error, axis = 1)
        
        return output_error
        