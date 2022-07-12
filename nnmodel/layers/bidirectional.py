import numpy as np
import copy as copy_object

class Bidirectional():
    """
    Applies the bidirectional architecture to the recurrent layers
    --------------------------------------------------------------
        Args:
            `layer` : the recurrent layer
            `merge_mode` : the mode of the merge of the forward outputs:
                `concatenate` : concatenate the forward outputs
                `sum` : sum the forward outputs
                `multiply` : multiply the forward outputs
                `average` : average the forward outputs
        Returns:
            output: the recurrent layer output according to the merge_mode
    """
    #TODO
    #add use access only for RNN, LSTM, GRU layers
    def __init__(self, layer, merge_mode = 'concatenate'):
        self.layer = layer
        self.direct_layer = copy_object.copy(layer)
        self.reverse_layer = copy_object.copy(layer)

        self.merge_mode = merge_mode
        self.input_shape = None

        self.optimizer = None


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

        if hasattr(self.layer, 'set_optimizer'):
            self.direct_layer.set_optimizer(self.optimizer)
            self.reverse_layer.set_optimizer(self.optimizer)

    def concatenate_mode(self, first_output, second_output):
        return  np.concatenate((first_output, second_output), axis=-1)
    def sum_mode(self, first_output, second_output):
        return first_output + second_output
    def multiply_mode(self, first_output, second_output):
        return first_output * second_output
    def average_mode(self, first_output, second_output):
        return (first_output + second_output) / 2


    def build(self):
        

        if self.merge_mode == "concatenate":
            self.merge_outputs = self.concatenate_mode
        elif self.merge_mode == "sum":
            self.merge_outputs = self.sum_mode
        elif self.merge_mode == "multiply":
            self.merge_outputs = self.multiply_mode
        elif self.merge_mode == "average":
            self.merge_outputs = self.average_mode


        self.direct_layer.input_shape = self.input_shape
        self.reverse_layer.input_shape = self.input_shape

        self.direct_layer.build()
        self.reverse_layer.build()

        if self.merge_mode != "concatenate":
            self.output_shape = self.direct_layer.output_shape
        else:
            self.output_shape = (self.direct_layer.output_shape[0], self.direct_layer.output_shape[1] * 2)

        
    def forward_prop(self, X, training):
        self.direct_input_data = X
        self.reverse_input_data = X[:, ::-1, ...]

        self.direct_forward_data = self.direct_layer.forward_prop(self.direct_input_data, training)
        self.reverse_forward_data = self.reverse_layer.forward_prop(self.reverse_input_data, training)

        return self.merge_outputs(self.direct_forward_data, self.reverse_forward_data)


    def backward_prop(self, error):

        if self.merge_mode == "concatenate":
            direct_layer_error, reverse_layer_error = np.split(error, 2, axis=-1)
        else:
            direct_layer_error = reverse_layer_error = error

        self.direct_backward_error = self.direct_layer.backward_prop(direct_layer_error)
        self.reverse_backward_error = self.reverse_layer.backward_prop(reverse_layer_error)

        return self.direct_backward_error + self.reverse_backward_error
        

    def update_weights(self, layer_num):
        self.direct_layer.update_weights(layer_num)
        self.reverse_layer.update_weights(layer_num)

    def get_grads(self):
        
        return self.direct_layer.get_grads() , self.reverse_layer.get_grads()

    def set_grads(self, grads):
        
        self.direct_layer.set_grads(grads[0])
        self.reverse_layer.set_grads(grads[1])