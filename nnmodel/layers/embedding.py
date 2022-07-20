import numpy as np
from nnmodel.exceptions.values_checker import ValuesChecker

class Embedding():
    """
    Add Embedding layer
    ---------------
        Args:
            `input_dim`: (int), size of vocabulary
            `output_dim` (int): number of neurons in the layer (vector size)
            `input_length` (int): length of the input sequence,  (without it, the shape of the dense outputs cannot be computed)
        Returns:
            input: data with shape (batch_size, input_length)
            output: data with shape (batch_size, input_length, output_dim)
    """

    def __init__(self, input_dim, output_dim, input_length = None):
        self.input_dim = ValuesChecker.check_integer_variable(input_dim, "input_dim")
        self.output_dim   = ValuesChecker.check_integer_variable(output_dim, "output_dim")
        self.input_length = ValuesChecker.check_integer_variable(input_length, "input_length")

        self.w = None

        self.optimizer = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def build(self):
        
        self.w = np.random.normal(0, pow(self.input_dim, -0.5), (self.input_dim, self.output_dim))

        self.v, self.m         = np.zeros_like(self.w), np.zeros_like(self.w)
        self.v_hat, self.m_hat = np.zeros_like(self.w), np.zeros_like(self.w)

        if self.input_length is not None:
            self.output_shape = (self.input_length, self.output_dim)
        else:
            print("Input_length is not set, can`t compute output_shape of the Embedding layer")

    #one hot encoding
    def prepare_labels(self, batch_labels):
        prepared_batch_labels = []
        
        for sequence in batch_labels:
            for label in sequence:
                correct_label = int(label)

                labels_list = np.zeros(self.input_dim)

                labels_list[correct_label] = 1

                prepared_batch_labels.append(labels_list)

        
        return np.asarray(prepared_batch_labels).reshape(self.batch_size, self.current_input_length, self.input_dim)


    def forward_prop(self, X, training):
        # print(X)
        self.input_data = X # (batch_size, input_length); inputs: values of vocabulary from 0 to input_dim - 1
        
        if not all([np.array_equal(len(self.input_data[0]), len(arr)) for arr in self.input_data]):
            raise ValueError("Input sequences must be of the same length")

        self.current_input_length = len(self.input_data[0])
        self.batch_size = len(self.input_data)

        self.input_data = self.prepare_labels(self.input_data)

        self.output_data = np.dot(self.input_data, self.w)
        
        return self.output_data

    def backward_prop(self, error):        
        self.grad_w = np.dot(np.transpose(self.input_data, axes = (0, 2, 1)), error).sum(axis = 0).sum(axis = 1).reshape(self.w.shape)

        # output_error = np.dot(error, self.w.T)

        # return output_error
        return None

    def update_weights(self, layer_num):
        self.w, self.v, self.m, self.v_hat, self.m_hat  = self.optimizer.update(self.grad_w, self.w, self.v, self.m, self.v_hat, self.m_hat, layer_num)
    

    def get_grads(self):
        return self.grad_w, self.grad_b

    def set_grads(self, grads):
        self.grad_w, self.grad_b = grads
        

