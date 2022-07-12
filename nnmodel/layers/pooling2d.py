import numpy as np
from numba import njit
from nnmodel.exceptions.values_checker import ValuesChecker

class Pooling2D():
    
    def __init__(self,  pool_size=(2, 2), input_shape = None, padding = (0, 0), stride = None, dilation = (1, 1)):
        self.pool_size   = ValuesChecker.check_size2_variable(pool_size, variable_name = "pool_size", min_acceptable_value = 1)
        self.input_shape = ValuesChecker.check_input_dim(input_shape, input_dim = 3)
        self.padding      = ValuesChecker.check_size2_variable(padding, variable_name = "padding", min_acceptable_value = 0)
        if stride == None: stride = self.pool_size
        self.stride = ValuesChecker.check_size2_variable(stride, variable_name = "pool_stride", min_acceptable_value = 1)
        self.dilation     = ValuesChecker.check_size2_variable(dilation, variable_name = "dilation", min_acceptable_value = 1)


    def build(self):

        self.pool_height, self.pool_width = self.pool_size
        self.channels_num, self.input_height, self.input_width = self.input_shape

        if self.padding == "valid":
            self.padding == (0, 0, 0, 0)
        elif self.padding == "same" or self.padding == "real same":
            if self.padding == "same": #keras "same" implementation, that returns the output of size "input_size + stride_size"
                padding_up_down = self.dilation[0] * (self.pool_height - 1) - self.stride[0] + 1 
                padding_left_right = self.dilation[1] * (self.pool_width  - 1) - self.stride[1] + 1
            elif self.padding == "real same": # my "same" implementation, that returns the output of size "input_size"
                padding_up_down = (self.stride[0] - 1) * (self.input_height - 1) + self.dilation[0] * (self.pool_height - 1)
                padding_left_right = (self.stride[1] - 1) * (self.input_width- 1) + self.dilation[1] * (self.pool_width  - 1)

            if padding_up_down % 2 == 0:
                padding_up, padding_down = padding_up_down // 2, padding_up_down // 2
            else:
                padding_up, padding_down = padding_up_down // 2, padding_up_down - padding_up_down // 2

            if padding_left_right % 2 == 0:
                padding_left, padding_right = padding_left_right // 2, padding_left_right // 2
            else:
                padding_left, padding_right = padding_left_right // 2, padding_left_right - padding_left_right // 2
    

            self.padding = (abs(padding_up), abs(padding_down), abs(padding_left), abs(padding_right))

        elif len(self.padding) == 2:
            self.padding = (self.padding[0], self.padding[0], self.padding[1], self.padding[1]) #(up, down, left, right) padding ≃ (2 * vertical, 2 *horizontal) padding


        # self.output_size = (self.input_shape[1] - self.pool_size[0]) // self.stride[0] + 1, (self.input_shape[2] - self.pool_size[1]) // self.stride[1] + 1
        self.output_height = (self.input_shape[1] + self.padding[0] + self.padding[1] - self.dilation[0] * (self.pool_size[0] - 1) - 1) // self.stride[0] + 1
        self.output_width =  (self.input_shape[2] + self.padding[2] + self.padding[3] - self.dilation[1] * (self.pool_size[1] - 1) - 1) // self.stride[1] + 1
        # self.output_shape = (self.input_shape[0], self.output_height, self.output_width)

        self.dilated_pool_height =  self.dilation[0] * (self.pool_height - 1) + 1
        self.dilated_pool_width = self.dilation[1] * (self.pool_width - 1) + 1

        # self.stride_compared_input_height = (self.output_height - 1) * self.stride[0] - self.padding[0] - self.padding[1] +  self.dilated_pool_height
        # self.stride_compared_input_width = (self.output_width - 1) * self.stride[1] - self.padding[2] - self.padding[3] +  self.dilated_pool_width
    
        # self.prepared_input_height = (self.stride_compared_input_height + self.padding[0] + self.padding[1])
        # self.prepared_input_width = (self.stride_compared_input_width + self.padding[2] + self.padding[3])

        self.pool_part_area = np.ones((self.pool_height, self.pool_width))
        self.pool_part_area = self.set_dilation_stride(self.pool_part_area, self.dilation)


        self.output_shape = (self.channels_num, self.output_height, self.output_width)
        

        
        


    @staticmethod
    @njit
    def pooling(conv_layer, pool_part_area, input_shape, pool_height, pool_width, output_height, output_width, stride, pooling_type):
        
        pooling_layer = np.zeros((input_shape[0], input_shape[1], output_height, output_width))
        pooling_layer_ind = np.zeros((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))

        for b in range(input_shape[0]):
            for s in range(input_shape[1]):
                for h in range(output_height):
                    for w in range(output_width):

                        pool_part = conv_layer[
                            b,
                            s,
                            h * stride[0] : h * stride[0] + pool_height, 
                            w * stride[1] : w * stride[1] + pool_width,
                        ] * pool_part_area

                        if pooling_type == "MaxPooling":
                            pooling_layer[b, s, h, w] = pool_part.max()
                        elif pooling_type == "AvgPooling":
                            pooling_layer[b, s, h, w] = pool_part.mean()
                        # pooling_layer[b, s, h, w] = pooling_type(pool_part)

                        for i in range(pool_height):
                            for j in range(pool_width):
                                if pool_part[i, j] == pooling_layer[b, s, h, w]:
                                    I = int(i + h * stride[0])

                                    J = int(j + w * stride[1])

                        pooling_layer_ind[b, s, I, J] = 1

        return pooling_layer, pooling_layer_ind

    @staticmethod
    @njit
    def pooling_error_backward_prop(
        pooling_layer_ind, error, input_shape
    ):  
        
        error = error.reshape((error.shape[0], error.shape[1], error.shape[2] * error.shape[3]))

        output_layer_error = np.zeros((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))

        for b in range(input_shape[0]):
            for s in range(input_shape[1]):
                i = 0
                for h in range(input_shape[2]):
                    for w in range(input_shape[3]):
                        if pooling_layer_ind[b, s, h, w] == 1:
                            output_layer_error[b, s, h, w] = error[b, s, i]
                            i += 1

        return output_layer_error


    
    @staticmethod
    @njit
    def set_dilation_stride(layer, stride):
        
        transposed_layer = np.zeros(
            (
                stride[0] * layer.shape[0] - (stride[0] - 1),
                stride[1] * layer.shape[1] - (stride[1] - 1),
            ),
            dtype=layer.dtype,
        )
        
        transposed_layer[::stride[0], ::stride[1]] = layer

        return transposed_layer


    @staticmethod
    @njit
    def set_padding(layer, padding):
        # padded_layer = np.pad(layer, ((0, 0), (0, 0), (padding[0], padding[1]), (padding[1], padding[0])), constant_values = 0)
        padded_layer = np.zeros(
            (   
                layer.shape[0],
                layer.shape[1],
                layer.shape[2] + padding[0] + padding[1],
                layer.shape[3] + padding[2] + padding[3],
            )
        )

        padded_layer[
                    :,
                    :,
                    padding[0] :padded_layer.shape[2] - padding[1],
                    padding[2] :padded_layer.shape[3] - padding[3],
                ] = layer

        return padded_layer

    @staticmethod
    @njit
    def remove_padding(layer, padding):
        # unpadded_layer = unpadded_layer[:, :, padding[0]:-padding[1], padding[1]:-padding[0]]
        unpadded_layer = np.zeros(
            (
                layer.shape[0],
                layer.shape[1],
                layer.shape[2] - padding[0] - padding[1],
                layer.shape[3] - padding[2] - padding[3],
            )
        )
        
        unpadded_layer = layer[
                    :,
                    :,
                    padding[0] :layer.shape[2] - padding[1],
                    padding[2] :layer.shape[3] - padding[3],
                ]

        return unpadded_layer


class MaxPooling2D(Pooling2D):
    """
    Applies MaxPooling to the 2d input data
    ---------------------------------------
        Args:
            `pool_size` (tuple), (list) of size 2 or (int): the size of the sliding pooling window
            `padding` (tuple), (list) of size 2 or (int) or `"same"`, `"real same"`, `"valid"` string value: the padding of the input window
            
            {
                `"valid"`: padding is 0
                `"same"`: keras "same" implementation, that returns the output of size "input_size + stride_size"
                `"real same"`: my "same" implementation, that returns the output of size "input_size"
            }
            `stride` (tuple), (list) of size 2 or (int): the stride of the sliding pooling window
            `dilation` (tuple), (list) of size 2 or (int): the dilation of the sliding pooling window

        Returns:
            output: output_layer (numpy.ndarray): the output layer of the MaxPooling with shape: (batch_size, channels_num, output_height, output_width)
    """

    def __init__(self,  pool_size=(2, 2), input_shape = None, padding = (0, 0), stride = None, dilation = (1, 1)):
        Pooling2D.__init__(self, pool_size, input_shape, padding, stride, dilation)

    def forward_prop(self, X, training):
        self.input_data = self.set_padding(X, self.padding)
       
        self.batch_size = len(self.input_data)

        self.pooling_layer, self.pooling_layer_ind = self.pooling(self.input_data, self.pool_part_area, self.input_data.shape,  self.dilated_pool_height, self.dilated_pool_width, self.output_height, self.output_width, self.stride, "MaxPooling")

        return self.pooling_layer

    def backward_prop(self, error):
        error = error.copy()

        output_error = self.pooling_error_backward_prop(self.pooling_layer_ind, error, self.input_data.shape)

        return self.remove_padding(output_error, self.padding)



class AveragePooling2D(Pooling2D):
    """
    Applies AveragePooling to the 2d input data
    -------------------------------------------
        Args:
            `pool_size` (tuple), (list) of size 2 or (int): the size of the sliding pooling window
            `padding` (tuple), (list) of size 2 or (int) or `"same"`, `"real same"`, `"valid"` string value: the padding of the input window
            
            {
                `"valid"`: padding is 0
                `"same"`: keras "same" implementation, that returns the output of size "input_size + stride_size"
                `"real same"`: my "same" implementation, that returns the output of size "input_size"
            }
            `stride` (tuple), (list) of size 2 or (int): the stride of the sliding pooling window
            `dilation` (tuple), (list) of size 2 or (int): the dilation of the sliding pooling window

        Returns:
            output: output_layer (numpy.ndarray): the output layer of the MaxPooling with shape: (batch_size, channels_num, output_height, output_width)
    """


    def __init__(self,  pool_size=(2, 2), input_shape = None, padding = (0, 0), stride = None, dilation = (1, 1)):
        Pooling2D.__init__(self, pool_size, input_shape, padding, stride, dilation)

    def forward_prop(self, X, training):
        self.input_data = self.set_padding(X, self.padding)
       
        self.batch_size = len(self.input_data)

        self.pooling_layer, self.pooling_layer_ind = self.pooling(self.input_data, self.pool_part_area, self.input_data.shape,  self.dilated_pool_height, self.dilated_pool_width, self.output_height, self.output_width, self.stride, "AvgPooling")

        return self.pooling_layer

    def backward_prop(self, error):
        error = error.copy()

        output_error = self.pooling_error_backward_prop(self.pooling_layer_ind, error, self.input_data.shape)

        return self.remove_padding(output_error, self.padding)




