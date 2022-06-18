

class ErrorHandler(Exception):

    
    class InvalidActivationType(Exception):

            def __init__(self, activation_type):
                self.activation_type = activation_type
                super().__init__(self.activation_type)

            def __str__(self):
                return f'Activation must be a string, None or Class of Activations, but not {self.activation_type} '

    class InvalidActivationName(Exception):

            def __init__(self, activation_name):
                self.activation_name = activation_name
                super().__init__(self.activation_name)

            def __str__(self):
                return f'Activation with this name {self.activation_name} does no exist'

    class InvalidSize2Variable(Exception):

            def __init__(self, variable_type, variable_name):
                self.variable_type = variable_type
                self.variable_name = variable_name
                super().__init__(self.variable_type, self.variable_name)

            def __str__(self):
                add_message = ""
                if self.variable_name == "padding":
                    add_message = ', or string: "same"/"valid"'

                return f' Unable to extract {self.variable_name} values from type {self.variable_type}; {self.variable_name} must be integer value, or list/tuple type of size 2 and contain integer values {add_message}'
    
    class InvalidIntegerValue(Exception):
            def __init__(self, variable_type, variable_name):
                self.variable_name = variable_name
                self.variable_type = variable_type
                super().__init__(self.variable_type, self.variable_name)

            def __str__(self):
                return f'Type of the variable {self.variable_name} must be integer, got {self.variable_type}'

    class InvalidFloatValue(Exception):
            def __init__(self, variable_type, variable_name):
                self.variable_name = variable_name
                self.variable_type = variable_type
                super().__init__(self.variable_type, self.variable_name)

            def __str__(self):
                return f'Type of the variable {self.variable_name} must be float, got {self.variable_type}'

    class InvalidInputDim(Exception):
            def __init__(self, variable_type, input_dim):
                self.input_dim = input_dim
                self.variable_type = variable_type
                super().__init__(self.variable_type, self.input_dim)

            def __str__(self):
                return f'''Unable to extract "input_shape" values from type {self.variable_type};\nPossibe cases:\n "input_shape" must be integer value or list/tuple type of size {self.input_dim} that contain integer values\nExample:\n for "Dense" layer "input_shape" only can be of two types: "123" or "(1, 123)"\n for "Conv2D" and some others layer "input_shape" must be only list/tuple type of size 3: (Channels, Height, Width), and contain integer value\n'''
    
    class InvalidShape(Exception):
            def __init__(self, variable_type):
                self.variable_type = variable_type
                super().__init__(self.variable_type)

            def __str__(self):
                return f'''Unable to extract "shape" values from type {self.variable_type};\nPossibe cases:\n "input_shape" must be integer value or list/tuple type that contain integer values'''
