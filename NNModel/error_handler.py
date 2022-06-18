

class ErrorHandler(Exception):

    
    class WrongActivationType(Exception):

            def __init__(self, activation_type):
                self.activation_type = activation_type
                super().__init__(self.activation_type)

            def __str__(self):
                return f'Activation must be a string, None or Class of Activations, but not {self.activation_type} '

    class WrongActivationName(Exception):

            def __init__(self, activation_name):
                self.activation_name = activation_name
                super().__init__(self.activation_name)

            def __str__(self):
                return f'Activation with this name {self.activation_name} does no exist'

    class WrongSize2Variable(Exception):

            def __init__(self, variable_type, variable_name):
                self.variable_type = variable_type
                self.variable_name = variable_name
                super().__init__(self.variable_type, self.variable_name)

            def __str__(self):
                add_message = ""
                if self.variable_name == "padding":
                    add_message = 'or string: "same"/"valid"'

                return f' Unable to extract {self.variable_name} values from type {self.variable_type}; {self.variable_name} must be a int or int list/tuple/np.array of size 2 {add_message}'
    
    class WrongIntegerValue(Exception):
            def __init__(self, variable_type, variable_name):
                self.variable_name = variable_name
                self.variable_type = variable_type
                super().__init__(self.variable_type, self.variable_name)

            def __str__(self):
                return f'Type of the variable {self.variable_name} must be integer, got {self.variable_type}'
    
