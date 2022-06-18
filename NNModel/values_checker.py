from nnmodel.error_handler import ErrorHandler

class ValuesChecker():

    def check_activation(activation_function, activations):

        if type(activation_function) is str or activation_function is None:
            try:
                activation = activations[activation_function]

            except: 
                raise ErrorHandler.InvalidActivationName(activation_function)
        else:
            try:
                if activation.__bases__.__class__.__name__ == "ActivationFunction":
                    activation = activation_function
            except: 
                
                raise ErrorHandler.InvalidActivationType(type(activation))

        return activation

    def check_size2_variable(variable, variable_name = None):
        if type(variable) is int:
            variable = (variable, variable)

            return variable
        elif type(variable) is str and variable_name  == "padding":
            if variable == "same" or variable == "valid":

                return variable

            else:
                raise ErrorHandler.InvalidSize2Variable(type(variable), variable_name)

        else: 
            if (type(variable) is list or type(variable) is tuple) and all(type(x) is int for x in variable) and len(variable) == 2:
                return variable
            else:
                raise ErrorHandler.InvalidSize2Variable(type(variable), variable_name)
            

    def check_integer_variable(variable, variable_name = None):
        if type(variable) is int:

            return variable
        else:
            raise ErrorHandler.InvalidIntegerValue(type(variable), variable_name)

    def check_float_variable(variable, variable_name = None):
        if type(variable) is float:

            return variable
        else:
            raise ErrorHandler.InvalidFloatValue(type(variable), variable_name)

    def check_input_dim(variable, input_dim):
        if variable == None or input_dim == None:
            return variable

        elif type(variable) is int and (input_dim == 2 or input_dim == 1):
            variable = (1, variable)

            return variable
        elif type(variable) is str:
           
            raise ErrorHandler.InvalidInputDim(type(variable), input_dim)
        else: 
            if (type(variable) is list or type(variable) is tuple) and all(type(x) is int for x in variable) and len(variable) == input_dim:

                return variable
            else:

                raise ErrorHandler.InvalidInputDim(type(variable), input_dim)

    def check_shape(variable):
        if type(variable) is int:
            variable = (1, variable)

            return variable

        elif (type(variable) is list or type(variable) is tuple) and all(type(x) is int for x in variable):

            return variable

        else:

            raise ErrorHandler.InvalidShape(type(variable))

            

