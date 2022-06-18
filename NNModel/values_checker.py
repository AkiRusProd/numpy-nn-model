from nnmodel.error_handler import ErrorHandler

class ValuesChecker():

    def check_activation(activation_function, activations):

        if type(activation_function) is str or activation_function is None:
            try:
                activation = activations[activation_function]

            except: 
                raise ErrorHandler.WrongActivationName(activation_function)
        else:
            try:
                if activation.__bases__.__class__.__name__ == "ActivationFunction":
                    activation = activation_function
            except: 
                
                raise ErrorHandler.WrongActivationType(type(activation))

        return activation

    def check_size2_variable(variable, variable_name = None):
        if type(variable) is int:
            variable = (variable, variable)
        elif type(variable) is str and variable_name  == "padding":
            if variable == "same" or variable == "valid":

                return variable

            else:
                raise ErrorHandler.WrongSize2Variable(type(variable), variable_name)

        else: 
            try:
                if (type(variable[0]) is int) and (type(variable[1]) is int) and len(variable) == 2:
                    variable = (variable[0], variable[1])
                else:
                    raise ErrorHandler.WrongSize2Variable(type(variable), variable_name)
            except:
                raise ErrorHandler.WrongSize2Variable(type(variable), variable_name)
        
        return variable

    def check_integer_variable(variable, variable_name = None):
        if type(variable) is int:

            return variable
        else:
            raise ErrorHandler.WrongIntegerValue(type(variable), variable_name)
