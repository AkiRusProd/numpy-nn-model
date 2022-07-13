from nnmodel.exceptions.error_handler import ErrorHandler

class ValuesChecker():
    def check_optimizer(optimizer, optimizers):
        if type(optimizer) == str:
            try:
                optimizer = optimizers[optimizer]
            except:
                raise ErrorHandler.InvalidOptimizerName(optimizer)
        else:
            if not any(isinstance(optimizer, type(optimizer)) for optimizer in tuple(optimizers.values())):
                raise ErrorHandler.InvalidOptimizerType(type(optimizer))

        return optimizer

                

    def check_loss(loss_function, loss_functions):
        if type(loss_function) == str:
            try:
                loss_function = loss_functions[loss_function]
            except:
                raise ErrorHandler.InvalidLossName(loss_function)
        else:
            if not any(isinstance(loss_function, type(loss)) for loss in tuple(loss_functions.values())):
                raise ErrorHandler.InvalidLossType(type(loss_function))

        return loss_function

    def check_activation(activation, activations):
        
        if type(activation) == str or activation == None:
            try:
                activation = activations[activation]
            except: 
                raise ErrorHandler.InvalidActivationName(activation)
        else:  
            if not any(isinstance(activation, type(activation)) for activation in tuple(activations.values())):
                raise ErrorHandler.InvalidActivationType(type(activation))
   
        return activation

    def check_recurrent_layer(layer, provided_layers, access_recurrent):
        if any(isinstance(layer, provided_layer) for provided_layer in provided_layers) == access_recurrent:
            return layer
        else:
            raise ErrorHandler.InvalidRecurrentLayer(layer, access_recurrent)

    def check_size2_variable(variable, variable_name = None, min_acceptable_value = 0):
        if type(variable) == int:
            variable = (variable, variable)

            return variable
        elif type(variable) == str and variable_name  == "padding":
            if variable == "same" or variable == "real same" or variable == "valid":

                return variable

            else:
                raise ErrorHandler.InvalidSize2Variable(type(variable), variable_name, min_acceptable_value)

        else: 
            if (type(variable) == list or type(variable) == tuple) and all(type(x) is int and x >= min_acceptable_value for x in variable) and len(variable) == 2:
                return variable
            else:
                raise ErrorHandler.InvalidSize2Variable(type(variable), variable_name, min_acceptable_value)
            

    def check_integer_variable(variable, variable_name = None):
        if type(variable) == int and variable > 0:

            return variable
        else:
            raise ErrorHandler.InvalidIntegerValue(type(variable), variable_name)

    def check_float_variable(variable, variable_name = None):
        if type(variable) == float and variable >= 0:

            return variable
        else:
            raise ErrorHandler.InvalidFloatValue(type(variable), variable_name)

    def check_input_dim(variable, input_dim):
        if variable == None or input_dim == None:
            return variable

        elif type(variable) == int and (input_dim == 2 or input_dim == 1):
            variable = (1, variable)

            return variable
        elif type(variable) == str:
           
            raise ErrorHandler.InvalidInputDim(type(variable), input_dim)
        else: 
            if (type(variable) is list or type(variable) is tuple) and all(type(x) is int and x > 0 for x in variable) and len(variable) == input_dim:

                return variable
            else:

                raise ErrorHandler.InvalidInputDim(type(variable), input_dim)

    def check_shape(variable):
        if type(variable) == int:
            variable = (1, variable)

            return variable

        elif (type(variable) is list or type(variable) is tuple) and all(type(x) is int and x > 0 for x in variable):

            return variable

        else:

            raise ErrorHandler.InvalidShape(type(variable))

    def check_boolean_type(variable, variable_name = None):
        
        if type(variable) == bool:
            
            return variable
        else:

            raise ErrorHandler.InvalidBoleanType(variable, variable_name)

            

