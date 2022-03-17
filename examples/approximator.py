from nn_model import Model
from matplotlib import pyplot
import numpy as np


def define_data(left_edge, right_edge, interval, applied_func):
    x_coords = np.array(np.linspace(left_edge, right_edge, interval), ndmin = 2)
    y_coords = np.asarray([applied_func(x) for x in x_coords])

    return x_coords.T, y_coords.T

#Examples of functions
x, y = define_data(-50, 50, 100, lambda x: x**2) #y = x^2
#x, y = define_data(-2, 2, 100, lambda x: np.sin(x)) #y = sin(x)
# x, y = define_data(0, 10, 100, lambda x: np.sqrt(x)) # y = sqrt(x)

#Normalize x and y coordinates
x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))  





#Define model
model = Model()

model.add_input_layer(1, input_type = '1d')
model.add_dense_layer(neurons_number = 100, activation_func = 'Sigmoid', bias = 0)
model.add_dense_layer(neurons_number = 100, activation_func = 'Sigmoid', bias = 0)
model.add_dense_layer(neurons_number = 1, activation_func = 'Sigmoid', bias = 0)

#Train model
model.train(x_norm, y_norm, epochs = 5000, loss_function_name = 'MSE', trained_model= 'encoder', batch_size = 10, optimizer_name = 'Adam')



#Get predicted y coordinates
yhat = []
for elem in x_norm: 
    yhat.append(model.predict(elem)[1][0])

yhat =  np.array(yhat, ndmin=2)

#Unnormalize predicted y coordinates
yhat_plot = yhat  * (np.max(y) - np.min(y)) + np.min(y)


pyplot.scatter(x, y, label='Actual')
pyplot.scatter(x, yhat_plot, label='Predicted')
pyplot.title('Input (x) versus Output (y)')
pyplot.xlabel('Input Variable (x)')
pyplot.ylabel('Output Variable (y)')
pyplot.legend()
pyplot.show()