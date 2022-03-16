from nn_model import Model
from matplotlib import pyplot
import numpy as np

# define the dataset
x = np.asarray([i for i in range(-50,51)])
y = np.asarray([i**2.0 for i in x])


x = x.reshape((len(x), 1))
y = y.reshape((len(y), 1))

x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

x_norm = (x - x_min) / (x_max - x_min)
y_norm = (y - y_min) / (y_max - y_min)

def define_data(func_name):
    if func_name == 'Parabola':
        pass


model = Model()


model.add_input_layer(1, input_type = '1d')
# model.add_dense_layer(neurons_number = 50, activation_func = 'Sigmoid', bias = 0)
model.add_dense_layer(neurons_number = 100, activation_func = 'Sigmoid', bias = 0)
model.add_dense_layer(neurons_number = 100, activation_func = 'Sigmoid', bias = 0)
# model.add_dense_layer(neurons_number = 10, activation_func = 'Sigmoid', bias = 0)
# model.add_dense_layer(neurons_number = 50, activation_func = 'Sigmoid', bias = 0)
model.add_dense_layer(neurons_number = 1, activation_func = 'Sigmoid', bias = 0)


loss, acc = model.train(x_norm, y_norm, epochs = 50000, loss_function_name = 'MSE', optimizer_name = 'Adam')
test_acc = model.test(x_norm, y_norm)

loss_per_epoch = np.asfarray(loss).reshape(-1, len(x)).mean(axis=1) 
acc_per_epoch = np.asfarray(acc).reshape(-1, len(y)).mean(axis=1) 

# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(2)
# axs[0].plot(loss_per_epoch)
# axs[1].plot(acc_per_epoch)

# # axs[0].set_title("model loss")
# # axs[1].set_title("model accuracy")

# axs[0].set_ylabel('loss')
# axs[1].set_ylabel('accuracy')

# # axs[0].set_xlabel('epoch')
# axs[1].set_xlabel('epoch')


# plt.show()


yhat = []

for elem in x_norm: 
    yhat.append(model.predict(elem)[1][0])


yhat =  np.array(yhat, ndmin=2)

x_plot = x 
y_plot = y



yhat_plot = yhat  * (y_max - y_min) + y_min


pyplot.scatter(x, y, label='Actual')

pyplot.scatter(x ,yhat_plot, label='Predicted')
pyplot.title('Input (x) versus Output (y)')
pyplot.xlabel('Input Variable (x)')
pyplot.ylabel('Output Variable (y)')
pyplot.legend()
pyplot.show()