import numpy as np
from tqdm import tqdm
from NNModel.optimizers import *
from NNModel.loss_functions import *
from NNModel.activations import *

class Model():

    def __init__(self):
        self.layers = []

        #Default params
        self.loss_function = MSE()
        self.optimizer = Nadam()

    def compile(self, optimizer, loss_function): 

        if type(optimizer) is str:
            self.optimizer = optimizers[optimizer]
        else:
            self.optimizer = optimizer

        if type(loss_function) is str:
            self.loss_function = loss_functions[loss_function]
        else:
            self.loss_function = loss_function


    def add(self, layer):
        if self.layers: layer.input_shape = self.layers[-1].output_shape #or layer.input_shape == None

        if hasattr(layer, 'build'):
                layer.build(self.optimizer)

        self.layers.append(layer)


    def forward_prop(self, ouput, training):
        for layer in self.layers:
            ouput = layer.forward_prop(ouput, training)

        return ouput

    def backward_prop(self, error):
        for layer in reversed(self.layers):
            error = layer.backward_prop(error)


    def update_weights(self):
        for i, layer in enumerate(reversed(self.layers)):
            if hasattr(layer, 'update_weights'):
                layer.update_weights(layer_num = i + 1)





    def prepare_targets(self, batch_targets):
        prepared_batch_targets = []

        for target in batch_targets:
            
            if type(target) is not list:
                
                correct_target = int(target)

                last_layer_activation = self.layers[-1].activation

                if last_layer_activation == activations["sigmoid"] or last_layer_activation == activations["softmax"]:
                    targets_list = np.zeros(self.layers[-1].units_num)
                elif last_layer_activation == activations["tanh"]:
                    targets_list = np.full(self.layers[-1].units_num, -1)

                targets_list[correct_target] = 1

            else:
                targets_list = target

            prepared_batch_targets.append(targets_list)
            

        return np.asarray(prepared_batch_targets)


    def fit(self, input_data, data_targets, batch_size, epochs):
        input_data = np.asarray(input_data)
        batch_num = len(input_data) // batch_size

        batches = np.array_split(input_data, batch_num)#np.stack
        batches_targets = np.array_split(data_targets, batch_num)#np.stack


        for i in range(epochs):
            tqdm_range = tqdm(enumerate(zip(batches, batches_targets)), total = len(batches))
            for j, (batch, batch_targets) in tqdm_range:
                predictions = self.forward_prop(batch, training = True)
                
                targets =     self.prepare_targets(batch_targets)
            
                error = self.loss_function.derivative(predictions, targets)
                loss = self.loss_function.loss(predictions, targets).mean()

                self.backward_prop(error)

                self.update_weights()

                tqdm_range.set_description(
                        f"training | loss: {loss:.4f} | epoch {i + 1}/{epochs}" #loss: {loss:.4f}
                    )

    def predict(self, input_data, data_targets):
        accuracy = []
        samples_num = true_samples_num = 0

        for i, (input, target) in tqdm(enumerate(zip(input_data, data_targets)), desc = "testing", total = len(input_data)):
            predictions = self.forward_prop(input.reshape(1, *input.shape), training = False)

            max_output_index = np.argmax(predictions)

            samples_num += 1

            if max_output_index == int(target):
                true_samples_num += 1

            accuracy.append(true_samples_num / samples_num)

            # print(f'inputs: {inputs[j]}, targets: {targets[j]}, output: {max_output_index}, output neurons values : {layers_outputs[len(layers_outputs)-1]}')

        print(f"> {accuracy[-1] * 100} %")