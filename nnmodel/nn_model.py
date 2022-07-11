import numpy as np
import pickle as pkl
from tqdm import tqdm
from nnmodel.optimizers import *
from nnmodel.loss_functions import *
from nnmodel.activations import *

class Model():
    #TODO
    #add loss func and optimizers exception + their input values
    #add gan, vae modules
    #add examples
    #maybe something more...

    def __init__(self):
        self.layers = []

        #Default params
        self.loss_function = MSE()
        self.optimizer = SGD()

    def compile(self, optimizer = SGD(), loss = MSE()): 

        if type(optimizer) is str:
            self.optimizer = optimizers[optimizer]
        else:
            self.optimizer = optimizer

        if type(loss) is str:
            self.loss_function = loss_functions[loss]
        else:
            self.loss_function = loss

    def set_optimizer(self):
        for layer in self.layers:
            if hasattr(layer, 'set_optimizer'):
                layer.set_optimizer(self.optimizer)

    def load(self, path):
        pickle_model = open(path, 'rb')
        self.layers = pkl.load(pickle_model)
        pickle_model.close()
        

    def save(self, path):
        pickle_model = open(path, 'wb')
        pkl.dump(self.layers, pickle_model)
        pickle_model.close()
            


    def add(self, layer):
        if self.layers: layer.input_shape = self.layers[-1].output_shape #or layer.input_shape == None

        if hasattr(layer, 'build'):
                layer.build()

        self.layers.append(layer)


    def forward_prop(self, ouput, training):
        for layer in self.layers:
            ouput = layer.forward_prop(ouput, training)

        return ouput

    def backward_prop(self, error):
        for layer in reversed(self.layers):
            error = layer.backward_prop(error)

        return error


    def update_weights(self):
        for i, layer in enumerate(reversed(self.layers)):
            if hasattr(layer, 'update_weights'):
                layer.update_weights(layer_num = i + 1)





    def prepare_targets(self, batch_targets):
        prepared_batch_targets = []

        for target in batch_targets:
            
            try:
                correct_target = int(target)
                
                try:
                    last_layer_activation = self.layers[-1].activation
                except:
                    last_layer_activation = activations[None] #for layers thats has no activation

                last_layer_units_num = self.layers[-1].output_shape[-1] #NOTE: Units num that correctly works with Last Dense Layer

                
                targets_list = np.zeros(last_layer_units_num)
                if last_layer_activation == activations["tanh"]:
                    targets_list = np.full(last_layer_units_num, -1)

                targets_list[correct_target] = 1

            except:
                targets_list = target

            prepared_batch_targets.append(targets_list)
            

        return np.asarray(prepared_batch_targets)


    def fit(self, input_data, data_targets, batch_size, epochs):
        input_data = np.asarray(input_data)
        data_targets = np.asarray(data_targets)
        batch_num = len(input_data) // batch_size

        batches = np.array_split(input_data, np.arange(batch_size,len(input_data),batch_size))#np.stack
        batches_targets = np.array_split(data_targets, np.arange(batch_size,len(input_data),batch_size))#np.stack

        if len(batches[-1]) < batch_size:
            indexes = np.random.choice(len(input_data), size = batch_size - len(batches[-1]), replace=False)
            new_samples = input_data[indexes]
            new_samples_target = data_targets[indexes]
            batches[-1] = np.concatenate((batches[-1], new_samples))
            batches_targets[-1] = np.concatenate((batches_targets[-1], new_samples_target))

        self.set_optimizer()

        loss_history = []
        for i in range(epochs):
            tqdm_range = tqdm(enumerate(zip(batches, batches_targets)), total = len(batches))
            for j, (batch, batch_targets) in tqdm_range:
                predictions = self.forward_prop(batch, training = True)
                
                targets =     self.prepare_targets(batch_targets)
                # print("pred", predictions.shape,"tar", targets.shape)
                error = self.loss_function.derivative(predictions, targets)
                loss_history.append(self.loss_function.loss(predictions, targets).mean())

                self.backward_prop(error)

                self.update_weights()

                tqdm_range.set_description(
                        f"training | loss: {loss_history[-1]:.7f} | epoch {i + 1}/{epochs}" #loss: {loss:.4f}
                    )

        return loss_history

    def predict(self, input_data, data_targets):
        accuracy_history = []
        samples_num = true_samples_num = 0

        for i, (input, target) in tqdm(enumerate(zip(input_data, data_targets)), desc = "testing", total = len(input_data)):
            predictions = self.forward_prop(input.reshape(1, *input.shape), training = False)
            
            max_output_index = np.argmax(predictions)

            samples_num += 1

            if max_output_index == int(target):
                true_samples_num += 1

            accuracy_history.append(true_samples_num / samples_num)

            # print(f'inputs: {inputs[j]}, targets: {targets[j]}, output: {max_output_index}, output neurons values : {layers_outputs[len(layers_outputs)-1]}')

        print(f"> {accuracy_history[-1] * 100} %")
        return accuracy_history

    def predict_classes(self, input_data):
        predictions = self.forward_prop(input_data.reshape(1, *input_data.shape), training = False)
        return np.argmax(predictions, axis = 1), predictions