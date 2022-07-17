import numpy as np
import pickle as pkl
from tqdm import tqdm
from nnmodel.optimizers import *
from nnmodel.loss_functions import *
from nnmodel.activations import *
from nnmodel.exceptions.values_checker import *

class Model():
    """
    Model class, that defines a model
    ---------------------------------
        Methods:
            `add` (layer): adds layer to model
            `compile`: choose optimizer and loss function
            `fit`: train the model
            `predict`: test the model
            `predict_`: predict the model
            `load`: load the model
            `save`: save the model
    """
    def __init__(self):
        self.layers = []

        #Default params
        self.loss_function = MSE()
        self.optimizer = SGD()

    def compile(self, optimizer = SGD(), loss = MSE()): 
        """
        Compile the model
        -----------------
            Args:
                `optimizer`: optimizer defined in `nnmodel.optimizers`; default is `SGD`
                `loss`: loss function defined in `nnmodel.loss_functions`; default is `MSE`
        """
        self.optimizer = ValuesChecker.check_optimizer(optimizer, optimizers)
        self.loss_function = ValuesChecker.check_loss(loss, loss_functions)

    def set_optimizer(self):
        for layer in self.layers:
            if hasattr(layer, 'set_optimizer'):
                layer.set_optimizer(self.optimizer)

    def load(self, path):
        """
        Load the model
        --------------
            Args:
                `path`: path to the saved model
        """
        pickle_model = open(path, 'rb')
        self.layers = pkl.load(pickle_model)
        pickle_model.close()
        

    def save(self, path):
        """
        Save the model
        --------------
            Args:
                `path`: path to the model which will be saved
        """
        pickle_model = open(path, 'wb')
        pkl.dump(self.layers, pickle_model)
        pickle_model.close()
            


    def add(self, layer):
        """
        Add layer to model
        ------------------
            Args:
                `layer` : layer to add to model
        """
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





    def prepare_labels(self, batch_labels):
        prepared_batch_labels = []

        for label in batch_labels:
            
            try:
                correct_label = int(label)
                
                try:
                    last_layer_activation = self.layers[-1].activation
                except:
                    last_layer_activation = activations[None] #for layers thats has no activation

                last_layer_units_num = self.layers[-1].output_shape[-1] #NOTE: Units num that correctly works with Last Dense Layer

                
                labels_list = np.zeros(last_layer_units_num)
                if last_layer_activation == activations["tanh"]:
                    labels_list = np.full(last_layer_units_num, -1)

                labels_list[correct_label] = 1

            except:
                labels_list = label

            prepared_batch_labels.append(labels_list)
            

        return np.asarray(prepared_batch_labels)


    def fit(self, input_data, data_labels, batch_size, epochs):
        """
        Train the model
        ---------------
            Args:
                `input_data`: input data for the model
                `data_labels`: labels of input data for the model
                `batch_size`: batch size of the model
                `epochs`: epochs number to train the model
            Returns:
                `loss history`
        """
        input_data = np.asarray(input_data)
        data_labels = np.asarray(data_labels)
        batch_num = len(input_data) // batch_size

        batches = np.array_split(input_data, np.arange(batch_size,len(input_data),batch_size))#np.stack
        label_batches = np.array_split(data_labels, np.arange(batch_size,len(input_data),batch_size))#np.stack

        if len(batches[-1]) < batch_size:
            indexes = np.random.choice(len(input_data), size = batch_size - len(batches[-1]), replace=False)
            new_samples = input_data[indexes]
            new_label_samples = data_labels[indexes]
            batches[-1] = np.concatenate((batches[-1], new_samples))
            label_batches[-1] = np.concatenate((label_batches[-1], new_label_samples))

        self.set_optimizer()

        loss_history = []
        for i in range(epochs):
            tqdm_range = tqdm(enumerate(zip(batches, label_batches)), total = len(batches))
            for j, (batch, batch_labels) in tqdm_range:
                predictions = self.forward_prop(batch, training = True)
                
                labels =     self.prepare_labels(batch_labels)
                # print("pred", predictions.shape,"tar", labels.shape)
                error = self.loss_function.derivative(predictions, labels)
                loss_history.append(self.loss_function.loss(predictions, labels).mean())

                self.backward_prop(error)

                self.update_weights()

                tqdm_range.set_description(
                        f"training | loss: {loss_history[-1]:.7f} | epoch {i + 1}/{epochs}" #loss: {loss:.4f}
                    )

        return loss_history

    def predict(self, input_data, data_labels):
        """
        Predict (Test) the model
        -----------------
            Args:
                `input_data`: input data on which the model will be tested
                `data_labels`: labels of input data for the model
            Returns:
                `predictions`: predictions of the model
        """
        accuracy_history = []
        samples_num = true_samples_num = 0

        for i, (input, label) in tqdm(enumerate(zip(input_data, data_labels)), desc = "testing", total = len(input_data)):
            predictions = self.forward_prop(input.reshape(1, *input.shape), training = False)
            
            max_output_index = np.argmax(predictions)

            samples_num += 1

            if max_output_index == int(label):
                true_samples_num += 1

            accuracy_history.append(true_samples_num / samples_num)

            # print(f'inputs: {inputs[j]}, labels: {labels[j]}, output: {max_output_index}, output neurons values : {layers_outputs[len(layers_outputs)-1]}')

        print(f"> {accuracy_history[-1] * 100} %")
        return accuracy_history

    def predict_(self, input_data):
        """
        Predict the model
        -----------------
            Args:
                `input_data`: input data for the model
            Returns:
                `max pridiction index`: index of the max pridiction
                `predictions`: predictions of the model
        """
        predictions = self.forward_prop(input_data.reshape(1, *input_data.shape), training = False)
        return np.argmax(predictions, axis = 1), predictions