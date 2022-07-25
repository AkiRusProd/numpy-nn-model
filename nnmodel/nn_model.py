import numpy as np
import pickle as pkl
from tqdm import tqdm
from nnmodel.optimizers import *
from nnmodel.losses import *
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
                `loss`: loss function defined in `nnmodel.losses`; default is `MSE`
        """
        self.optimizer = ValuesChecker.check_optimizer(optimizer, optimizers)
        self.loss_function = ValuesChecker.check_loss(loss, losses)

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

        try:
            last_layer_activation = self.layers[-1].activation
        except:
            last_layer_activation = activations[None] #for layers thats has no activation

        last_layer_units_num = self.layers[-1].output_shape[-1] #NOTE: Units num that correctly works with Last Dense Layer

        for label in batch_labels:
            
            try:
                correct_label = int(label)
                                
                if last_layer_units_num != 1:
                    labels_list = np.zeros(last_layer_units_num)
                    if last_layer_activation == activations["tanh"]:
                        labels_list = np.full(last_layer_units_num, -1)

                    labels_list[correct_label] = 1
                elif last_layer_units_num == 1:
                    labels_list = [correct_label]

            except:
                labels_list = label

            prepared_batch_labels.append(labels_list)
            
        # print(prepared_batch_labels)
        return np.asarray(prepared_batch_labels)


    def fit(self, input_data, label_data, batch_size, epochs):
        """
        Train the model
        ---------------
            Args:
                `input_data`: input data for the model
                `label_data`: label for the model
                `batch_size`: batch size of the model
                `epochs`: epochs number to train the model
            Returns:
                `loss history`
        """
        input_data = np.asarray(input_data)
        label_data = np.asarray(label_data)
        batch_num = len(input_data) // batch_size

        batches = np.array_split(input_data, np.arange(batch_size,len(input_data),batch_size))#np.stack
        label_batches = np.array_split(label_data, np.arange(batch_size,len(input_data),batch_size))#np.stack

        if len(batches[-1]) < batch_size:
            indexes = np.random.choice(len(input_data), size = batch_size - len(batches[-1]), replace=False)
            new_samples = input_data[indexes]
            new_samples_label = label_data[indexes]
            batches[-1] = np.concatenate((batches[-1], new_samples))
            label_batches[-1] = np.concatenate((label_batches[-1], new_samples_label))

        self.set_optimizer()

        loss_history = []
        for i in range(epochs):
            tqdm_range = tqdm(enumerate(zip(batches, label_batches)), total = len(batches))
            for j, (batch, label_batch) in tqdm_range:
                predictions = self.forward_prop(batch, training = True)
                
                labels =     self.prepare_labels(label_batch)
                # print("pred", predictions.shape,"tar", labels.shape)
                error = self.loss_function.derivative(predictions, labels)
                loss_history.append(self.loss_function.loss(predictions, labels).mean())

                self.backward_prop(error)

                self.update_weights()

                tqdm_range.set_description(
                        f"training | loss: {loss_history[-1]:.7f} | epoch {i + 1}/{epochs}" #loss: {loss:.4f}
                    )

        return loss_history

    def predict(self, input_data, label_data = None):
        """
        Predict (Test) the model
        -----------------
            Args:
                `input_data`: input data on which the model will be tested
                `label_data`: label data of input data for the model
            Returns:
                `predictions`: predictions of the model on input data if 'label_data' is None;
                `prediction` and 'max pridiction index` if 'label_data' is not None
        """

        if label_data == None:
            predictions = self.forward_prop(np.asarray(input_data).reshape(1, *np.asarray(input_data).shape), training = False)
            return np.argmax(predictions, axis = 1), predictions
            
        accuracy_history = []
        samples_num = true_samples_num = 0
        
        input_data = np.asarray(input_data)
        label_data = np.asarray(label_data)

        try:
            last_layer_activation = self.layers[-1].activation
        except:
            last_layer_activation = activations[None] #for layers thats has no activation



        for i, (input, label) in tqdm(enumerate(zip(input_data, label_data)), desc = "testing", total = len(input_data)):
            predictions = self.forward_prop(input.reshape(1, *input.shape), training = False)
            
            
            if len(predictions[0]) != 1:
                max_output_index = np.argmax(predictions)

            elif len(predictions[0]) == 1:
               
                if last_layer_activation == activations["tanh"]:
                    if predictions[0][0] > 0:
                        max_output_index = 1
                    else:
                        max_output_index = 0
                if last_layer_activation == activations["sigmoid"]:
                    
                    if predictions[0][0] > 0.5:
                        max_output_index = 1
                    else:
                        max_output_index = 0
                        
            samples_num += 1

            if max_output_index == int(label):
                true_samples_num += 1

            accuracy_history.append(true_samples_num / samples_num)

            # print(f'inputs: {inputs[j]}, labels: {labels[j]}, output: {max_output_index}, output neurons values : {layers_outputs[len(layers_outputs)-1]}')

        print(f"> {accuracy_history[-1] * 100} %")
        return accuracy_history
