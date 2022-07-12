import numpy as np
import os
from tqdm import tqdm
from nnmodel import Model
from nnmodel.optimizers import *
from nnmodel.loss_functions import *
from nnmodel.activations import *


class GAN():
    """
    Generative Adversarial Network (GAN) module, thats responsible for training the generator and discriminator
    -----------------------------------------------------------------------------------------------------------
        Args:
            `generator` (Model): Generator model
            `discriminator` (Model): Discriminator model
        Methods:
            `compile`: choose optimizer, loss function and predict mode every epoch
            `fit`: train the model
            `predict`: predict the model
            `load`: load the model
            `save`: save the model
            `get_each_epoch_predictions`: get the each epoch predictions of the model, if each_epoch_predict["mode"] is True
        References:
            https://arxiv.org/abs/1406.2661
    """

    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

        self.loss_function = MSE()
        self.optimizer = SGD()

    def compile(self, optimizer = SGD(), loss = MSE(), each_epoch_predict = {"mode" : False, "num" : 0}): 
        """
        Compile the model
        -----------------
            Args:
                `optimizer`: optimizer defined in `nnmodel.optimizers`; default is `SGD`
                `loss`: loss function defined in `nnmodel.loss_functions`; default is `MSE`
                `each_epoch_predict`: if `each_epoch_predict["mode"]` is `True`, the model will give `num` generator predictions every epoch
        """
        if type(optimizer) is str:
            self.optimizer = optimizers[optimizer]
        else:
            self.optimizer = optimizer

        if type(loss) is str:
            self.loss_function = loss_functions[loss]
        else:
            self.loss_function = loss

        self.each_epoch_predict = each_epoch_predict
        self.each_epoch_predict["predictions"] = []

    def load(self, path):
        """
        Load the model
        --------------
            Args:
                `path`: path to the saved model
        """
        self.generator = Model()
        self.discriminator = Model()

        self.generator.load(f"{path}/generator")
        self.discriminator.load(f"{path}/discriminator")

    def save(self, path):
        """
        Save the model
        --------------
            Args:
                `path`: path to the model which will be saved
        """
        try:
            os.mkdir(path)
        except:
            pass
        self.generator.save(f"{path}/generator")
        self.discriminator.save(f"{path}/discriminator")

    def prepare_targets(self):

        d_last_layer_units_num = self.discriminator.layers[-1].output_shape[-1]

        if d_last_layer_units_num == 2:
            self.real_targets, self.fake_targets = np.array([0, 1]), np.array([1, 0])

        elif d_last_layer_units_num == 1:
            self.real_targets, self.fake_targets = np.array([1]), np.array([0])
            


    def prepare_loss_function(self):
        if self.loss_function.__class__.__name__ == "MiniMaxCrossEntropy":
            self.d_real_loss_function_deriv = self.loss_function.discriminator_real_derivative
            self.d_fake_loss_function_deriv = self.loss_function.discriminator_fake_derivative
            self.g_loss_function_deriv = self.loss_function.generator_derivative

            # self.d_real_loss = self.loss_function.discriminator_real_loss
            # self.d_fake_loss = self.loss_function.discriminator_fake_loss
            # self.g_loss = self.loss_function.generator_loss
        else:
            self.d_real_loss_function_deriv = self.loss_function.derivative
            self.d_fake_loss_function_deriv = self.loss_function.derivative
            self.g_loss_function_deriv = self.loss_function.derivative

            # self.d_real_loss = self.loss_function.loss
            # self.d_fake_loss = self.loss_function.loss
            # self.g_loss = self.loss_function.loss
    

    def g_forward_prop(self, batch_noise):
        return self.generator.forward_prop(batch_noise, training = True)
    
    def d_forward_prop(self, batch_data):
        return self.discriminator.forward_prop(batch_data, training = True)

    def g_backward_prop(self, d_fake_error):
        batch_errors = self.discriminator.backward_prop(d_fake_error)
        self.generator.backward_prop(batch_errors)
    
    def d_backward_prop(self, batch_error):
        self.discriminator.backward_prop(batch_error)

        each_layer_grads = []
        for layer in self.discriminator.layers:
            
            if hasattr(layer, 'get_grads'):
                each_layer_grads.append(layer.get_grads())

        return each_layer_grads

    def set_grads(self, model, grads):
        i = 0
        for layer in (model.layers):
            if hasattr(layer, 'set_grads'):
                layer.set_grads(grads[i])

                i += 1
                

        

    def fit(self, input_data,  batch_size,  epochs, noise_vector_size = 100):
        """
        Train the model
        ---------------
            Args:
                `input_data`: input data for the model
                `batch_size`: batch size of the model
                `epochs`: epochs number to train the model
                `noise_vector_size`: noise vector size (default = 100)
            Returns:
                `generator loss history`,
                `discriminator loss history`
        """
        real_data = np.asarray(input_data)

        batches = np.array_split(input_data,  np.arange(batch_size,len(input_data),batch_size))
        noises = np.array_split(
            np.random.normal(
                0, 1, (len(real_data), noise_vector_size)
            ),
            np.arange(batch_size,len(input_data),batch_size),
        )

        if len(batches[-1]) < batch_size:
            size = batch_size - len(batches[-1])
            new_samples = real_data[np.random.choice(len(real_data), size = size, replace=False)]
            batches[-1] = np.concatenate((batches[-1], new_samples))
            noises[-1] = np.concatenate((noises[-1], np.random.normal(0, 1, (size, noise_vector_size))))

        if self.each_epoch_predict["mode"] == True:
            each_epoch_noises = np.random.normal(0, 1, (self.each_epoch_predict["num"], noise_vector_size))
            
        g_loss_history = []
        d_loss_history = []


        self.generator.optimizer = self.optimizer
        self.discriminator.optimizer = self.optimizer
        # self.generator.loss_function = self.loss_function
        # self.discriminator.loss_function = self.loss_function

        self.generator.set_optimizer()
        self.discriminator.set_optimizer()

        self.prepare_targets()
        self.prepare_loss_function()
        
        for i in range(epochs):
            tqdm_range =  tqdm(enumerate(zip(batches, noises)), total = len(batches))

            for j, (batch, noise) in tqdm_range:
                g_outputs = self.g_forward_prop(noise)
                
                d_real_predictions = self.d_forward_prop(batch)
                d_real_error = self.d_real_loss_function_deriv(d_real_predictions, self.real_targets)
                d_real_grads = self.d_backward_prop(d_real_error)

                d_fake_predictions = self.d_forward_prop(g_outputs)
                d_fake_error = self.d_fake_loss_function_deriv(d_fake_predictions, self.fake_targets)
                d_fake_grads = self.d_backward_prop(d_fake_error)
                
                # d_loss_history.append((self.d_real_loss(d_real_predictions, self.real_targets).mean() + self.d_fake_loss(d_fake_predictions, self.fake_targets).mean()))
                # g_loss_history.append((self.g_loss(d_real_predictions, self.real_targets)).mean())

                d_loss_history.append((-np.log(d_real_predictions) - np.log(1 - d_fake_predictions)).mean())
                g_loss_history.append((-np.log(d_real_predictions).mean()))
                

                
                self.set_grads(self.discriminator, d_fake_grads)
                self.discriminator.update_weights()
                self.set_grads(self.discriminator, d_real_grads)
                self.discriminator.update_weights()
                
                d_fake_predictions = self.discriminator.forward_prop(g_outputs, training = True)
                d_fake_error = self.g_loss_function_deriv(d_fake_predictions, self.real_targets)
                self.g_backward_prop(d_fake_error)
                self.generator.update_weights()

                tqdm_range.set_description(
                    f"GAN training | G loss: {g_loss_history[-1]:.4f} | D loss: {d_loss_history[-1]:.4f} | epoch {i + 1}/{epochs}"
                )

            if self.each_epoch_predict["mode"] == True:
                self.each_epoch_predict["predictions"].append(self.generator.forward_prop(each_epoch_noises, training = False))
                

        return g_loss_history, d_loss_history

    def predict(self, input_noise):
        """
        Predict the model
        -----------------
            Args:
                `input_noise`: input noise for the generator model
            Returns:
                `predictions`: predictions of the generator model
        """
        return self.generator.forward_prop(input_noise, training = False)

    def get_each_epoch_predictions(self):
        """
        Get the predictions of the model for each epoch
        -----------------------------------------------
            Returns:
                `predictions`: if `each_epoch_predict["mode"]` in the 'compile' method is `True`, the model will give `num` generator predictions every epoch
        """
        return self.each_epoch_predict["predictions"]
                