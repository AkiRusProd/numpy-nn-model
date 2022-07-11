import numpy as np
import os
from tqdm import tqdm
from nnmodel import Model
from nnmodel.optimizers import *
from nnmodel.loss_functions import *
from nnmodel.activations import *


class VAE():
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

        self.loss_function = MSE()
        self.optimizer = SGD()

        self.latent_dim_size = int(0)

    def compile(self, optimizer = SGD(), loss = MSE()):
        if type(optimizer) is str:
            self.optimizer = optimizers[optimizer]
        else:
            self.optimizer = optimizer

        if type(loss) is str:
            self.loss_function = loss_functions[loss]
        else:
            self.loss_function = loss

    def load(self, path):
        self.encoder = Model()
        self.decoder = Model()

        self.encoder.load(f"{path}/encoder")
        self.decoder.load(f"{path}/decoder")

    def save(self, path):
        try:
            os.mkdir(path)
        except:
            pass
        self.encoder.save(f"{path}/encoder")
        self.decoder.save(f"{path}/decoder")

    def reparametrize(self, outputs):
        mean, log_var = np.asfarray(outputs[..., : self.latent_dim_size]), np.asfarray(outputs[..., self.latent_dim_size :])
        std = np.exp(0.5 * log_var)
        eps = np.random.normal(0, 1, log_var.shape)
        
        z = mean + std * eps
        return z, mean, log_var, eps




    def encoder_forward_prop(self, batch):
        outputs = self.encoder.forward_prop(batch, training = True)
        z, mean, log_var, eps = self.reparametrize(outputs)
        
        return outputs, z, mean, log_var, eps
    
    def decoder_forward_prop(self, batch):
        return self.decoder.forward_prop(batch, training = True)

    def encoder_backward_prop(self, error, encoder_outputs,  mean, log_var, eps):
        rec_mu_loss = error
        rec_log_var_loss = error * 0.5 * np.exp(log_var * 0.5) * eps

        dkl_mu =  -0.5 * -2 * mean
        dkl_log_var =  0.5 * (np.exp(log_var) - 1)

        loss_mu = rec_mu_loss + dkl_mu
        loss_log_var = rec_log_var_loss + dkl_log_var

        error = np.concatenate((loss_mu, loss_log_var), axis = 1)
        return self.encoder.backward_prop(error)

    def decoder_backward_prop(self, error):
        return self.decoder.backward_prop(error)

    

    def fit(self, input_data, target_data, batch_size, epochs):
        input_data = np.asarray(input_data)
        target_data = np.asarray(target_data)

        input_batches = np.array_split(input_data,  np.arange(batch_size,len(input_data),batch_size))
        target_batches = np.array_split(target_data,  np.arange(batch_size,len(target_data),batch_size))
        
        self.latent_dim_size = self.encoder.layers[-1].output_shape[-1] // 2

        if len(input_batches[-1]) < batch_size:
            size = batch_size - len(input_batches[-1])
            indexes = np.random.choice(len(input_data), size = size, replace=False)
            
            new_samples = input_data[indexes]
            input_batches[-1] = np.concatenate((input_batches[-1], new_samples))
            new_target_samples = target_data[indexes]
            target_batches[-1] = np.concatenate((target_batches[-1], new_target_samples))

        decoder_loss_history = []
        kl_loss_history = []
        loss_history = []

        self.encoder.optimizer = self.optimizer
        self.decoder.optimizer = self.optimizer

        self.encoder.set_optimizer()
        self.decoder.set_optimizer()
        
        for i in range(epochs):
            tqdm_range = tqdm(enumerate(zip(input_batches, target_batches)), total = len(input_batches))
            for j, (input_batch, target_batch) in tqdm_range:

                encoder_outputs, reparametrized_outputs, mean, log_var, eps = self.encoder_forward_prop(input_batch)
                decoder_outputs = self.decoder_forward_prop(reparametrized_outputs)

                decoder_loss_history.append(self.loss_function.loss(decoder_outputs, target_batch).mean())
                kl_loss_history.append(
                    -0.5
                    * np.sum(
                        1 + log_var - np.power(mean, 2) - np.exp(log_var)
                    )
                    / (batch_size * self.latent_dim_size)
                )
                loss_history.append(decoder_loss_history[-1] + kl_loss_history[-1])

                error = self.loss_function.derivative(decoder_outputs, target_batch)
                error = self.decoder_backward_prop(error)
                self.encoder_backward_prop(error, encoder_outputs, mean, log_var, eps)

                self.decoder.update_weights()
                self.encoder.update_weights()

                tqdm_range.set_description(
                    f"VAE training | loss: {loss_history[-1]:.4f} | decoder loss: {decoder_loss_history[-1]:.4f} |kl loss: {kl_loss_history[-1]:.4f} | epoch {i + 1}/{epochs}"
                )

        return loss_history, decoder_loss_history, kl_loss_history

    def predict(self, input, from_decoder=False):

        if from_decoder == False:
            encoder_output = self.encoder.forward_prop(input, training = False)

            reparametrized_output, _, _, _ = self.reparametrize(encoder_output)

            decoder_output = self.decoder.forward_prop(reparametrized_output, training = False)
        else:
            decoder_output = self.decoder.forward_prop(input, training = False)

        return decoder_output


