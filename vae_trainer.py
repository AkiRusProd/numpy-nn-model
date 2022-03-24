
from nn_model import Model
from tqdm import tqdm
import os
import numpy as np

class VAE():

    def __init__(self, encoder_model = None, decoder_model = None):
        self.encoder = encoder_model
        self.decoder = decoder_model

        self.latent_dim_size = 0
        self.decoder_loss_function = None

    def save(self, name):
    
        try:
            os.mkdir(name)
        except: pass

        self.encoder.save(f'{name}/encoder')
        self.decoder.save(f'{name}/decoder')



    def load(self, name):
        self.encoder = Model()
        self.decoder = Model()

        self.encoder.load(f'{name}/encoder')
        self.decoder.load(f'{name}/decoder')

        self.latent_dim_size = self.encoder.topology[-1]['neurons num'] // 2
    
 

    def encoder_forward_prop(self, inputs):
        batch_layers_outputs = []
        last_outputs = []
        means = []
        log_variances = []
        epsilons = []

        for k in range(len(inputs)):

            outputs = self.encoder.forward_prop(inputs[k])

            reparametrized_outputs, mean, log_variance, epsilon = self.reparametrize(outputs[-1][0])

            batch_layers_outputs.append(outputs)
            last_outputs.append(reparametrized_outputs)

            means.append(mean)
            log_variances.append(log_variance)
            epsilons.append(epsilon)


        return batch_layers_outputs, np.asarray(last_outputs), np.asarray(means), np.asarray(log_variances), np.asarray(epsilons)

    def reparametrize(self, outputs):

        mean = np.asfarray(outputs[:self.latent_dim_size])
        log_variance = np.asfarray(outputs[self.latent_dim_size:])

        epsilon = np.random.normal(0, 1, (log_variance.shape))  #(1, self.latent_dim_size)

        
        z = mean + np.exp(log_variance * 0.5) * epsilon


        return z, mean, log_variance, epsilon

    def decoder_forward_prop(self, inputs):
        batch_layers_outputs = []
        batch_layers_last_outputs = []

        for k in range(len(inputs)):

            outputs = self.decoder.forward_prop(inputs[k])
            batch_layers_outputs.append(outputs)
            batch_layers_last_outputs.append(outputs[-1])

        
        return batch_layers_outputs, batch_layers_last_outputs




    def encoder_backward_prop(self, batch_layers_outputs, batch_decoder_losses,  batch_means, batch_log_variances, batch_epsilons):
        batch_layers_losses = []

        for k in range(len(batch_layers_outputs)):

            # loss = np.concatenate((batch_decoder_losses[k][0], np.zeros(batch_decoder_losses[k][0].shape)), axis = 1)
            
            rec_mu_loss = batch_decoder_losses[k][0]
            rec_log_var_loss = batch_decoder_losses[k][0] * 0.5 * np.exp(batch_log_variances[k] * 0.5) *  batch_epsilons[k]#np.random.normal(0, 1, ((batch_log_variances[k].shape))) #(1, self.latent_dim_size)

            dkl_mu = -0.5 * -2 * batch_means[k]
            dkl_logvar = 0.5*(np.exp(batch_log_variances[k]) - 1)

            loss_mu = rec_mu_loss + dkl_mu
            loss_log_var = rec_log_var_loss +  dkl_logvar
            # print(loss_mu.shape, loss_log_var.shape)

            loss = np.concatenate((loss_mu, loss_log_var), axis = 1)

            # loss = np.concatenate((rec_mu_loss, rec_log_var_loss), axis = 1)

            losses = self.encoder.backward_prop(batch_layers_outputs[k], loss)
            batch_layers_losses.append(losses)

        self.encoder.weights_updating(batch_layers_outputs, batch_layers_losses, self.encoder_optimizer)
        

    def decoder_backward_prop(self, batch_layers_outputs, batch_targets):
        batch_layers_losses = []

        for k in range(len(batch_layers_outputs)):

            losses = self.decoder.backward_prop(batch_layers_outputs[k], self.decoder_loss_function(batch_layers_outputs[k][-1], batch_targets[k]))
            batch_layers_losses.append(losses)

        self.decoder.weights_updating(batch_layers_outputs, batch_layers_losses, self.decoder_optimizer)

        return batch_layers_losses


    def train(self, inputs, targets, epochs, optimizer_name, loss_function_name,  batch_size = 10, **optimizer_params):
        self.latent_dim_size = self.encoder.topology[-1]['neurons num']//2

        inputs = np.asfarray(inputs)
        targets = np.asfarray(targets)

        batch_num = len(inputs) // batch_size

        inputs_batches = np.array_split(inputs, batch_num)
        targets_batches = np.array_split(targets, batch_num)

        self.encoder_optimizer = self.encoder.optimizers[optimizer_name]
        self.decoder_optimizer = self.decoder.optimizers[optimizer_name]

        self.decoder_loss_function = self.decoder.loss_functions[loss_function_name]
        self.decoder_loss_function_metric = self.decoder.loss_functions_metrics[loss_function_name]

        self.encoder.set_params(optimizer_params)
        self.decoder.set_params(optimizer_params)

        if len(self.encoder.weights) == 0:
            self.encoder.weights_init()
        if len(self.decoder.weights) == 0:
            self.decoder.weights_init()


        for i in range(epochs):
            
            tqdm_range = tqdm(range(batch_num))
            # tqdm_range = (range(batch_num))
            for j in tqdm_range:
                
                
                encoder_outputs, reparametrized_outputs, means, log_variances, epsilons = self.encoder_forward_prop(inputs_batches[j])
                decoder_outputs, decoder_last_outputs = self.decoder_forward_prop(reparametrized_outputs)

                decoder_losses = self.decoder_backward_prop(decoder_outputs, targets_batches[j])
                self.encoder_backward_prop(encoder_outputs, decoder_losses,  means, log_variances, epsilons)
                
               
                decoder_loss = self.decoder_loss_function_metric(np.asfarray(decoder_last_outputs), targets_batches[j]).mean()
                kl_loss = -0.5 * np.sum(1 + log_variances - np.power(means, 2) - np.exp(log_variances)) / (batch_size * self.latent_dim_size)
                loss =  decoder_loss + kl_loss
         

                tqdm_range.set_description(f'training | optimizer: {optimizer_name} | loss: {loss:.4f} | decoder loss: {decoder_loss:.4f} |kl loss: {kl_loss:.4f} | epoch {i + 1}/{epochs}')#


    def predict(self, input, from_decoder = False):

        if from_decoder == False:
            encoder_output = self.encoder.forward_prop(input)
      
            reparametrized_output, _, _, _ = self.reparametrize(encoder_output[-1][0])
            
            decoder_output = self.decoder.forward_prop(reparametrized_output)
        else:
            decoder_output = self.decoder.forward_prop(input)

        return decoder_output[-1]

                
