from nn_model import Model
from tqdm import tqdm
import os
import numpy as np


class GAN:


    def __init__(self, generator_model=None, discriminator_model=None):
        self.generator = generator_model
        self.discriminator = discriminator_model

        self.epoch_prediction = False
        self.outputs_num_per_one_epoch = 1
        self.all_outputs_per_epochs = []
        self.noises_for_gen_per_epoch = []


    def prepare_loss_functions(self, loss_function_name):
        if loss_function_name == "minimax crossentropy":
            self.discriminator_real_loss_func = lambda output, targets=None: -1 / (
                output + 1e-8
            )  # min -log(D(x))
            self.discriminator_fake_loss_func = lambda output, targets=None: 1 / (
                1 - output + 1e-8
            )  # max log(1 - D(G(z)))
            self.generator_loss_func = lambda output, targets=None: -1 / (
                output + 1e-8
            )  # -log(D(G(z)))
        else:
            self.discriminator_real_loss_func = self.discriminator.loss_functions[
                loss_function_name
            ]
            self.discriminator_fake_loss_func = self.discriminator.loss_functions[
                loss_function_name
            ]
            self.generator_loss_func = self.discriminator.loss_functions[
                loss_function_name
            ]


    def prepare_targets(self):
        if self.discriminator.topology[-1]["neurons num"] == 2:
            self.real_targets, self.fake_targets = np.array([0, 1]), np.array([1, 0])

        elif self.discriminator.topology[-1]["neurons num"] == 1:
            self.real_targets, self.fake_targets = np.array([1]), np.array([0])

        if self.discriminator.topology[-1]["activation func"] == "Tanh":
            self.real_targets = self.real_targets * 2 - 1
            self.fake_targets = self.fake_targets * 2 - 1


    def save(self, name):

        try:
            os.mkdir(name)
        except:
            pass

        self.generator.save(f"{name}/generator")
        self.discriminator.save(f"{name}/discriminator")


    def load(self, name):
        self.generator = Model()
        self.discriminator = Model()

        self.generator.load(f"{name}/generator")
        self.discriminator.load(f"{name}/discriminator")


    def generator_forward_prop(self, noise_vectors):
        batch_layers_outputs = []
        fake_images = []

        for k in range(len(noise_vectors)):

            generator_outputs = self.generator.forward_prop(noise_vectors[k])
            batch_layers_outputs.append(generator_outputs)
            fake_images.append(generator_outputs[-1])

        return batch_layers_outputs, fake_images


    def discriminator_forward_prop(self, inputs):
        batch_layers_outputs = []
        last_outputs = []

        for k in range(len(inputs)):

            discriminator_outputs = self.discriminator.forward_prop(inputs[k])
            batch_layers_outputs.append(discriminator_outputs)

            last_outputs.append(discriminator_outputs[-1])

        return batch_layers_outputs, np.asarray(last_outputs)


    def generator_backward_prop(self, batch_layers_outputs):
        batch_layers_losses = []

        for k in range(len(batch_layers_outputs)):

            discriminator_outputs = self.discriminator.forward_prop(
                np.array(batch_layers_outputs[k][-1], ndmin=2)
            )

            discriminator_losses = self.discriminator.backward_prop(
                discriminator_outputs,
                self.generator_loss_func(discriminator_outputs[-1], self.real_targets),
            )

            generator_losses = self.generator.backward_prop(
                batch_layers_outputs[k],
                discriminator_losses[0].reshape(batch_layers_outputs[k][-1].shape),
            )  # reshape input discr layer to last gener shape
            batch_layers_losses.append(generator_losses)

        self.generator.weights_updating(
            batch_layers_outputs, batch_layers_losses, self.generator_optimizer
        )


    def discriminator_backward_prop(self, real_discr_outs, fake_discr_outs):
        batch_layers_losses = []
        batch_layers_outputs = []

        for k in range(len(real_discr_outs)):

            # real data backprop
            discriminator_losses = self.discriminator.backward_prop(
                real_discr_outs[k],
                self.discriminator_real_loss_func(
                    real_discr_outs[k][-1], self.real_targets
                ),
            )
            batch_layers_losses.append(discriminator_losses)
            batch_layers_outputs.append(real_discr_outs[k])

            # fake data backprop
            discriminator_losses = self.discriminator.backward_prop(
                fake_discr_outs[k],
                self.discriminator_fake_loss_func(
                    fake_discr_outs[k][-1], self.fake_targets
                ),
            )
            batch_layers_losses.append(discriminator_losses)
            batch_layers_outputs.append(fake_discr_outs[k])

        self.discriminator.weights_updating(
            batch_layers_outputs, batch_layers_losses, self.discriminator_optimizer
        )


    def train(
        self,
        real_inputs,
        epochs,
        optimizer_name,
        loss_function_name="binary crossentropy",
        batch_size=10,
        **optimizer_params,
    ):
        real_inputs = np.asfarray(real_inputs)

        batch_num = len(real_inputs) // batch_size
        batches = np.array_split(real_inputs, batch_num)
        noises = np.array_split(
            np.random.normal(
                0, 1, (len(real_inputs), self.generator.topology[0]["inputs num"])
            ),
            batch_num,
        )

        self.generator_loss_metric, self.discriminator_loss_metric = [], []
        generator_loss, discriminator_loss = 0, 0

        self.discriminator_optimizer = self.discriminator.optimizers[optimizer_name]
        self.generator_optimizer = self.generator.optimizers[optimizer_name]

        self.prepare_loss_functions(loss_function_name)
        self.prepare_targets()

        self.discriminator.set_params(optimizer_params, optimizer_name)
        self.generator.set_params(optimizer_params, optimizer_name)

        if len(self.discriminator.weights) == 0:
            self.discriminator.weights_init()
        if len(self.generator.weights) == 0:
            self.generator.weights_init()

        for i in range(epochs):

            tqdm_range = tqdm(range(batch_num))
            for j in tqdm_range:

                # noise_vectors = np.random.normal(0, 1, (len(batches[j]), self.generator.topology[0]['inputs num']))

                generator_outputs, fake_images = self.generator_forward_prop(noises[j])
                (
                    real_discriminator_outputs,
                    real_last_outputs,
                ) = self.discriminator_forward_prop(batches[j])
                (
                    fake_discriminator_outputs,
                    fake_last_outputs,
                ) = self.discriminator_forward_prop(fake_images)

                discriminator_loss = (
                    -np.log(real_last_outputs) - np.log(1 - fake_last_outputs)
                ).mean()
                self.discriminator_loss_metric.append(discriminator_loss)

                generator_loss = -np.log(real_last_outputs).mean()
                self.generator_loss_metric.append(generator_loss)

                self.discriminator_backward_prop(
                    real_discriminator_outputs, fake_discriminator_outputs
                )
                self.generator_backward_prop(generator_outputs)

                tqdm_range.set_description(
                    f"GAN training | optimizer: {optimizer_name} | G loss: {generator_loss:.4f} | D loss: {discriminator_loss:.4f} | epoch {i + 1}/{epochs}"
                )

            if self.epoch_prediction == True:
                one_epoch_outputs = []
                for k in range(self.outputs_num_per_one_epoch):
                    one_epoch_outputs.append(
                        self.predict(self.noises_for_gen_per_epoch[k])
                    )
                self.all_outputs_per_epochs.append(one_epoch_outputs)

        return self.generator_loss_metric, self.discriminator_loss_metric


    def predict_per_epoch(self, mode=False, outputs_per_epoch=0):
        self.epoch_prediction = mode
        self.outputs_num_per_one_epoch = outputs_per_epoch
        self.noises_for_gen_per_epoch = np.random.normal(
            0,
            1,
            (self.outputs_num_per_one_epoch, self.generator.topology[0]["inputs num"]),
        )


    def predict(self, noise_vector):
        generator_outputs = self.generator.forward_prop(noise_vector)

        return generator_outputs[-1]
