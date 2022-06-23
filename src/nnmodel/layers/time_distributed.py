import numpy as np


class TimeDistributed:
    def __init__(self, layer):
        self.layer = layer
        self.input_shape = None

        self.optimizer = None

        self.grads_w = 0
        self.grads_b = 0

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def build(self):
        self.timesteps = self.input_shape[0]
        self.layer.input_shape = self.input_shape
        self.layer.build()
        self.layer.set_optimizer(self.optimizer)

        if len(self.layer.output_shape) == 2:
            self.output_shape = (self.timesteps, *self.layer.output_shape[1:])
        elif len(self.layer.output_shape) == 3:
            self.output_shape = (self.timesteps, *self.layer.output_shape)

    def forward_prop(self, X, training):
        self.input_data = X

        if len(self.input_data.shape) == 2:
            self.input_data = self.input_data[:, np.newaxis, :]

        self.batch_size = len(self.input_data)

        self.forward_states = np.zeros((self.batch_size, *self.output_shape))

        for t in range(self.timesteps):
            self.forward_states[:, t, ...] = self.layer.forward_prop(
                X[:, t, ...], training
            ).copy()

        return self.forward_states

    def backward_prop(self, error):

        self.grads_w = 0
        self.grads_b = 0

        self.backward_states = np.zeros((self.batch_size, *self.input_shape))
        for t in range(self.timesteps):  # reversed
            self.layer.input_data = self.forward_states[:, t, ...]
            self.backward_states[:, t, ...] = self.layer.backward_prop(
                error[:, t, ...]
            ).copy()

            self.grads_w += self.layer.grad_w.copy()
            self.grads_b += self.layer.grad_b.copy()
        return self.backward_states

    def update_weights(self, layer_num):

        (
            self.layer.w,
            self.layer.v,
            self.layer.m,
            self.layer.v_hat,
            self.layer.m_hat,
        ) = self.optimizer.update(
            self.grads_w,
            self.layer.w,
            self.layer.v,
            self.layer.m,
            self.layer.v_hat,
            self.layer.m_hat,
            layer_num,
        )
        if self.layer.use_bias == True:
            (
                self.layer.b,
                self.layer.vb,
                self.layer.mb,
                self.layer.vb_hat,
                self.layer.mb_hat,
            ) = self.optimizer.update(
                self.grads_b,
                self.layer.b,
                self.layer.vb,
                self.layer.mb,
                self.layer.vb_hat,
                self.layer.mb_hat,
                layer_num,
            )
