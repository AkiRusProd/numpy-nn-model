import numpy as np


class TimeDistributed():
    #TODO 
    #ADD restriction on the recurrent layers using
    def __init__(self, layer):
        self.layer = layer
        self.input_shape = None

        self.optimizer = None

        self.grads_w = 0
        self.grads_b = 0

        self.grads_gamma = 0
        self.grads_beta = 0

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
       
        if hasattr(self.layer, 'set_optimizer'):
                self.layer.set_optimizer(self.optimizer)
        

    def build(self):
        self.timesteps = self.input_shape[0]

        if len(self.input_shape) == 2:
            self.layer.input_shape = (1, self.input_shape[1]) #set to layer only one sample; not with timesteps
        else:
            self.layer.input_shape = self.input_shape[1:]
        self.layer.build()

        
        if len(self.layer.output_shape) == 2:
            self.output_shape = (self.timesteps, *self.layer.output_shape[1:])
        elif len(self.layer.output_shape) == 3:
            self.output_shape = (self.timesteps, *self.layer.output_shape)

        self._backward_prop = self._default_backward_prop
        self._forward_prop = self._default_forward_prop
        self._update_weights = self._default_update_weights

        if self.layer.__class__.__name__ == "BatchNormalization":
           self._backward_prop = self._batchnorm_backward_prop
           self._forward_prop  = self._batchnorm_forward_prop
           self._update_weights = self._batchnorm_update_weights

    def forward_prop(self, X, training):

        return self._forward_prop(X, training)

    def _default_forward_prop(self, X, training):
        self.input_data = X

        if len(self.input_data.shape) == 2:
            self.input_data = self.input_data[:, np.newaxis, :]

        self.batch_size = len(self.input_data)
        
        self.forward_states = np.zeros((self.batch_size, *self.output_shape))
        
        for t in range(self.timesteps):
            self.forward_states[:, t, ...] = self.layer.forward_prop(self.input_data[:, t, ...], training).copy()

        
        return self.forward_states

    def _batchnorm_forward_prop(self, X, training):
        self.input_data = X
        self.X_centered = np.zeros_like(self.input_data)
        self.stddev_inv = np.zeros_like(self.input_data)

        if  self.layer.moving_mean is None:
            self.moving_var = np.zeros((self.input_data.shape[1:]))
            self.moving_mean = np.zeros((self.input_data.shape[1:]))

        if len(self.input_data.shape) == 2:
            self.input_data = self.input_data[:, np.newaxis, :]

        self.batch_size = len(self.input_data)
        
        self.forward_states = np.zeros((self.batch_size, *self.output_shape))
        
        for t in range(self.timesteps):

            if self.layer.moving_mean is not None:
                self.layer.moving_var  = self.moving_var[t, :]
                self.layer.moving_mean = self.moving_mean[t, :]

            self.forward_states[:, t, ...] = self.layer.forward_prop(self.input_data[:, t, ...], training).copy()

            if training == True:
                self.moving_var[t, :]  = self.layer.moving_var
                self.moving_mean[t, :] = self.layer.moving_mean

                self.X_centered[:, t, ...] = self.layer.X_centered
                self.stddev_inv[:, t, ...] = self.layer.stddev_inv
        
        return self.forward_states


    def backward_prop(self, error):

        return self._backward_prop(error)


    def _default_backward_prop(self, error):
        self.grads_w = 0
        self.grads_b = 0

        self.backward_states = np.zeros((self.batch_size, *self.input_shape))
        for t in (range(self.timesteps)):#reversed

            self.layer.input_data = self.input_data[:, t, ...]
            self.layer.output_data = self.forward_states[:, t, ...]
            self.backward_states[:, t, ...] = self.layer.backward_prop(error[:, t, ...]).copy()

            if hasattr(self.layer, 'update_weights'): # temporary solution
                self.grads_w += self.layer.grad_w.copy()
                self.grads_b += self.layer.grad_b.copy()

        return self.backward_states


    def _batchnorm_backward_prop(self, error):
        self.grads_gamma = 0
        self.grads_beta = 0

        self.backward_states = np.zeros((self.batch_size, *self.input_shape))
        for t in (range(self.timesteps)):
            self.layer.X_centered = self.X_centered[:, t, ...]
            self.layer.stddev_inv = self.stddev_inv[:, t, ...]

            self.backward_states[:, t, ...] = self.layer.backward_prop(error[:, t, ...]).copy()
            self.grads_gamma += self.layer.grad_gamma.copy()
            self.grads_beta += self.layer.grad_beta.copy()
        
        return self.backward_states

    def update_weights(self, layer_num):

        self._update_weights(layer_num)

    def _default_update_weights(self, layer_num):
        
        if hasattr(self.layer, 'update_weights'):

            self.layer.grad_w = self.grads_w
            self.layer.grad_b = self.grads_b
            
            self.layer.update_weights(layer_num)


    def _batchnorm_update_weights(self, layer_num):

        self.layer.grad_gamma = self.grads_gamma
        self.layer.grad_beta = self.grads_beta

        self.layer.update_weights(layer_num)
    
           
