import numpy as np
from nnmodel.activations import activations
from nnmodel.exceptions.values_checker import ValuesChecker

class GRU():
    """
    Add GRU layer
    --------------
        Args:
            `units_num` (int): number of neurons in the layer
            `activation` (str) or (`ActivationFunction` class): activation function
            `recurrent_activation` (str) or (`ActivationFunction` class): activation function
            `return_sequences` (bool): if `True`, the output of the layer is a sequence of vectors, else the output is a last vector
            `use_bias` (bool):  `True` if used. `False` if not used
            `cycled_states` (bool): `True` if future iteration init state equals previous iteration last state. `False` if future iteration init state equals 0
        Returns:
            output: data with shape (batch_size, timesteps, units_num)
        References:
            https://arxiv.org/pdf/1612.07778.pdf
            
            https://github.com/erikvdplas/gru-rnn/blob/master/main.py
        
    """

    def __init__(self, units_num, activation = 'tanh', recurrent_activation = 'sigmoid', input_shape = None, return_sequences = False, use_bias = True, cycled_states = True):
        self.units_num   = ValuesChecker.check_integer_variable(units_num, "units_num")
        self.input_shape = ValuesChecker.check_input_dim(input_shape, input_dim = 2)
        self.activation  = ValuesChecker.check_activation(activation, activations)
        self.recurrent_activation  = ValuesChecker.check_activation(recurrent_activation, activations)
        self.use_bias    = ValuesChecker.check_boolean_type(use_bias, "use_bias")

        self.return_sequences = ValuesChecker.check_boolean_type(return_sequences, "cycled_states")
        self.cycled_states = ValuesChecker.check_boolean_type(cycled_states, "return_sequences")

        self.w_z = None
        self.w_r = None
        self.w_h = None

        self.wh_z = None
        self.wh_r = None
        self.wh_h = None

        self.b_z = None
        self.b_r = None
        self.b_h = None


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def build(self):
        self.timesteps, self.input_size = self.input_shape

        self.w_z = np.random.normal(0, pow(self.input_size, -0.5), (self.input_size, self.units_num))
        self.w_r = np.random.normal(0, pow(self.input_size, -0.5), (self.input_size, self.units_num))
        self.w_h = np.random.normal(0, pow(self.input_size, -0.5), (self.input_size, self.units_num))

        self.wh_z = np.random.normal(0, pow(self.units_num, -0.5), (self.units_num, self.units_num))
        self.wh_r = np.random.normal(0, pow(self.units_num, -0.5), (self.units_num, self.units_num))
        self.wh_h = np.random.normal(0, pow(self.units_num, -0.5), (self.units_num, self.units_num))

        self.b_z = np.zeros(self.units_num)
        self.b_r = np.zeros(self.units_num)
        self.b_h = np.zeros(self.units_num)


        self.v_z, self.m_z         = np.zeros_like(self.w_z), np.zeros_like(self.w_z)
        self.v_hat_z, self.m_hat_z = np.zeros_like(self.w_z), np.zeros_like(self.w_z)

        self.v_r, self.m_r         = np.zeros_like(self.w_r), np.zeros_like(self.w_r)
        self.v_hat_r, self.m_hat_r = np.zeros_like(self.w_r), np.zeros_like(self.w_r)

        self.v_h, self.m_h         = np.zeros_like(self.w_h), np.zeros_like(self.w_h)
        self.v_hat_h, self.m_hat_h = np.zeros_like(self.w_h), np.zeros_like(self.w_h)


        self.vh_z, self.mh_z         = np.zeros_like(self.wh_z), np.zeros_like(self.wh_z)
        self.vh_hat_z, self.mh_hat_z = np.zeros_like(self.wh_z), np.zeros_like(self.wh_z)

        self.vh_r, self.mh_r         = np.zeros_like(self.wh_r), np.zeros_like(self.wh_r)
        self.vh_hat_r, self.mh_hat_r = np.zeros_like(self.wh_r), np.zeros_like(self.wh_r)

        self.vh_h, self.mh_h         = np.zeros_like(self.wh_h), np.zeros_like(self.wh_h)
        self.vh_hat_h, self.mh_hat_h = np.zeros_like(self.wh_h), np.zeros_like(self.wh_h)


        self.vb_z, self.mb_z         = np.zeros_like(self.b_z), np.zeros_like(self.b_z)
        self.vb_hat_z, self.mb_hat_z = np.zeros_like(self.b_z), np.zeros_like(self.b_z)

        self.vb_r, self.mb_r         = np.zeros_like(self.b_r), np.zeros_like(self.b_r)
        self.vb_hat_r, self.mb_hat_r = np.zeros_like(self.b_r), np.zeros_like(self.b_r)

        self.vb_h, self.mb_h         = np.zeros_like(self.b_h), np.zeros_like(self.b_h)
        self.vb_hat_h, self.mb_hat_h = np.zeros_like(self.b_h), np.zeros_like(self.b_h)

        self.cprev = None
        self.hprev = None

        self.output_shape = (1, self.units_num) if self.return_sequences == False else (self.timesteps, self.units_num)


    def forward_prop(self, X, training):
        self.input_data = X
        self.batch_size = len(self.input_data) 

        if len(self.input_data.shape) == 2:
            self.input_data = self.input_data[:, np.newaxis, :]
       
        # self.batch_size, self.timesteps = self.input_data.shape[0], self.input_data.shape[1]

        self.update_gates = np.zeros((self.batch_size, self.timesteps, self.units_num))
        self.unactivated_update_gates = np.zeros_like(self.update_gates)

        self.reset_gates = np.zeros((self.batch_size, self.timesteps, self.units_num))
        self.unactivated_reset_gates = np.zeros_like(self.reset_gates)

        self.cell_states = np.zeros((self.batch_size, self.timesteps + 1, self.units_num))
        self.unactivated_cell_states = np.zeros_like(self.cell_states)

        self.hidden_states = np.zeros((self.batch_size, self.timesteps + 1, self.units_num))

        if self.cprev is None: self.cprev = np.zeros_like(self.cell_states[:, 0, :])
        if self.hprev is None: self.hprev = np.zeros_like(self.hidden_states[:, 0, :])

        if self.cycled_states == True and training == True and self.hidden_states[:, -1, :].shape == self.hprev.shape:
            self.cell_states[:, -1, :] = self.cprev.copy()
            self.hidden_states[:, -1, :] = self.hprev.copy()

        #Z_t = recurrent_activation(x_t * W_z + h_t-1 * U_z + b_z)
        #R_t = recurrent_activation(x_t * W_r + h_t-1 * U_r + b_r)
        #C_t = activation(x_t * W_h + R_t * h_t-1 * U_h + b_h)
        #H_t = z_t * H_t-1 + (1 - z_t) * C_t

        
        for t in range(self.timesteps):
            self.unactivated_update_gates[:, t, :] = np.dot(self.input_data[:, t, :], self.w_z) + np.dot(self.hidden_states[:, t-1, :], self.wh_z) + self.b_z
            self.update_gates[:, t, :] = self.recurrent_activation.function(self.unactivated_update_gates[:, t, :])

            self.unactivated_reset_gates[:, t, :] = np.dot(self.input_data[:, t, :], self.w_r) + np.dot(self.hidden_states[:, t-1, :], self.wh_r) + self.b_r
            self.reset_gates[:, t, :]  = self.recurrent_activation.function(self.unactivated_reset_gates[:, t, :])

            self.unactivated_cell_states[:, t, :] = np.dot(self.input_data[:, t, :], self.w_h) + np.dot(self.reset_gates[:, t, :] * self.hidden_states[:, t - 1, :], self.wh_h) + self.b_h
            self.cell_states[:, t, :] = self.activation.function(self.unactivated_cell_states[:, t, :])

            self.hidden_states[:, t, :] = self.update_gates[:, t, :] * self.hidden_states[:, t - 1, :] + (1 - self.update_gates[:, t, :]) * self.cell_states[:, t, :]


        self.cprev = self.cell_states[:, self.timesteps - 1, :].copy()
        self.hprev = self.hidden_states[:, self.timesteps - 1, :].copy()

        if self.return_sequences == False:
           
            return self.hidden_states[:, -2, :]
        else:
            
            return self.hidden_states[:, 0 : -1, :]


    def backward_prop(self, error):
            if self.return_sequences == False:
                temp = np.zeros_like((self.hidden_states))
                temp[:, -2, :] = error
                error = temp

            next_hidden_delta = np.zeros((self.units_num))
            next_cell_delta = np.zeros((self.units_num))

            self.grad_w_z = np.zeros_like(self.w_z)
            self.grad_w_r = np.zeros_like(self.w_r)
            self.grad_w_h = np.zeros_like(self.w_h)

            self.grad_wh_z = np.zeros_like(self.wh_z)
            self.grad_wh_r = np.zeros_like(self.wh_r)
            self.grad_wh_h = np.zeros_like(self.wh_h)

            self.grad_b_z = np.zeros_like(self.b_z)
            self.grad_b_r = np.zeros_like(self.b_r)
            self.grad_b_h = np.zeros_like(self.b_h)

            output_error = np.zeros((self.input_data.shape))

            for t in reversed(range(self.timesteps)):
               
                hidden_delta = error[:, t, :] + next_hidden_delta
               
                cell_gates_delta = hidden_delta * (1 - self.update_gates[:, t, :]) * self.activation.derivative(self.unactivated_cell_states[:, t, :])
                self.grad_w_h   += np.dot(self.input_data[:, t, :].T,  cell_gates_delta)
                self.grad_wh_h  += np.dot(self.hidden_states[:, t - 1, :].T * self.reset_gates[:, t, :].T, cell_gates_delta)
                self.grad_b_h   += cell_gates_delta.sum(axis=0)
            
                reset_gates_delta = np.dot(cell_gates_delta, self.wh_h.T) * self.hidden_states[:, t - 1, :] * self.recurrent_activation.derivative(self.unactivated_reset_gates[:, t, :])
                
                self.grad_w_r   += np.dot(self.input_data[:, t, :].T,  reset_gates_delta)
                self.grad_wh_r  += np.dot(self.hidden_states[:, t - 1, :].T, reset_gates_delta)
                self.grad_b_r   += reset_gates_delta.sum(axis=0)

                update_gates_delta = hidden_delta * (self.hidden_states[:, t - 1, :] - self.cell_states[:, t, :]) * self.recurrent_activation.derivative(self.unactivated_update_gates[:, t, :])
                self.grad_w_z   += np.dot(self.input_data[:, t, :].T,  update_gates_delta)
                self.grad_wh_z  += np.dot(self.hidden_states[:, t - 1, :].T,  update_gates_delta)
                self.grad_b_z   += update_gates_delta.sum(axis=0)

   
                next_hidden_delta =  np.dot(update_gates_delta, self.wh_z.T) + np.dot(reset_gates_delta, self.wh_r.T) + np.dot(cell_gates_delta, self.wh_h.T) * self.reset_gates[:, t, :] + hidden_delta * self.update_gates[:, t, :] 
               

            

                output_error[:, t, :] = np.dot(cell_gates_delta, self.w_h.T) + np.dot(update_gates_delta, self.w_z.T) + np.dot(reset_gates_delta, self.w_r.T)

            return output_error


    def update_weights(self, layer_num):
        
        self.w_h, self.v_h, self.m_h, self.v_hat_h, self.m_hat_h = self.optimizer.update(self.grad_w_h, self.w_h, self.v_h, self.m_h, self.v_hat_h, self.m_hat_h, layer_num)
        self.w_r, self.v_r, self.m_r, self.v_hat_r, self.m_hat_r = self.optimizer.update(self.grad_w_r, self.w_r, self.v_r, self.m_r, self.v_hat_r, self.m_hat_r, layer_num)
        self.w_z, self.v_z, self.m_z, self.v_hat_z, self.m_hat_z = self.optimizer.update(self.grad_w_z, self.w_z, self.v_z, self.m_z, self.v_hat_z, self.m_hat_z, layer_num)
        
        self.wh_h, self.vh_h, self.mh_h, self.vh_hat_h, self.mh_hat_f = self.optimizer.update(self.grad_wh_h, self.wh_h, self.vh_h, self.mh_h, self.vh_hat_h, self.mh_hat_h, layer_num)
        self.wh_r, self.vh_r, self.mh_r, self.vh_hat_r, self.mh_hat_r = self.optimizer.update(self.grad_wh_r, self.wh_r, self.vh_r, self.mh_r, self.vh_hat_r, self.mh_hat_r, layer_num)
        self.wh_z, self.vh_z, self.mh_z, self.vh_hat_z, self.mh_hat_z = self.optimizer.update(self.grad_wh_z, self.wh_z, self.vh_z, self.mh_z, self.vh_hat_z, self.mh_hat_z, layer_num)
        
        if self.use_bias == True:
            self.b_h, self.vb_h, self.mb_h, self.vb_hat_h, self.mb_hat_h  = self.optimizer.update(self.grad_b_h, self.b_h, self.vb_h, self.mb_h, self.vb_hat_h, self.mb_hat_h, layer_num)
            self.b_r, self.vb_r, self.mb_r, self.vb_hat_r, self.mb_hat_r  = self.optimizer.update(self.grad_b_r, self.b_r, self.vb_r, self.mb_r, self.vb_hat_r, self.mb_hat_r, layer_num)
            self.b_z, self.vb_z, self.mb_z, self.vb_hat_z, self.mb_hat_z  = self.optimizer.update(self.grad_b_z, self.b_z, self.vb_z, self.mb_z, self.vb_hat_z, self.mb_hat_z, layer_num)

    def get_grads(self):
        return self.grad_w_h, self.grad_w_r, self.grad_w_z, self.grad_wh_h, self.grad_wh_r, self.grad_wh_z, self.grad_b_h, self.grad_b_r, self.grad_b_z

    def set_grads(self, grads):
        self.grad_w_h, self.grad_w_r, self.grad_w_z, self.grad_wh_h, self.grad_wh_r, self.grad_wh_z, self.grad_b_h, self.grad_b_r, self.grad_b_z = grads

            