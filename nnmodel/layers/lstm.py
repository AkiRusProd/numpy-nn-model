import numpy as np
from nnmodel.activations import activations
from nnmodel.exceptions.values_checker import ValuesChecker

class LSTM():
    #https://en.wikipedia.org/wiki/Long_short-term_memory
    #https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/
    #https://arunmallya.github.io/writeups/nn/lstm/index.html#/
    def __init__(self, units_num, activation = 'tanh', recurrent_activation = 'sigmoid', input_shape = None, return_sequences = False, use_bias = True, cycled_states = False):
        self.units_num   = ValuesChecker.check_integer_variable(units_num, "units_num")
        self.input_shape = ValuesChecker.check_input_dim(input_shape, input_dim = 2)
        self.activation  = ValuesChecker.check_activation(activation, activations)
        self.recurrent_activation  = ValuesChecker.check_activation(recurrent_activation, activations)
        self.use_bias    = ValuesChecker.check_boolean_type(use_bias, "use_bias")

        self.return_sequences = ValuesChecker.check_boolean_type(return_sequences, "cycled_states")
        self.cycled_states = ValuesChecker.check_boolean_type(cycled_states, "return_sequences")

        self.w_f = None
        self.w_i = None
        self.w_o = None
        self.w_c = None

        self.wh_f = None
        self.wh_i = None
        self.wh_o = None
        self.wh_c = None

        self.b_f = None
        self.b_i = None
        self.b_o = None
        self.b_c = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def build(self):
        self.timesteps, self.input_size = self.input_shape

        self.w_f = np.random.normal(0, pow(self.input_size, -0.5), (self.input_size, self.units_num))
        self.w_i = np.random.normal(0, pow(self.input_size, -0.5), (self.input_size, self.units_num))
        self.w_o = np.random.normal(0, pow(self.input_size, -0.5), (self.input_size, self.units_num))
        self.w_c = np.random.normal(0, pow(self.input_size, -0.5), (self.input_size, self.units_num))

        self.wh_f = np.random.normal(0, pow(self.units_num, -0.5), (self.units_num, self.units_num))
        self.wh_i = np.random.normal(0, pow(self.units_num, -0.5), (self.units_num, self.units_num))
        self.wh_o = np.random.normal(0, pow(self.units_num, -0.5), (self.units_num, self.units_num))
        self.wh_c = np.random.normal(0, pow(self.units_num, -0.5), (self.units_num, self.units_num))

        self.b_f = np.zeros(self.units_num)
        self.b_i = np.zeros(self.units_num)
        self.b_o = np.zeros(self.units_num)
        self.b_c = np.zeros(self.units_num)


        self.v_f, self.m_f         = np.zeros_like(self.w_f), np.zeros_like(self.w_f) # optimizers params
        self.v_hat_f, self.m_hat_f = np.zeros_like(self.w_f), np.zeros_like(self.w_f) # optimizers params

        self.v_i, self.m_i         = np.zeros_like(self.w_i), np.zeros_like(self.w_i) # optimizers params
        self.v_hat_i, self.m_hat_i = np.zeros_like(self.w_i), np.zeros_like(self.w_i) # optimizers params

        self.v_o, self.m_o         = np.zeros_like(self.w_o), np.zeros_like(self.w_o) # optimizers params
        self.v_hat_o, self.m_hat_o = np.zeros_like(self.w_o), np.zeros_like(self.w_o) # optimizers params

        self.v_c, self.m_c         = np.zeros_like(self.w_c), np.zeros_like(self.w_c) # optimizers params
        self.v_hat_c, self.m_hat_c = np.zeros_like(self.w_c), np.zeros_like(self.w_c) # optimizers params


        self.vh_f, self.mh_f         = np.zeros_like(self.wh_f), np.zeros_like(self.wh_f) # optimizers params
        self.vh_hat_f, self.mh_hat_f = np.zeros_like(self.wh_f), np.zeros_like(self.wh_f) # optimizers params

        self.vh_i, self.mh_i         = np.zeros_like(self.wh_i), np.zeros_like(self.wh_i) # optimizers params
        self.vh_hat_i, self.mh_hat_i = np.zeros_like(self.wh_i), np.zeros_like(self.wh_i) # optimizers params

        self.vh_o, self.mh_o         = np.zeros_like(self.wh_o), np.zeros_like(self.wh_o) # optimizers params
        self.vh_hat_o, self.mh_hat_o = np.zeros_like(self.wh_o), np.zeros_like(self.wh_o) # optimizers params

        self.vh_c, self.mh_c         = np.zeros_like(self.wh_c), np.zeros_like(self.wh_c) # optimizers params
        self.vh_hat_c, self.mh_hat_c = np.zeros_like(self.wh_c), np.zeros_like(self.wh_c) # optimizers params


        self.vb_f, self.mb_f         = np.zeros_like(self.b_f), np.zeros_like(self.b_f) # optimizers params
        self.vb_hat_f, self.mb_hat_f = np.zeros_like(self.b_f), np.zeros_like(self.b_f) # optimizers params

        self.vb_i, self.mb_i         = np.zeros_like(self.b_i), np.zeros_like(self.b_i) # optimizers params
        self.vb_hat_i, self.mb_hat_i = np.zeros_like(self.b_i), np.zeros_like(self.b_i) # optimizers params

        self.vb_o, self.mb_o         = np.zeros_like(self.b_o), np.zeros_like(self.b_o) # optimizers params
        self.vb_hat_o, self.mb_hat_o = np.zeros_like(self.b_o), np.zeros_like(self.b_o) # optimizers params

        self.vb_c, self.mb_c         = np.zeros_like(self.b_c), np.zeros_like(self.b_c) # optimizers params
        self.vb_hat_c, self.mb_hat_c = np.zeros_like(self.b_c), np.zeros_like(self.b_c) # optimizers params

        self.cprev = None
        self.hprev = None

        self.output_shape = (1, self.units_num) if self.return_sequences == False else (self.timesteps, self.units_num)

    def forward_prop(self, X, training):
        self.input_data = X
        self.batch_size = len(self.input_data) 

        if len(self.input_data.shape) == 2:
            self.input_data = self.input_data[:, np.newaxis, :]
       
        # self.batch_size, self.timesteps = self.input_data.shape[0], self.input_data.shape[1]

        self.forget_gates = np.zeros((self.batch_size, self.timesteps, self.units_num))
        self.unactivated_forget_gates = np.zeros_like(self.forget_gates)

        self.input_gates = np.zeros((self.batch_size, self.timesteps, self.units_num))
        self.unactivated_input_gates = np.zeros_like(self.input_gates)

        self.output_gates = np.zeros((self.batch_size, self.timesteps, self.units_num))
        self.unactivated_output_gates = np.zeros_like(self.output_gates)

        self.cell_gates = np.zeros((self.batch_size, self.timesteps, self.units_num))
        self.unactivated_cell_gates = np.zeros_like(self.cell_gates)
        
        self.cell_states = np.zeros((self.batch_size, self.timesteps + 1, self.units_num))
        self.hidden_states = np.zeros((self.batch_size, self.timesteps + 1, self.units_num))

        if self.cprev is None: self.cprev = np.zeros_like(self.cell_states[:, 0, :])
        if self.hprev is None: self.hprev = np.zeros_like(self.hidden_states[:, 0, :])

        if self.cycled_states == True and training == True:
            self.cell_states[:, -1, :] = self.cprev.copy()
            self.hidden_states[:, -1, :] = self.hprev.copy()

        #f_t = recurrent_activation(self.input_data @ self.w_f + hs_t-1 @ self.wh_f + self.b_f)
        #i_t = recurrent_activation(self.input_data @ self.w_i + hs_t-1 @ self.wh_i + self.b_i)
        #o_t = recurrent_activation(self.input_data @ self.w_o + hs_t-1 @ self.wh_o + self.b_o)
        #c_t =           activation(self.input_data @ self.w_c + hs_t-1 @ self.wh_c + self.b_c)
        #cs_t = f_t * cs_t-1 + i_t * c_t
        #hs_t = o_t * recurrent_activation(cs_t)
        
        for t in range(self.timesteps):
            self.unactivated_forget_gates[:, t, :] = np.dot(self.input_data[:, t, :], self.w_f) + np.dot(self.hidden_states[:, t-1, :], self.wh_f) + self.b_f
            self.forget_gates[:, t, :] = self.recurrent_activation.function(self.unactivated_forget_gates[:, t, :])

            self.unactivated_input_gates[:, t, :] = np.dot(self.input_data[:, t, :], self.w_i) + np.dot(self.hidden_states[:, t-1, :], self.wh_i) + self.b_i
            self.input_gates[:, t, :]  = self.recurrent_activation.function(self.unactivated_input_gates[:, t, :])

            self.unactivated_output_gates[:, t, :] = np.dot(self.input_data[:, t, :], self.w_o) + np.dot(self.hidden_states[:, t-1, :], self.wh_o) + self.b_o
            self.output_gates[:, t, :] = self.recurrent_activation.function(self.unactivated_output_gates[:, t, :])

            self.unactivated_cell_gates[:, t, :] = np.dot(self.input_data[:, t, :], self.w_c) + np.dot(self.hidden_states[:, t-1, :], self.wh_c) + self.b_c
            self.cell_gates[:, t, :]   = self.activation.function(self.unactivated_cell_gates[:, t, :])

            self.cell_states[:, t, :] = self.forget_gates[:, t, :] * self.cell_states[:, t-1, :] + self.input_gates[:, t, :] * self.cell_gates[:, t, :]
            self.hidden_states[:, t, :] = self.output_gates[:, t, :] * self.activation.function(self.cell_states[:, t, :])


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

        self.grad_w_f = np.zeros_like(self.w_f)
        self.grad_w_i = np.zeros_like(self.w_i)
        self.grad_w_o = np.zeros_like(self.w_o)
        self.grad_w_c = np.zeros_like(self.w_c)

        self.grad_wh_f = np.zeros_like(self.wh_f)
        self.grad_wh_i = np.zeros_like(self.wh_i)
        self.grad_wh_o = np.zeros_like(self.wh_o)
        self.grad_wh_c = np.zeros_like(self.wh_f)

        self.grad_b_f = np.zeros_like(self.b_f)
        self.grad_b_i = np.zeros_like(self.b_i)
        self.grad_b_o = np.zeros_like(self.b_o)
        self.grad_b_c = np.zeros_like(self.b_c)

        output_error = np.zeros((self.input_data.shape))

        for t in reversed(range(self.timesteps)):
            # hidden_delta = self.activation.derivative(self.cell_states[:, t, :]) * self.output_gates[:, t, :] * error[:, t, :] + next_hidden_delta #()?
            
            # output_gates_delta =  error[:, t, :] * self.activation.function(self.cell_states[:, t, :]) * self.recurrent_activation.derivative(self.output_gates[:, t, :])
            hidden_delta = error[:, t, :] + next_hidden_delta
            cell_delta = hidden_delta * self.output_gates[:, t, :] * self.activation.derivative(self.cell_states[:, t, :]) + next_cell_delta#✓

            output_gates_delta = hidden_delta * self.activation.function(self.cell_states[:, t, :]) * self.recurrent_activation.derivative(self.unactivated_output_gates[:, t, :]) #✓
            self.grad_w_o   += np.dot(self.input_data[:, t, :].T,  output_gates_delta)
            self.grad_wh_o  += np.dot(self.hidden_states[:, t - 1, :].T, output_gates_delta)
            self.grad_b_o   += output_gates_delta.sum(axis=0)
            # dX_o = np.dot(np.dot(output_gates_delta, self.wh_o.T), self.w_o.T) #unverified solution

            # forget_gates_delta = (hidden_delta * self.cell_states[:, t - 1, :]) * self.recurrent_activation.derivative(self.forget_gates[:, t, :])
            forget_gates_delta = (cell_delta * self.cell_states[:, t - 1, :]) * self.recurrent_activation.derivative(self.unactivated_forget_gates[:, t, :])
            self.grad_w_f   += np.dot(self.input_data[:, t, :].T,  forget_gates_delta)
            self.grad_wh_f  += np.dot(self.hidden_states[:, t - 1, :].T, forget_gates_delta)
            self.grad_b_f   += forget_gates_delta.sum(axis=0)
            # dX_f = np.dot(np.dot(forget_gates_delta, self.wh_f.T), self.w_f.T) #unverified solution

            # input_gates_delta = (hidden_delta * self.cell_gates[:, t, :]) * self.recurrent_activation.derivative(self.input_gates[:, t, :])
            input_gates_delta = (cell_delta * self.cell_gates[:, t, :]) * self.recurrent_activation.derivative(self.unactivated_input_gates[:, t, :])
            self.grad_w_i   += np.dot(self.input_data[:, t, :].T,  input_gates_delta)
            self.grad_wh_i  += np.dot(self.hidden_states[:, t - 1, :].T, input_gates_delta)
            self.grad_b_i   += input_gates_delta.sum(axis=0)
            # dX_i = np.dot(np.dot(input_gates_delta, self.wh_i.T), self.w_i.T) #unverified solution

            # cell_gates_delta = (hidden_delta * self.input_gates[:, t, :]) * self.activation.derivative(self.cell_gates[:, t, :])
            cell_gates_delta = (cell_delta * self.input_gates[:, t, :]) * self.activation.derivative(self.unactivated_cell_gates[:, t, :])
            self.grad_w_c   += np.dot(self.input_data[:, t, :].T,  cell_gates_delta)
            self.grad_wh_c  += np.dot(self.hidden_states[:, t - 1, :].T, cell_gates_delta)
            self.grad_b_c   += cell_gates_delta.sum(axis=0)
            # dX_c = np.dot(np.dot(cell_gates_delta, self.wh_c.T), self.w_c.T) #unverified solution

            #next_hidden_delta = hidden_delta #* self.forget_gates[:, t, :]
            next_hidden_delta = np.dot(cell_gates_delta, self.wh_c.T) + np.dot(input_gates_delta, self.wh_i.T) + np.dot(forget_gates_delta, self.wh_f.T) + np.dot(output_gates_delta, self.wh_o.T)
            next_cell_delta = cell_delta * self.forget_gates[:, t, :]

            
            # output_error[:, t, :] = dX_o + \
            #                         dX_f + \
            #                         dX_i + \
            #                         dX_c

            output_error[:, t, :] = np.dot(cell_gates_delta, self.w_c.T) + np.dot(input_gates_delta, self.w_i.T) + np.dot(forget_gates_delta, self.w_f.T) + np.dot(output_gates_delta, self.w_o.T)

        return output_error


    
    def update_weights(self, layer_num):
        self.w_f, self.v_f, self.m_f, self.v_hat_f, self.m_hat_f = self.optimizer.update(self.grad_w_f, self.w_f, self.v_f, self.m_f, self.v_hat_f, self.m_hat_f, layer_num)
        self.w_i, self.v_i, self.m_i, self.v_hat_i, self.m_hat_i = self.optimizer.update(self.grad_w_i, self.w_i, self.v_i, self.m_i, self.v_hat_i, self.m_hat_i, layer_num)
        self.w_o, self.v_o, self.m_o, self.v_hat_o, self.m_hat_o = self.optimizer.update(self.grad_w_o, self.w_o, self.v_o, self.m_o, self.v_hat_o, self.m_hat_o, layer_num)
        self.w_c, self.v_c, self.m_c, self.v_hat_c, self.m_hat_c = self.optimizer.update(self.grad_w_c, self.w_c, self.v_c, self.m_c, self.v_hat_c, self.m_hat_c, layer_num)
        
        self.wh_f, self.vh_f, self.mh_f, self.vh_hat_f, self.mh_hat_f = self.optimizer.update(self.grad_wh_f, self.wh_f, self.vh_f, self.mh_f, self.vh_hat_f, self.mh_hat_f, layer_num)
        self.wh_i, self.vh_i, self.mh_i, self.vh_hat_i, self.mh_hat_i = self.optimizer.update(self.grad_wh_i, self.wh_i, self.vh_i, self.mh_i, self.vh_hat_i, self.mh_hat_i, layer_num)
        self.wh_o, self.vh_o, self.mh_o, self.vh_hat_o, self.mh_hat_o = self.optimizer.update(self.grad_wh_o, self.wh_o, self.vh_o, self.mh_o, self.vh_hat_o, self.mh_hat_o, layer_num)
        self.wh_c, self.vh_c, self.mh_c, self.vh_hat_c, self.mh_hat_c = self.optimizer.update(self.grad_wh_c, self.wh_c, self.vh_c, self.mh_c, self.vh_hat_c, self.mh_hat_c, layer_num)
        
        if self.use_bias == True:
            self.b_f, self.vb_f, self.mb_f, self.vb_hat_f, self.mb_hat_f  = self.optimizer.update(self.grad_b_f, self.b_f, self.vb_f, self.mb_f, self.vb_hat_f, self.mb_hat_f, layer_num)
            self.b_i, self.vb_i, self.mb_i, self.vb_hat_i, self.mb_hat_i  = self.optimizer.update(self.grad_b_i, self.b_i, self.vb_i, self.mb_i, self.vb_hat_i, self.mb_hat_i, layer_num)
            self.b_o, self.vb_o, self.mb_o, self.vb_hat_o, self.mb_hat_o  = self.optimizer.update(self.grad_b_o, self.b_o, self.vb_o, self.mb_o, self.vb_hat_o, self.mb_hat_o, layer_num)
            self.b_c, self.vb_c, self.mb_c, self.vb_hat_c, self.mb_hat_c  = self.optimizer.update(self.grad_b_c, self.b_c, self.vb_c, self.mb_c, self.vb_hat_c, self.mb_hat_c, layer_num)