from tqdm import tqdm
from numba import njit
import numpy as np
import os





class Model:
    def __init__(self):
        self.topology = []
        self.weights = []

        self.v, self.m = [], [] #optimizers params
        self.v_hat, self.m_hat = [], [] #optimizers params

        self.alpha = None
        self.beta = None
        self.beta2 = None
        self.epsilon = None

        self.optimizers = {'BGD':self.BGD_optimizer, 
                           'Momentum': self.Momentum_optimizer, 
                           'RMSProp': self.RMSProp_optimizer, 
                           'Adam': self.Adam_optimizer, 
                           'Nadam': self.Nadam_optimizer
                           }

        self.loss_functions = {'MSE': lambda output, target: -(target-output),
                               'binary crossentropy': lambda output, target: -target/output + (1 - target)/(1 - output),
                               'categorical crossentropy': lambda output, target: target/output,
                                None: lambda output, target = None: output,
                                }

        self.loss_functions_metrics = {'MSE': lambda output, target: np.power(target-output, 2),
                               'binary crossentropy': lambda output, target: -target * np.log(output) + (1 - target) * np.log(1 - output),
                               'categorical crossentropy': lambda output, target: target * np.log(output),
                                None: lambda output, target = None: output,
                                }

    def activation_func(self,output, name):

        if name == 'Sigmoid':
            return 1/(1.+np.exp(-output))
        elif name == 'Tanh':
            return np.tanh(output)
        elif name == 'ReLU':
            return np.maximum(0, output)
        elif name == 'Leaky ReLU':
            return np.where(output <= 0, 0.01 * output, output)
        elif name == 'ELU':
            mu = 1
            return np.where(output <= 0,  mu * (np.exp(output) - 1), output)
        elif name == 'GELU':
            return 0.5 * output * (1 + np.tanh(np.sqrt(2/np.pi) * (output + 0.044715 * np.power(output,3))))    
        elif name == None:
            return output
        else:
            raise SystemExit(f'Activation function with name "{name}" not found')

 
    def act_func_der(self, output, name): #derivatives for gradient computation

        if name == 'Sigmoid':
            return output * (1. - output)
        elif name == 'Tanh':
            return 1. - np.power(output, 2)
        elif name == 'ReLU':
            return np.where(output <= 0, 0, 1)
        elif name == 'Leaky ReLU':
            return np.where(output <= 0, 0.01, 1)
        elif name == 'ELU':
            mu = 1
            return np.where(output <= 0, output + mu, 1)
        elif name == 'GELU':
            sech = lambda x: 2/(np.exp(x) + np.exp(-x))
            return 0.5 * np.tanh(0.0356774 * np.power(output,3) + 0.797885 * output) + (0.0535161 * np.power(output,3) + 0.398942 * output) * np.power(sech(0.0356774 * np.power(output,3) + 0.797885 * output),2) + 0.5
        elif name == None:
            return np.ones((output.shape))



    def summary(self):

        print('----------------------------------------------------------------')
        print ("{:<13} {:<15} {:<20} {:<20}".format('Layer', 'Shape', 'Activation func', 'Weights Shape'))
        print('================================================================')

        weights_shapes = []
        total_weights = 0

        if self.topology[1]['type'] == 'Dense':
            weights_shapes.append(f"({self.topology[0]['inputs num']}x{self.topology[1]['neurons num']})")

            total_weights += self.topology[0]['inputs num'] * self.topology[1]['neurons num']
        elif self.topology[1]['type'] == 'Conv2d':
            weights_shapes.append(f"({self.topology[1]['kernels num']}x{self.topology[1]['kernels size']}x{self.topology[1]['kernels size']})")

            total_weights += self.topology[1]['kernels num'] * self.topology[1]['kernels size'] * self.topology[1]['kernels size']


        for k in range(1, len(self.topology)-1):
            if self.topology[k+1]['type'] == 'Dense':

                if self.topology[k]['type'] == 'Dense':
                    weights_shapes.append(f"({self.topology[k]['neurons num']}x{self.topology[k+1]['neurons num']})")

                    total_weights += self.topology[k]['neurons num'] * self.topology[k+1]['neurons num']

                elif self.topology[k]['pooling type'] == 'MaxPooling' or self.topology[k]['pooling type'] == 'AvgPooling':
                    weights_shapes.append(f"({self.topology[k]['kernels num'] * self.topology[k]['pool size'] * self.topology[k]['pool size']}x{self.topology[k+1]['neurons num']})")

                    total_weights += self.topology[k]['kernels num'] * self.topology[k]['pool size'] * self.topology[k]['pool size'] * self.topology[k+1]['neurons num']
                                                                  
                elif self.topology[k]['type'] == 'Conv2d':
                    weights_shapes.append(f"({self.topology[k]['kernels num'] * self.topology[k]['conv size'] * self.topology[k]['conv size']}x{self.topology[k+1]['neurons num']})")

                    total_weights += self.topology[k]['kernels num'] * self.topology[k]['conv size'] * self.topology[k]['conv size'] * self.topology[k+1]['neurons num']
                                                             
            elif self.topology[k+1]['type'] == 'Conv2d':
                weights_shapes.append(f"({self.topology[k+1]['kernels num']}x{self.topology[k+1]['kernels size']}x{self.topology[k+1]['kernels size']})")

                total_weights += self.topology[k+1]['kernels num'] * self.topology[k+1]['kernels size'] * self.topology[k+1]['kernels size']
        
        i = 0

        for k in range(len(self.topology)):
            if self.topology[k]['type'] == 'Input':

                if self.topology[k+1]['type'] == 'Conv2d':
                    if self.topology[k]['input type'] == '2d':
                        print("{:<13} {:<15} {:<20} {:<20}".format(self.topology[k]['type'], f"({self.topology[k]['inputs num']}x{self.topology[k+1]['init input size']}x{self.topology[k+1]['init input size']})", '-', '-'))
                    elif self.topology[k]['input type'] == '1d':
                        print("{:<13} {:<15} {:<20} {:<20}".format(self.topology[k]['type'], f"({self.topology[k]['inputs num']})", '-', '-'))

                else: 
                    print("{:<13} {:<15} {:<20} {:<20}".format(self.topology[k]['type'], f"({self.topology[k]['inputs num']})", '-', '-'))


            if self.topology[k]['type'] == 'Dense':
                print("{:<13} {:<15} {:<20} {:<20}".format(self.topology[k]['type'], f"({self.topology[k]['neurons num']})", str(self.topology[k]['activation func']), weights_shapes[i]))
                
                if self.topology[k]['dropout rate'] != None: 
                    print("{:<13} {:<15} {:<20} {:<20}".format(f"Dropout {self.topology[k]['dropout rate']}", '-', '-', '-' ))
                
                i += 1

            if self.topology[k]['type'] == 'Conv2d':      
                print("{:<13} {:<15} {:<20} {:<20}".format(self.topology[k]['type'], f"({self.topology[k]['kernels num']}x{self.topology[k]['conv size']}x{self.topology[k]['conv size']})", str(self.topology[k]['activation func']), weights_shapes[i]))

                if self.topology[k]['pooling type'] != None: 
                    print("{:<13} {:<15} {:<20} {:<20}".format(self.topology[k]['pooling type'], f"({self.topology[k]['kernels num']}x{self.topology[k]['pool size']}x{self.topology[k]['pool size']})", '-', '-' ))

                i += 1

        print('================================================================')
        print(f'Total learnable parameters: {total_weights}')
        print('----------------------------------------------------------------')


    def save(self, name):
        
        try:
            os.mkdir(name)
        except: pass

        np.save(f'{name}/topology.npy', np.array(self.topology, dtype = object))
        np.savez(f'{name}/weights.npz', *self.weights)
        np.savez(f'{name}/optimizer_params.npz', v = self.v, m = self.m, v_hat = self.v_hat, m_hat = self.m_hat)


    def load(self, name):
        
        self.topology = np.load(f'{name}/topology.npy', allow_pickle = True)

        container = np.load(f'{name}/weights.npz',)
        self.weights = [container[key] for key in container]

        optimizer_params = np.load(f'{name}/optimizer_params.npz', allow_pickle = True)
        
        self.v = optimizer_params['v']
        self.m = optimizer_params['m']
        self.v_hat = optimizer_params['v_hat']
        self.m_hat = optimizer_params['m_hat']

        


    def add_input_layer(self, inputs_number, input_type):

        if not( input_type == '1d' or input_type == '2d'): raise SystemExit('Please input "1d" or "2d" input types')
        
        self.topology.append({f'type': 'Input','inputs num': inputs_number, 'input type': input_type, 'dropout rate': None, 'activation func': None, })


    def add_dense_layer(self, neurons_number, activation_func = None, bias = 0):
        
        self.topology.append({'type': 'Dense', 'neurons num': neurons_number, f'activation func': activation_func, 'bias': bias, 'dropout rate': None})

        self.v.append(0); self.v_hat.append(0)
        self.m.append(0); self.m_hat.append(0)

    def add_dropout_layer(self, rate):
        if self.topology[-1]['type'] == 'Dense' or self.topology[-1]['type'] == 'Input':
            self.topology[-1]['dropout rate'] = rate
        else:
            raise SystemExit('Dropout layer must come after Dense layer')



    def add_conv2d_layer(self,kernels_number, kernels_size, input_size, transposing_stride = 1, upsampling_scale_factor = 1, padding = 0, activation_func = None, bias = 0):

        previous_kernels_number = None
        init_input_size = input_size
            
              
        if self.topology[-1]['type'] == 'Dense':
            previous_kernels_number = self.topology[-1]['neurons num']  // init_input_size ** 2
        elif self.topology[-1]['type'] == 'Conv2d':
            previous_kernels_number = self.topology[-1]['kernels num']
        elif self.topology[-1]['type'] == 'Input' and self.topology[-1]['input type']=='2d':
            previous_kernels_number = self.topology[-1]['inputs num']
        elif self.topology[-1]['type'] == 'Input' and self.topology[-1]['input type']=='1d': 
            previous_kernels_number = self.topology[-1]['inputs num']  // init_input_size ** 2


        if previous_kernels_number <= kernels_number:
            kernel_per_input = kernels_number // previous_kernels_number
        else:
            kernel_per_input = previous_kernels_number // kernels_number
        
        if upsampling_scale_factor != 1:
            input_size = input_size * upsampling_scale_factor

        if transposing_stride != 1:
            input_size = input_size * transposing_stride - (transposing_stride - 1)

        if padding != 0:
            input_size = 2 * padding + input_size

        # input_size = 2 * padding + stride * init_input_size * scale_factor - 1
        conv_size = input_size - kernels_size + 1
          
        self.topology.append({'type': 'Conv2d', 'kernels num': kernels_number, 'previous kernels num': previous_kernels_number,  
                              'kernels size': kernels_size, 'kernel per input':kernel_per_input, 'conv size': conv_size, 'input size': input_size, 'init input size': init_input_size, 
                              'activation func': activation_func, 'bias': bias, 'padding': padding, 'stride': transposing_stride, 'upsampling scale factor': upsampling_scale_factor,
                              
                              'pooling type': None, 'block size': None, 'pool size': None,'pool inds': None})
        
        self.v.append(np.zeros((self.topology[-1]['kernels num'], self.topology[-1]['kernels size'], self.topology[-1]['kernels size'])))
        self.m.append(np.zeros((self.topology[-1]['kernels num'], self.topology[-1]['kernels size'], self.topology[-1]['kernels size'])))

        self.v_hat.append(np.zeros((self.topology[-1]['kernels num'], self.topology[-1]['kernels size'], self.topology[-1]['kernels size'])))
        self.m_hat.append(np.zeros((self.topology[-1]['kernels num'], self.topology[-1]['kernels size'], self.topology[-1]['kernels size'])))

    def add_pooling_layer(self, block_size, pooling_type):

        if self.topology[-1]['type'] == 'Conv2d':
            self.topology[-1]['pooling type'] = pooling_type
            self.topology[-1]['block size'] = block_size
            self.topology[-1]['pool size'] = self.topology[-1]['conv size'] // block_size
        else:
            raise SystemExit('Pooling layer must come after Convolutional layer')





    def prepare_targets(self, targets):

        if type(targets) is int:
            correct_target = int(targets)

            last_layer_act_func = self.topology[-1]['activation func']

            if last_layer_act_func == 'Sigmoid':
                targets_list = np.zeros(self.topology[-1]['neurons num'])
            elif last_layer_act_func == 'Tanh':
                targets_list =  np.full(self.topology[-1]['neurons num'], -1)
            else:
                raise SystemExit(f'Activation function with name "{self.topology[-1]["activation func"]}" is not found')

            targets_list[correct_target] = 1
  
        else: 
            targets_list = targets

        return targets_list



        
    def weights_init(self):# Done

        self.weights = []

        if self.topology[1]['type'] == 'Dense':
            self.weights.append(np.random.normal(0.0, pow(self.topology[0]['inputs num'], -0.5), (self.topology[0]['inputs num'], self.topology[1]['neurons num'])))
            
        elif self.topology[1]['type'] == 'Conv2d':
            self.weights.append(np.random.normal(0.0, pow(self.topology[1]['kernels size'] * self.topology[1]['kernels size'], -0.5), (self.topology[1]['kernels num'], self.topology[1]['kernels size'], self.topology[1]['kernels size'])))


        for i in range(1, len(self.topology)-1):
            if self.topology[i+1]['type'] == 'Dense':

                if self.topology[i]['type'] == 'Dense':
                    self.weights.append(np.random.normal(0.0, pow(self.topology[i]['neurons num'], -0.5), (self.topology[i]['neurons num'], self.topology[i+1]['neurons num'])))
                elif self.topology[i]['pooling type'] == 'MaxPooling':
                    self.weights.append(np.random.normal(0.0, pow(self.topology[i]['kernels size'] * self.topology[i]['kernels size'], -0.5), (self.topology[i]['kernels num'] * self.topology[i]['pool size'] * self.topology[i]['pool size'], 
                                                                  self.topology[i+1]['neurons num'])))
                elif self.topology[i]['type'] == 'Conv2d':
                    self.weights.append(np.random.normal(0.0, pow(self.topology[i]['kernels size'] * self.topology[i]['kernels size'], -0.5) , (self.topology[i]['kernels num'] * self.topology[i]['conv size'] * self.topology[i]['conv size'], 
                                                                  self.topology[i+1]['neurons num'])))
            elif self.topology[i+1]['type'] == 'Conv2d':
                self.weights.append(np.random.normal(0.0, pow( self.topology[i+1]['kernels size'] * self.topology[i+1]['kernels size'], -0.5) , (self.topology[i+1]['kernels num'], self.topology[i+1]['kernels size'], 
                                                              self.topology[i+1]['kernels size'])))

                
    @staticmethod
    @njit
    def conv_weights_updating(gradient, optimizer, index, weights, alpha, beta, beta2, epsilon, v, m, v_hat, m_hat, kernels_num, kernels_size):
        
        for ker in range(kernels_num):
            for h in range(kernels_size):
                for w in range(kernels_size):
                    weights[ker, h, w], v[ker, h, w], m[ker, h, w], v_hat[ker, h, w], m_hat[ker, h, w] \
                    = optimizer(gradient[ker,h,w], weights[ker, h, w], index, alpha, beta, beta2, epsilon, 
                                                    v[ker, h, w], m[ker, h, w], v_hat[ker, h, w], m_hat[ker, h, w])

        return weights, v, m, v_hat, m_hat
                      

    @staticmethod
    @njit
    def compute_conv_gradients(input_layer, conv_layer_error, weights, prev_kernels_num, kernels_num, conv_size, kernels_size, kernel_per_input):
        
        l = 0
        r = kernel_per_input

        gradient = np.zeros((weights.shape))
        
        if  prev_kernels_num <= kernels_num:
            for ker in range(prev_kernels_num):
                for i in range(l, r): 
                    for h in range(kernels_size):
                        for w in range(kernels_size):
                            
                            gradient[i][h][w] = np.sum(conv_layer_error[i] * input_layer[ker, h:h + conv_size, w:w + conv_size])

                            # optimizer(gradient, [index, i, h, w])

                l = r
                r += kernel_per_input
        else:
            for ker in range(kernels_num):
                for h in range(kernels_size):
                    for w in range(kernels_size):
                        for i in range(l, r):
                            gradient[ker][h][w] += np.sum(conv_layer_error[ker] * input_layer[i, h:h + conv_size, w:w + conv_size])

                        # optimizer(gradient, [index, ker, h, w])
                        # gradient = 0

                l = r
                r += kernel_per_input

            
            # for ker in range(self.topology[index]['kernels num']):
            #     for i in range(l, r): 
            #         for h in range(self.topology[index]['kernels size']):
            #             for w in range(self.topology[index]['kernels size']):
                            
            #                 gradient[ker][h][w] = np.sum(conv_layer_error[ker] * input_layer[i, h:h + self.topology[index]['conv size'], w:w + self.topology[index]['conv size']])

            #                 #optimizer(gradient, [index, ker, h, w])

            #     l = r
            #     r += kernel_per_input
        return gradient

            
    def set_params(self, params, optimizer_name):

        if 'alpha' in params: self.alpha = params['alpha']
        if 'beta' in params: self.beta = params['beta']
        if 'beta2' in params: self.beta2 = params['beta2']
        if 'epsilon' in params: self.epsilon = params['epsilon']

        if optimizer_name == 'BGD':

            if self.alpha == None: self.alpha = 0.001
        elif optimizer_name == 'Momentum':

            if self.alpha == None: self.alpha = 0.01
            if self.beta == None: self.beta = 0.9
        elif optimizer_name == 'RMSProp':

            if self.alpha == None: self.alpha = 0.01
            if self.beta == None: self.beta = 0.9
            if self.epsilon == None: self.epsilon = 0.000000001
        elif optimizer_name == 'Adam':

            if self.alpha == None: self.alpha = 0.001
            if self.beta == None: self.beta = 0.9
            if self.beta2 == None: self.beta2 = 0.999
            if self.epsilon == None: self.epsilon = 0.000000001
        elif optimizer_name == 'Nadam': 

            if self.alpha == None: self.alpha = 0.001
            if self.beta == None: self.beta = 0.9
            if self.beta2 == None: self.beta2 = 0.999
            if self.epsilon == None: self.epsilon = 0.000000001



   
    @staticmethod
    @njit
    def BGD_optimizer(gradient, weights, k, alpha, beta, beta2, epsilon, v, m, v_hat, m_hat):
        weights -=  gradient * alpha

        return weights, v, m, v_hat, m_hat

    @staticmethod
    @njit
    def Momentum_optimizer(gradient, weights, k, alpha, beta, beta2, epsilon, v, m, v_hat, m_hat):
        v = beta * v + (1 - beta) * gradient
        weights -=  v * alpha

        return weights, v, m, v_hat, m_hat


    @staticmethod
    @njit
    def RMSProp_optimizer(gradient, weights, k, alpha, beta, beta2, epsilon, v, m, v_hat, m_hat):
        v = beta * v + (1 - beta) * np.power(gradient,2)
        weights -=  alpha * gradient / (np.sqrt(v) + epsilon)

        return weights, v, m, v_hat, m_hat


    @staticmethod
    @njit
    def Adam_optimizer(gradient, weights, k, alpha, beta, beta2, epsilon, v, m, v_hat, m_hat):

        m = beta * m + (1 - beta) * gradient
        v = beta2 * v + (1 - beta2) * np.power(gradient, 2)

        m_hat = m / (1 - np.power(beta, k))
        v_hat = v / (1 - np.power(beta2, k))

        weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        return weights, v, m, v_hat, m_hat


    @staticmethod
    @njit
    def Nadam_optimizer(gradient, weights, k, alpha, beta, beta2, epsilon, v, m, v_hat, m_hat):

        m = beta * m + (1 - beta) * gradient
        v = beta2 * v + (1 - beta2) * np.power(gradient, 2)

        m_hat = m / (1 - np.power(beta, k)) + (1 - beta) * gradient / (1 - np.power(beta, k))
        v_hat = v / (1 - np.power(beta2, k))

        weights -=  alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        return weights, v, m, v_hat, m_hat


    @staticmethod
    @njit
    def make_padding(layer, padding_size):
        # padded_layer = np.pad(layer, ((0, 0), (padding_size, padding_size), (padding_size, padding_size)), constant_values = 0)
        padded_layer = np.zeros((layer.shape[0], layer.shape[1] + 2 * padding_size, layer.shape[2] + 2 * padding_size))

        for i in range(layer.shape[0]):
            padded_layer[i, padding_size: layer.shape[1] + padding_size, padding_size: layer.shape[1] + padding_size] = layer[i]

        return padded_layer


    @staticmethod
    @njit
    def make_unpadding(layer, padding_size):
        # losses[k] = losses[k][...,self.topology[k+1]['padding']:-self.topology[k+1]['padding'],self.topology[k+1]['padding']:-self.topology[k+1]['padding']]
        unpadded_layer = np.zeros((layer.shape[0], layer.shape[1] - 2 * padding_size, layer.shape[2] - 2 * padding_size))
    
        for i in range(layer.shape[0]):
            unpadded_layer[i] = layer[i, padding_size: layer.shape[1] - padding_size, padding_size: layer.shape[1] - padding_size]

        return unpadded_layer


    @staticmethod
    @njit
    def reshape_to_2d(layer, num, size):
        reshaped_layer = np.reshape(layer, (num, size, size))

        return reshaped_layer


    @staticmethod
    @njit
    def reshape_to_1d(layer, num, size):
        reshaped_layer = np.reshape(layer, (num, size * size))

        return reshaped_layer


    @staticmethod
    @njit
    def make_transposing(layer, stride):
        transposed_layer = np.zeros((layer.shape[0], stride * layer.shape[1] - (stride - 1), stride * layer.shape[1] - (stride - 1)), dtype=layer.dtype)
        transposed_layer[:, ::stride, ::stride] = layer

        return transposed_layer


    @staticmethod
    @njit
    def make_untransposing(layer, stride):
        # losses[k] = losses[k][:,::self.topology[k+1]['stride'], ::self.topology[k+1]['stride']]
        untransposed_layer = np.zeros((layer.shape[0], (layer.shape[1] + (stride - 1)) // stride, (layer.shape[1] + (stride - 1)) // stride), dtype=layer.dtype)
        untransposed_layer = layer[:, ::stride, ::stride]

        return untransposed_layer


    @staticmethod
    @njit
    def make_upsampling(layer, scale_factor): #njit doesnt work
        # upsampled_layer = layer.repeat(scale_factor, axis=1).repeat(scale_factor, axis=2)
        upsampled_layer = np.zeros((layer.shape[0], layer.shape[1] * scale_factor, layer.shape[2] * scale_factor))

        ic, jc = 0, 0 

        for k in range(layer.shape[0]):
            for i in range(layer.shape[1]):
                for j in range(layer.shape[2]):
                    upsampled_layer[k, i + ic:(i+1) * scale_factor, j + jc:(j+1) * scale_factor] = layer[k, i, j]

                    jc += scale_factor - 1
                ic += scale_factor - 1

                jc = 0

            ic = 0 

        return upsampled_layer


    @staticmethod
    @njit
    def make_downsampling(layer, scale_factor):
        # downsampled_layer = layer[..., :layer.shape[1], : layer.shape[1]].reshape(layer.shape[0], layer.shape[1]//scale_factor, scale_factor, layer.shape[1]//scale_factor, scale_factor).mean(axis=(2,4))

        downsampled_layer = np.zeros((layer.shape[0], layer.shape[1] // scale_factor, layer.shape[2] // scale_factor))

        ic, jc = 0, 0 

        for k in range(downsampled_layer.shape[0]):
            for i in range(downsampled_layer.shape[1]):
                for j in range(downsampled_layer.shape[2]):
                    downsampled_layer[k, i, j]  = layer[k, i + ic:(i+1) * scale_factor, j + jc:(j+1) * scale_factor].mean()

                    jc += scale_factor - 1
                ic += scale_factor - 1

                jc = 0
            
            ic = 0

        return downsampled_layer

    
    def weights_updating(self, outputs, losses, optimizer):

        for k in range(len(self.topology)-1,0,-1):
            gradient = 0

            for l in range(len(outputs)):

                if self.topology[k]['type'] == 'Dense':
                    
                    input_layer = np.array(outputs[l][k-1].flatten(),ndmin = 2).T
                    
                    gradient += np.dot(input_layer, losses[l][k] ) #* self.act_func_der(outputs[l][k], self.topology[k]['activation func']) внутри скобки

                    # optimizer(gradient, [k])

                elif self.topology[k]['type'] == 'Conv2d':

                    input_layer = self.reshape_to_2d(outputs[l][k-1], num = self.topology[k]['previous kernels num'], size = self.topology[k]['init input size'])

                    #upsampling
                    if self.topology[k]['upsampling scale factor'] != 1:
                        input_layer = self.make_upsampling(input_layer, self.topology[k]['upsampling scale factor'])
                    #transposing
                    if self.topology[k]['stride'] != 1:
                        input_layer = self.make_transposing(input_layer, self.topology[k]['stride'])
                    #padding
                    if self.topology[k]['padding'] != 0:
                        input_layer = self.make_padding(input_layer, self.topology[k]['padding'])


                    gradient += self.compute_conv_gradients(input_layer, losses[l][k], self.weights[k-1],   self.topology[k]['previous kernels num'], 
                                                                                                            self.topology[k]['kernels num'], 
                                                                                                            self.topology[k]['conv size'], 
                                                                                                            self.topology[k]['kernels size'],
                                                                                                            self.topology[k]['kernel per input'])
                    pass

            # gradients[k] = gradient

            if self.topology[k]['type'] == 'Dense':
                self.weights[k-1], self.v[k-1], self.m[k-1], self.v_hat[k-1], self.m_hat[k-1] = optimizer(gradient, self.weights[k-1], k, self.alpha, self.beta, self.beta2, self.epsilon,
                                   self.v[k-1], self.m[k-1], self.v_hat[k-1], self.m_hat[k-1])
                    
            elif self.topology[k]['type'] == 'Conv2d':
               
                self.weights[k-1], self.v[k-1], self.m[k-1], self.v_hat[k-1], self.m_hat[k-1] = self.conv_weights_updating(gradient, optimizer, k, self.weights[k-1], self.alpha, self.beta, self.beta2, self.epsilon, 
                                   self.v[k-1], self.m[k-1], self.v_hat[k-1], self.m_hat[k-1],  self.topology[k]['kernels num'],  self.topology[k]['kernels size'])
         

    @staticmethod
    @njit
    def conv_backward_prop(conv_layer_error, weights, kernels_num, input_size, conv_size, kernels_size):

        weights_rot_180 = weights

        if conv_size >= kernels_size:
            error_pattern = np.zeros((kernels_num, input_size + conv_size - 1, input_size + conv_size - 1))
        else:
            error_pattern = np.zeros((kernels_num, input_size + kernels_size - 1, input_size + kernels_size - 1))
        
        conv_layer_backward_prop_error = np.zeros((kernels_num, input_size, input_size))


        for i in range(kernels_num):
            error_pattern[i, kernels_size - 1: conv_size + kernels_size - 1,kernels_size- 1:conv_size + kernels_size  - 1] = conv_layer_error[i] #Матрица ошибок нужного размера для прогона по ней весов

        for s in range(kernels_num):
            weights_rot_180[s] = np.fliplr(weights_rot_180[s])
            weights_rot_180[s] = np.flipud(weights_rot_180[s])
       
        for s in range(kernels_num):
            for h in range(input_size):
                for w in range(input_size):
                    
                    conv_layer_backward_prop_error[s, h, w] = np.sum(error_pattern[s, h:h + kernels_size, w:w + kernels_size] * weights_rot_180[s]) 

        return conv_layer_backward_prop_error


    @staticmethod
    @njit
    def loss_convertation(loss_wrong_num, der_input_layer, prev_kernels_num, kernels_num, input_size, kernel_per_input): #
        pooling_layer_error = np.zeros((prev_kernels_num, input_size, input_size))


        l = 0
        r = kernel_per_input

        if  prev_kernels_num <= kernels_num:
            for k in range(prev_kernels_num,):
                for s in range(l, r):
            
                    pooling_layer_error[k] += loss_wrong_num[s] * der_input_layer[k]#* self.act_func_der(pooling_layer[k], self.topology[index - 1]['activation func']) #index

                l = r
                r += kernel_per_input
        else:
            for k in range(kernels_num):
                for s in range(l, r):
            
                    pooling_layer_error[s] = loss_wrong_num[k] * der_input_layer[k] #* self.act_func_der(pooling_layer[s], self.topology[index - 1]['activation func']) #index

                l = r
                r += kernel_per_input

        return pooling_layer_error


    @staticmethod
    @njit
    def pooling_error_expansion(pooling_layer_ind, pooling_layer_error, kernels_num, conv_size):
        
        conv_layer_error = np.zeros((kernels_num, conv_size, conv_size))

        for s in range(kernels_num):
            i = 0
            for h in range(conv_size):
                for w in range(conv_size):
                    if (pooling_layer_ind[s, h, w] == 1):
                        conv_layer_error[s, h, w] = pooling_layer_error[s, i]
                        i += 1

        return conv_layer_error


    @staticmethod
    @njit
    def pooling(conv_layer, kernels_num, conv_size, pool_size, block_size, pooling_type):
        
        pooling_layer = np.zeros((kernels_num, pool_size, pool_size))
        pooling_layer_ind = np.zeros((kernels_num, conv_size, conv_size))

        for s in range(kernels_num):
            for h in range(pool_size):
                for w in range(pool_size):

                    pool_part = conv_layer[s, h * block_size:h * block_size + block_size,w * block_size:w * block_size + block_size]#.astype(np.float64)

                    if pooling_type == 'MaxPooling':
                        pooling_layer[s, h, w] = pool_part.max()
                    elif pooling_type == 'AvgPooling':
                        pooling_layer[s, h, w] = pool_part.mean()

                    for i in range(block_size):
                        for j in range(block_size):
                            if pool_part[i, j] == pooling_layer[s, h, w]:
                                I = int(i + h * block_size)

                                J = int(j + w * block_size)

                    pooling_layer_ind[s, I, J] = 1

        
        return pooling_layer, pooling_layer_ind


    @staticmethod
    @njit
    def convolution(input_layer, weights, bias, prev_kernels_num, kernels_num, conv_size, kernels_size, kernel_per_input):
        
        conv_layer = np.zeros((kernels_num,  conv_size,  conv_size))

        l = 0
        r = kernel_per_input
        if  prev_kernels_num <= kernels_num:

            for k in range(prev_kernels_num):
                for i in range(l, r):
                    for h in range(conv_size):
                        for w in range(conv_size):
                        
                            conv_layer[i, h, w] = np.sum(input_layer[k, h:h + kernels_size, w:w + kernels_size] * weights[i]) + bias
                l = r
                r += kernel_per_input

        else:
            
            for k in range(kernels_num):
                for i in range(l, r):
                    for h in range(conv_size):
                        for w in range(conv_size):
                        
                            conv_layer[k, h, w] += np.sum(input_layer[i, h:h + kernels_size, w:w + kernels_size] * weights[k]) + bias
                l = r
                r += kernel_per_input

            # collected_input_layer = np.zeros((self.topology[index]['kernels num'], self.topology[index]['input size'], self.topology[index]['input size']))

            # for k in range(self.topology[index]['kernels num']):
            #     for i in range(l, r):
            #         collected_input_layer[k]+=input_layer[i]
               
            #     l = r
            #     r += kernel_per_input

            # for k in range(self.topology[index]['kernels num']):
            #         for h in range(self.topology[index]['conv size']):
            #             for w in range(self.topology[index]['conv size']):
                        
            #                 conv_layer[k, h, w] = np.sum(collected_input_layer[k, h:h + self.topology[index]['kernels size'], w:w + self.topology[index]['kernels size']] * self.weights[index-1][k]) + self.topology[index]['bias']

 
        return conv_layer
                

    def forward_prop(self, input_values):
        
        outputs = []
        outputs.append(np.expand_dims(np.asfarray(input_values), axis=0))

   
        for k in range(0,len(self.topology)-1):

            if self.topology[k+1]['type'] == 'Dense':

                if (self.topology[k]['type'] == 'Dense' or self.topology[k]['type'] == 'Input') and self.topology[k]['dropout rate'] != None:
                    outputs[k] *= np.random.binomial(n = 1, p = 1 - self.topology[k]['dropout rate'], size = outputs[-1].shape)
                
                outputs.append(self.activation_func(np.dot(np.array(outputs[k].flatten(),ndmin = 2), self.weights[k])+self.topology[k+1]['bias'],self.topology[k+1]['activation func']))


            elif self.topology[k+1]['type'] == 'Conv2d':
                
                if self.topology[k]['type']=='Dense' or self.topology[k]['type']=='Input':
                    input_layer = self.reshape_to_2d(outputs[k], num = self.topology[k+1]['previous kernels num'], size = self.topology[k+1]['init input size'])

                elif self.topology[k]['type']=='Conv2d' or self.topology[k]['pooling type'] != None:
                    input_layer = outputs[k]

                #upsampling
                if self.topology[k+1]['upsampling scale factor'] != 1:
                    input_layer = self.make_upsampling(input_layer, self.topology[k+1]['upsampling scale factor'])
                #transposing
                if self.topology[k+1]['stride'] != 1:
                    input_layer = self.make_transposing(input_layer, self.topology[k+1]['stride'])
                #padding
                if self.topology[k+1]['padding'] != 0:
                    input_layer = self.make_padding(input_layer, self.topology[k+1]['padding'])
                  
                outputs.append(self.activation_func(self.convolution(input_layer, self.weights[k], self.topology[k+1]['bias'], 
                                                                                                        self.topology[k+1]['previous kernels num'],  
                                                                                                        self.topology[k+1]['kernels num'],
                                                                                                        self.topology[k+1]['conv size'],
                                                                                                        self.topology[k+1]['kernels size'],
                                                                                                        self.topology[k+1]['kernel per input']
                
                                                                                                                                                ), self.topology[k+1]['activation func']))

                    
                if self.topology[k+1]['pooling type'] != None:
                    outputs[-1], self.topology[k+1]['pool inds'] = self.pooling(outputs[-1],    self.topology[k+1]['kernels num'], 
                                                                                                self.topology[k+1]['conv size'], 
                                                                                                self.topology[k+1]['pool size'],
                                                                                                self.topology[k+1]['block size'],
                                                                                                self.topology[k+1]['pooling type'])

        return outputs


    def backward_prop(self, layers_outputs, loss_func):
        losses = [None for _ in range(len(self.topology))]

        losses[-1] = loss_func * self.act_func_der(layers_outputs[-1], self.topology[-1]['activation func'])
        
        if self.topology[-1]['type']=='Conv2d':
            if self.topology[-1]['pooling type'] != None:
                losses[-1] = self.pooling_error_expansion(self.topology[-1]['pool inds'], 
                                                                self.reshape_to_1d(losses[-1], num = self.topology[-1]['kernels num'], size = self.topology[-1]['pool size']),
                                                                self.topology[-1]['kernels num'],
                                                                self.topology[-1]['conv size']) 



        for k in range(len(self.topology)-2, -1, -1):  #0

            if ((self.topology[k]['type']=='Dense' or self.topology[k]['type']=='Input')) and (self.topology[k+1]['type'] == 'Dense'): #input
                losses[k] = (np.dot(losses[k+1],self.weights[k].T)) * self.act_func_der(layers_outputs[k].flatten(), self.topology[k]['activation func'])

            elif (self.topology[k]['type']=='Conv2d') and (self.topology[k]['pooling type'] == None) and (self.topology[k+1]['type'] == 'Dense'):
                losses[k] = self.reshape_to_2d(np.dot(losses[k+1], self.weights[k].T) * self.act_func_der(layers_outputs[k], self.topology[k]['activation func']).flatten(), num = self.topology[k]['kernels num'], size = self.topology[k]['conv size'])
                
            elif (self.topology[k]['type']=='Conv2d') and (self.topology[k]['pooling type'] != None) and (self.topology[k+1]['type'] == 'Dense'):
                pooling_losses = self.reshape_to_1d(np.dot(losses[k+1], self.weights[k].T) * self.act_func_der(layers_outputs[k], self.topology[k]['activation func']).flatten(), num = self.topology[k]['kernels num'], size = self.topology[k]['pool size'])

                losses[k] = self.pooling_error_expansion(self.topology[k]['pool inds'], pooling_losses, self.topology[k]['kernels num'], self.topology[k]['conv size']) 

            elif self.topology[k+1]['type'] == 'Conv2d':
                loss_wrg_num = self.conv_backward_prop(losses[k+1], self.weights[k], self.topology[k+1]['kernels num'], self.topology[k+1]['input size'], self.topology[k+1]['conv size'], self.topology[k+1]['kernels size'])

                der_reshaped_outputs =  self.act_func_der(self.reshape_to_2d(layers_outputs[k], num = self.topology[k+1]['previous kernels num'], size = self.topology[k+1]['init input size']), self.topology[k]['activation func'])
                #upsampling
                if self.topology[k+1]['upsampling scale factor'] != 1:
                    der_reshaped_outputs = self.make_upsampling(der_reshaped_outputs, self.topology[k+1]['upsampling scale factor'])
                #transposing
                if self.topology[k+1]['stride'] != 1:
                    der_reshaped_outputs = self.make_transposing(der_reshaped_outputs, self.topology[k+1]['stride'])
                #padding
                if self.topology[k+1]['padding'] != 0:
                    der_reshaped_outputs = self.make_padding(der_reshaped_outputs, self.topology[k+1]['padding'])
                
               

                losses[k] = self.loss_convertation(loss_wrg_num, der_reshaped_outputs,
                                                                        self.topology[k+1]['previous kernels num'],
                                                                        self.topology[k+1]['kernels num'], 
                                                                        self.topology[k+1]['input size'], 
                                                                        self.topology[k+1]['kernel per input'])
                

                #unpading
                if self.topology[k+1]['padding'] != 0:
                    losses[k] = self.make_unpadding(losses[k], self.topology[k+1]['padding'])
                #untransposing
                if self.topology[k+1]['stride'] != 1:
                    losses[k] = self.make_untransposing(losses[k], self.topology[k+1]['stride'])
                #downsampling
                if self.topology[k+1]['upsampling scale factor'] != 1:
                    losses[k] = self.make_downsampling(losses[k], self.topology[k+1]['upsampling scale factor'])

                losses[k] = np.array(losses[k],ndmin = 2)
                if self.topology[k]['type']=='Dense': 
                    losses[k] = np.array(losses[k].flatten(),ndmin = 2)
            
                elif self.topology[k]['type']=='Conv2d':
                    if self.topology[k]['pooling type'] != None:
                        losses[k] = self.pooling_error_expansion(self.topology[k]['pool inds'],
                                                                self.reshape_to_1d(losses[k], num = self.topology[k]['kernels num'], size = self.topology[k]['pool size']),
                                                                self.topology[k]['kernels num'],
                                                                self.topology[k]['conv size'])
                    
        
        if self.topology[0]['input type'] == '1d':
            losses[0] = np.array(losses[0].flatten(),ndmin = 2)
        # elif  self.topology[0]['input type'] == '2d':
        #     losses[0] = losses[0].reshape(self.topology[0]['inputs num'], self.topology[1]['init input size'], self.topology[1]['init input size'])
            
            
        return losses 
        

    def train(self, inputs, targets, epochs, loss_function_name, optimizer_name, batch_size = 1, trained_model = 'classifier', **params):

        batch_num = len(inputs) // batch_size
        inputs = np.asarray(inputs)

        batches = np.array_split(inputs, batch_num)
        batches_targets = np.array_split(targets, batch_num)

        loss_metric = []
        accuracy_metric = []

        total_samples_num = true_samples_num = 0

        batch_layers_outputs = []
        batch_layers_losses = []

        if len(self.weights) == 0: self.weights_init()

        optimizer = self.optimizers[optimizer_name]
        loss_function = self.loss_functions[loss_function_name]
        loss_function_metric = self.loss_functions_metrics[loss_function_name]

        self.set_params(params, optimizer_name)

        # for i in tqdm(range(epochs),desc = f'training; optimizer: {optimizer_name}'):
        for i in range(epochs):
            tqdm_range = tqdm(range(batch_num))
            for j in tqdm_range:
                for k in range(len(batches[j])):
                
                    if trained_model == 'classifier':
                        targets_list = self.prepare_targets(int(batches_targets[j][k]))
                    elif trained_model == 'encoder':
                        targets_list = self.prepare_targets(batches_targets[j][k])
                    

                    
                    layers_outputs = self.forward_prop(batches[j][k])
                    batch_layers_outputs.append(layers_outputs)
                    
                    layers_losses = self.backward_prop(layers_outputs, loss_function(layers_outputs[-1], targets_list.reshape(layers_outputs[-1].shape)))
                    batch_layers_losses.append(layers_losses)


                    # loss_metric.append(layers_losses[-1].mean())

                    loss = loss_function_metric(layers_outputs[-1], targets_list).mean()
                    loss_metric.append(loss)
                    if trained_model == 'classifier':

                        max_output_index = np.argmax(layers_outputs[-1])

                        total_samples_num += 1
                        
                        if (max_output_index == int(batches_targets[j][k])):
                            true_samples_num += 1

                        accuracy_metric.append(true_samples_num/total_samples_num)

                tqdm_range.set_description(f'training | optimizer: {optimizer_name} | loss: {loss:.4f} | epoch {i + 1}/{epochs}')

                self.weights_updating(batch_layers_outputs, batch_layers_losses, optimizer)

                batch_layers_outputs = []
                batch_layers_losses = []

        return loss_metric, accuracy_metric


    def test(self, inputs, targets):
        
        inputs = np.asarray(inputs)

        accuracy_metric = []
        samples_num = true_samples_num = 0
        
        for j in tqdm(range(len(inputs)), desc = 'testing'):
            
            layers_outputs = self.forward_prop(inputs[j])

            max_output_index = np.argmax(layers_outputs[-1])

            samples_num += 1
            
            if (max_output_index == int(targets[j])):
                true_samples_num += 1

            accuracy_metric.append(true_samples_num/samples_num)

            # print(f'inputs: {inputs[j]}, targets: {targets[j]}, output: {max_output_index}, output neurons values : {layers_outputs[len(layers_outputs)-1]}')

        print(f'> {accuracy_metric[-1] * 100} %')

        return accuracy_metric


    def predict(self, input):
        
        layers_outputs = self.forward_prop(input)

        max_output_index = np.argmax(layers_outputs[-1])

        return max_output_index, layers_outputs[-1]






