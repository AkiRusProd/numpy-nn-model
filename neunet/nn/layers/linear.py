from neunet.autograd import Tensor
import numpy as np




# Y = X matmul W.T + b

class _LinearTensor(Tensor): # tensor for static backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)


    def backward(self, grad=1):
        self.args[0].backward(np.matmul(grad, self.args[1].data))
        self.args[1].backward(np.matmul(self.args[0].data.swapaxes(-1, -2), grad).swapaxes(-1, -2))
        if self.args[2] is not None:
            self.args[2].backward(np.sum(grad, axis = 0, keepdims = True))


class Linear(): # layer with static backpropagation
    def __init__(self, in_features, out_features, bias = True):
        self.in_features = in_features
        self.out_features = out_features

        stdv = 1. / np.sqrt(in_features)
        self.weight = Tensor(np.random.uniform(-stdv, stdv, (out_features, in_features)), dtype=np.float32)

        if bias == True:
            self.bias = Tensor(np.zeros((1, out_features)), dtype=np.float32)
        else:
            self.bias = None

    def forward(self, X): 
        self.X = X
        
        self.O = np.matmul(self.X.data, self.weight.data.T)
        if self.bias is not None:
            self.O = self.O + self.bias.data
    
        return _LinearTensor(self.O, [self.X, self.weight, self.bias], "linear")

    def __call__(self, X):
        return self.forward(X)


# class Linear(): # layer with dynamic backpropagation
#     def __init__(self, in_features, out_features, bias = True):
#         self.in_features = in_features
#         self.out_features = out_features

#         stdv = 1. / np.sqrt(in_features)
#         self.weight = Tensor(np.random.uniform(-stdv, stdv, (out_features, in_features)), dtype=np.float32)

#         if bias == True:
#             self.bias = Tensor(np.zeros((1, out_features)), dtype=np.float32)
#         else:
#             self.bias = None

#     def forward(self, X):
#         O = X.matmul(self.weight.T)

#         if self.bias is not None:
#             O = O.add(self.bias)

#         return O

#     def __call__(self, X):
#         return self.forward(X)


# class LinearTensor(Tensor):
#     def __init__(self, data, args, op):
#         super().__init__(data, args, op)


#     def backward(self, grad=1):
#         # return super().backward(grad)
       
#         self.args[0].backward(np.matmul(grad, self.args[1].data.swapaxes(-1, -2)))
#         self.args[1].backward(np.matmul(self.args[0].data.swapaxes(-1, -2), grad))
#         self.args[2].backward(np.sum(grad, axis = 0, keepdims = True))


# class Linear():
#     def __init__(self, in_features, out_features, bias = True):
#         self.in_features = in_features
#         self.out_features = out_features

#         stdv = 1. / np.sqrt(in_features)
#         self.weight = Tensor(np.random.uniform(-stdv, stdv, (in_features, out_features)), dtype=np.float32)
#         self.bias = Tensor(np.zeros((1, out_features)), requires_grad = bias, dtype=np.float32)
#         # self.weight = Tensor(np.random.normal(0, pow(out_features, -0.5), (in_features, out_features)), dtype=np.float32)
#         # self.bias = Tensor(np.zeros((1, out_features)), dtype=np.float32)

#     def forward(self, X): 
#         self.X = X

#         self.O = np.matmul(self.X.data, self.weight.data) + self.bias.data
        
#         return LinearTensor(self.O, [self.X, self.weight, self.bias], "linear")

#     def __call__(self, X):
       
#         return self.forward(X)