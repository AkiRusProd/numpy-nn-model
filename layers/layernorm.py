from autograd import Tensor
import numpy as np






class LayerNorm(): #layer with dynamic backpropagation
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        self.normalized_shape = (normalized_shape, ) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weight = Tensor(np.ones((normalized_shape)))
        self.bias = Tensor(np.zeros((normalized_shape)))

    def forward(self, X):
        axis = tuple(range(-len(self.normalized_shape), 0))

        mean = X.mean(axis = axis, keepdims=True)
        var = X.var(axis = axis, keepdims=True)

        X_centered = X - mean # BUG X - mean because sign -
        varaddeps = var + self.eps
        powvaraddeps = varaddeps.power(0.5)
        stddev_inv = Tensor(1).div(powvaraddeps) #1 / np.sqrt(var + self.eps) BUG

        O = X_centered * stddev_inv

        if self.elementwise_affine:
            O = self.weight * O + self.bias
        
        return O

    def __call__(self, X):
        return self.forward(X)

x_arr = np.random.randn(2, 3, 3)
# x_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
x = Tensor(x_arr)
ln = LayerNorm((3, 3))
y = ln(x)
print(y)

y.backward(np.ones_like(y.data))
print("-----------------")
print(x.grad)
x_grad = x.grad
ln_weight_grad = ln.weight.grad
# print(ln.weight.grad.shape, ln.weight.shape)
print(ln.weight.grad)
print(ln.bias.grad)






class _LayerNormTensor(Tensor): # tensor for static backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad = 1):
        X, weight, bias, X_centered, stddev_inv, axis = self.args

        _axis = list(axis) if isinstance(axis, tuple) else axis
        X_hat = X_centered * stddev_inv

        dX_hat = weight.data * grad
        dstddev_inv = -0.5 * np.power(stddev_inv, 3) * np.sum(dX_hat * X_centered, axis = axis, keepdims = True)
        dvar = np.ones_like(X.data) * dstddev_inv * 2 * X_centered / np.prod(np.array(X.shape)[_axis])
        dmean = np.ones_like(X.data) * np.sum(dX_hat * stddev_inv, axis = axis, keepdims = True) * (-1) / np.prod(np.array(X.shape)[_axis])
        grad_X = dX_hat * stddev_inv + dvar + dmean



        # grad_X = (1 / weight.size) * weight.data * stddev_inv * (
        #     weight.size * grad
        #     - np.sum(grad, axis = axis, keepdims = True)
        #     - X_centered * np.power(stddev_inv, 2) * np.sum(grad * X_centered, axis = axis, keepdims = True)
        #     )

        # dX_hat = weight.data * grad
        # dvar = np.sum(dX_hat * X_centered, axis = axis, keepdims = True) * (-0.5) * np.power(stddev_inv, 3)
        # dmean = np.sum(dX_hat * (-stddev_inv), axis = axis, keepdims = True) + dvar * np.mean(-2.0 * X_centered, axis = axis, keepdims = True)
        # # grad_X = dX_hat * stddev_inv + dvar * 2 * X_centered / weight.size + dmean / weight.size
        # axis = list(axis) if isinstance(axis, tuple) else axis
        # grad_X = dX_hat * stddev_inv + dvar * 2 * X_centered / np.prod(np.array(X.shape)[axis])+ np.ones_like(X.data) * dmean / np.prod(np.array(X.shape)[axis])

        grad_weight = np.sum(grad * X_hat, axis = 0)
        grad_bias = np.sum(grad, axis = 0)

        X.backward(grad_X)
        weight.backward(grad_weight)
        bias.backward(grad_bias)




class LayerNorm(): # layer with static backpropagation
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        self.normalized_shape = (normalized_shape, ) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weight = Tensor(np.ones((normalized_shape)))
        self.bias = Tensor(np.zeros((normalized_shape)))

    def forward(self, X):
        axis = tuple(range(-len(self.normalized_shape), 0))

        mean = np.mean(X.data, axis = axis, keepdims=True)
        var = np.var(X.data, axis = axis, keepdims=True)

        X_centered = X.data - mean
        stddev_inv = 1 / np.sqrt(var + self.eps)

        O = X_centered * stddev_inv

        if self.elementwise_affine:
            O = self.weight.data * O + self.bias.data

        return _LayerNormTensor(O, [X, self.weight, self.bias, X_centered, stddev_inv, axis], "layernorm")

    def __call__(self, X):
        return self.forward(X)




x = Tensor(x_arr)
ln = LayerNorm((3, 3))
y = ln(x)
print(y)

y.backward(np.ones_like(y.data))
print("----------------- grads from static backpropagation")
print(x.grad)
# print(ln.weight.grad.shape, ln.weight.shape)
print(ln.weight.grad)
print(ln.bias.grad)
print(x_grad/x.grad)
# print(ln_weight_grad/ln.weight.grad)
print(np.allclose(x_grad, x.grad))
# print(ln_weight_grad/ln.weight.grad)


# import torch
# import torch.nn as nn

# x = torch.tensor(x_arr, dtype=torch.float32, requires_grad=True)
# ln = nn.LayerNorm(3)
# y = ln(x)
# print(y)

# y.backward(torch.ones_like(y.data))
# print("----------------- grads from pytorch")
# print(x.grad)
# print(ln.weight.grad)
# print(ln.bias.grad)



# class LayerNorm(nn.Module):
#     def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
#         super().__init__()
#         self.normalized_shape = (normalized_shape, ) if isinstance(normalized_shape, int) else normalized_shape
#         self.eps = eps
#         self.elementwise_affine = elementwise_affine

#         self.weight = nn.Parameter(torch.ones((normalized_shape)))
#         self.bias = nn.Parameter(torch.zeros((normalized_shape)))

#     def forward(self, X):
#         axis = tuple(range(-len(self.normalized_shape), 0))
#         mean = X.mean(dim = axis, keepdims=True)
#         var = X.var(dim = axis, keepdims=True, unbiased=False)

#         X_centered = X - mean
#         varaddeps = var + self.eps
#         powvaraddeps = varaddeps.pow(0.5)
#         stddev_inv = 1 / powvaraddeps

#         O = X_centered * stddev_inv

#         if self.elementwise_affine:
#             O = self.weight * O + self.bias
        
#         return O

#     def __call__(self, X):
#         return self.forward(X)


# ln = LayerNorm(3)
# x = torch.tensor(x_arr, dtype=torch.float32, requires_grad=True)
# y = ln(x)
# print(y)

# y.backward(torch.ones_like(y.data))
# print("-----------------")
# print(x.grad)
# print(ln.weight.grad)
# print(ln.bias.grad)



# # test torch mean and Tensor mean
# # x_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
# # x_arr = np.random.randn(2, 3, 4, 5)
# # x = torch.tensor(x_arr, dtype=torch.float32, requires_grad=True)


# x = torch.tensor(x_arr, dtype=torch.float32, requires_grad=True)
# print("X shape: ", x.shape)
# z = x.mean(axis = (1, 2))
# print(z.shape)
# z.backward(torch.ones_like(z.data))
# print("X grad: ", x.grad.shape, z.data.shape)


# x = Tensor(x_arr)


# z = x.mean(axis = (1, 2))
# print(z.shape, np.ones_like(z.data).shape)
# z.backward(np.ones_like(z.data))
# print("X grad: ", x.grad.shape, z.data.shape)

