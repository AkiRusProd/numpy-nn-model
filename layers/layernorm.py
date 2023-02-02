from autograd import Tensor
import numpy as np






# class LayerNorm(): #layer with dynamic backpropagation
#     def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
#         self.normalized_shape = (normalized_shape, ) if isinstance(normalized_shape, int) else normalized_shape
#         self.eps = eps
#         self.elementwise_affine = elementwise_affine

#         if elementwise_affine:
#             self.weight = Tensor(np.ones((normalized_shape)))
#             self.bias = Tensor(np.zeros((normalized_shape)))
#         else:
#             self.weight = None
#             self.bias = None

#     def forward(self, X):
#         axis = tuple(range(-len(self.normalized_shape), 0))

#         mean = X.mean(axis = axis, keepdims=True)
#         var = X.var(axis = axis, keepdims=True)

#         X_centered = X - mean
#         varaddeps = var + self.eps
#         powvaraddeps = varaddeps.power(0.5)
#         stddev_inv = Tensor(1).div(powvaraddeps) #1 / np.sqrt(var + self.eps) BUG

#         O = X_centered * stddev_inv

#         if self.elementwise_affine:
#             O = self.weight * O + self.bias
        
#         return O

#     def __call__(self, X):
#         return self.forward(X)


class _LayerNormTensor(Tensor): # tensor for static backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad = 1):
        X, weight, bias, X_centered, stddev_inv, axis, elementwise_affine = self.args

        # _axis = list(axis) if isinstance(axis, tuple) else axis
        X_hat = X_centered * stddev_inv

        weight_data = weight.data if elementwise_affine else 1
        weight_size = weight.size if elementwise_affine else 1

        dX_hat = weight_data * grad
        dstddev_inv = -0.5 * np.power(stddev_inv, 3) * np.sum(dX_hat * X_centered, axis = axis, keepdims = True)
        dvar = np.ones_like(X.data) * dstddev_inv * 2 * X_centered / weight_size #np.prod(np.array(X.shape)[_axis])
        dmean = np.ones_like(X.data) * np.sum(dX_hat * stddev_inv, axis = axis, keepdims = True) * (-1) / weight_size #np.prod(np.array(X.shape)[_axis])
        grad_X = dX_hat * stddev_inv + dvar + dmean

        # grad_X = (1 / weight_size) * weight_data * stddev_inv * (
        #     weight_size * grad
        #     - np.sum(grad, axis = axis, keepdims = True)
        #     - X_centered * np.power(stddev_inv, 2) * np.sum(grad * X_centered, axis = axis, keepdims = True)
        #     )

        # dX_hat = weight_data * grad
        # dvar = np.sum(dX_hat * X_centered, axis = axis, keepdims = True) * (-0.5) * np.power(stddev_inv, 3) * 2 * X_centered / weight_size
        # dmean = (np.sum(dX_hat * (-stddev_inv), axis = axis, keepdims = True) + dvar * np.mean(-2.0 * X_centered, axis = axis, keepdims = True)) * np.ones_like(X.data) / weight_size
        # grad_X = dX_hat * stddev_inv + dvar + dmean

        if elementwise_affine:
            grad_weight = np.sum(grad * X_hat, axis = 0)
            grad_bias = np.sum(grad, axis = 0)

        X.backward(grad_X)
        if elementwise_affine:
            weight.backward(grad_weight)
            bias.backward(grad_bias)




class LayerNorm(): # layer with static backpropagation
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        self.normalized_shape = (normalized_shape, ) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = Tensor(np.ones((normalized_shape)))
            self.bias = Tensor(np.zeros((normalized_shape)))
        else:
            self.weight = None
            self.bias = None

    def forward(self, X):
        axis = tuple(range(-len(self.normalized_shape), 0))

        mean = np.mean(X.data, axis = axis, keepdims=True)
        var = np.var(X.data, axis = axis, keepdims=True)

        X_centered = X.data - mean
        stddev_inv = 1 / np.sqrt(var + self.eps)

        O = X_centered * stddev_inv

        if self.elementwise_affine:
            O = self.weight.data * O + self.bias.data

        return _LayerNormTensor(O, [X, self.weight, self.bias, X_centered, stddev_inv, axis, self.elementwise_affine], "layernorm")

    def __call__(self, X):
        return self.forward(X)

