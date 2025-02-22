from typing import Literal

import numpy as np

import neunet
from neunet.autograd import Tensor
from neunet.nn.modules import Module
from neunet.nn.parameter import Parameter


class _EmbeddingTensor(Tensor):  # tensor for static backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(X: np.ndarray, weight: Tensor, grad):
            axis = list(range(len(X.shape)))
            axis[-1], axis[-2] = axis[-2], axis[-1]

            weight_grad = weight.xp.matmul(X.transpose(*axis), grad)
            weight.apply_grad(weight_grad)

        self.grad_fn = grad_fn


# class Embedding(Module):
#     def __init__(self, num_embeddings: int, embedding_dim: int, device: str="cpu"):
#         self.num_embeddings = num_embeddings
#         self.embedding_dim = embedding_dim

#         # stdv = 1. / self.xp.sqrt(embedding_dim)
#         # self.weight = Tensor(self.xp.random.uniform(-stdv, stdv, (num_embeddings, embedding_dim)), dtype=self.xp.float32)
#         self.weight = Parameter(
#             neunet.tensor(np.random.randn(num_embeddings, embedding_dim), dtype=np.float32)
#         )  # Torch's initialization
#         self.to(device)

#     def one_hot(self, X):
#         O = self.xp.zeros((X.size, self.num_embeddings), dtype=self.weight.dtype)
#         O[self.xp.arange(X.size), X.reshape(1, -1)] = 1

#         return O.reshape(*X.shape, self.num_embeddings)

#     def forward(self, X: Tensor) -> Tensor:
#         if not isinstance(X, Tensor):
#             raise TypeError("Input must be a tensor")
#         if X.device != self.device:
#             raise ValueError("Tensors must be on the same device")

#         X_one_hot = self.one_hot(X if isinstance(X, self.xp.ndarray) else X.data)
#         return _EmbeddingTensor(
#             self.xp.dot(X_one_hot, self.weight.data),
#             (X_one_hot, self.weight),
#             "Embedding",
#             device=self.device,
#         )

#     def __call__(self, X):
#         return self.forward(X)


class Embedding(Module): # layer with dynamic backpropagation
    def __init__(self, num_embeddings: int, embedding_dim: int, device: Literal["cpu", "cuda"] = "cpu"):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(
            neunet.tensor(np.random.randn(num_embeddings, embedding_dim), dtype=np.float32)
        ) 
        self.to(device)

    def forward(self, X: Tensor) -> Tensor:
        return self.weight[X.data.astype(np.int32)]

    def __call__(self, X):
        return self.forward(X)
