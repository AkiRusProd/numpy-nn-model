import numpy as np

import neunet
from neunet.autograd import Tensor
from neunet.nn.modules import Module
from neunet.nn.parameter import Parameter


class _EmbeddingTensor(Tensor):  # tensor for static backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

    def backward(self, grad=1):
        X, weight = self.args

        axis = list(range(len(X.shape)))
        axis[-1], axis[-2] = axis[-2], axis[-1]

        weight_grad = self.xp.matmul(X.transpose(*axis), grad)
        weight.backward(weight_grad)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device="cpu"):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # stdv = 1. / self.xp.sqrt(embedding_dim)
        # self.weight = Tensor(self.xp.random.uniform(-stdv, stdv, (num_embeddings, embedding_dim)), dtype=self.xp.float32)
        self.weight = Parameter(
            neunet.tensor(np.random.randn(num_embeddings, embedding_dim), dtype=np.float32)
        )  # Torch's initialization
        self.to(device)

    def one_hot(self, X):
        O = self.xp.zeros((X.size, self.num_embeddings), dtype=self.weight.dtype)
        O[self.xp.arange(X.size), X.reshape(1, -1)] = 1

        return O.reshape(*X.shape, self.num_embeddings)

    def forward(self, X):
        if not isinstance(X, Tensor):
            raise TypeError("Input must be a tensor")
        if X.device != self.device:
            raise ValueError("Tensors must be on the same device")

        X_one_hot = self.one_hot(X if isinstance(X, self.xp.ndarray) else X.data)
        return _EmbeddingTensor(
            self.xp.dot(X_one_hot, self.weight.data),
            (X_one_hot, self.weight),
            "Embedding",
            device=self.device,
        )

    def __call__(self, X):
        return self.forward(X)
