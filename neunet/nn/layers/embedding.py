import numpy as np
import cupy as cp
from neunet.autograd import Tensor



class _EmbeddingTensor(Tensor): #tensor for static backpropagation
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device = device)

    def backward(self, grad=1):
        X, weight = self.args

        axis = list(range(len(X.shape)))
        axis[-1], axis[-2] = axis[-2], axis[-1]

        weight_grad = self.xp.matmul(X.transpose(*axis), grad)
        weight.backward(weight_grad)


class Embedding():
    def __init__(self, num_embeddings, embedding_dim, device = "cpu"):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # stdv = 1. / self.xp.sqrt(embedding_dim)
        # self.weight = Tensor(self.xp.random.uniform(-stdv, stdv, (num_embeddings, embedding_dim)), dtype=self.xp.float32)
        self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim), dtype=np.float32) # Torch's initialization
        self.to(device)

    def one_hot(self, X):
        O = self.xp.zeros((X.size, self.num_embeddings))
        O[self.xp.arange(X.size), X.reshape(1, -1)] = 1

        return O.reshape(*X.shape, self.num_embeddings)

    def forward(self, X):
        X_one_hot = self.one_hot(X if isinstance(X, self.xp.ndarray) else X.data)
        return _EmbeddingTensor(self.xp.dot(X_one_hot, self.weight.data), (X_one_hot, self.weight), "Embedding", device = self.device)

    def __call__(self, X):
        return self.forward(X)

    def to (self, device):
        assert device in ["cpu", "cuda"], "Device must be 'cpu' or 'cuda'"
        if device == "cpu":
            self.xp = np
        else:
            self.xp = cp

        self.device = device
        self.weight = self.weight.to(device)

        return self


