import numpy as np
from neunet.autograd import Tensor



class _EmbeddingTensor(Tensor): #tensor for static backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        X, weight = self.args

        axis = list(range(len(X.shape)))
        axis[-1], axis[-2] = axis[-2], axis[-1]

        weight_grad = np.matmul(X.transpose(*axis), grad)
        weight.backward(weight_grad)


class Embedding():
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # stdv = 1. / np.sqrt(embedding_dim)
        # self.weight = Tensor(np.random.uniform(-stdv, stdv, (num_embeddings, embedding_dim)), dtype=np.float32)
        self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim), dtype=np.float32) # Torch's initialization

    def one_hot(self, X):
        O = np.zeros((X.size, self.num_embeddings))
        O[np.arange(X.size), X.reshape(1, -1)] = 1

        return O.reshape(*X.shape, self.num_embeddings)

    def forward(self, X):
        X_one_hot = self.one_hot(X if isinstance(X, np.ndarray) else X.data)
        return _EmbeddingTensor(np.dot(X_one_hot, self.weight.data), (X_one_hot, self.weight), "Embedding")

    def __call__(self, X):
        return self.forward(X)

