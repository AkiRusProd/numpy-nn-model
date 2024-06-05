import sys
from pathlib import Path

sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
from tqdm import tqdm

from tqdm import tqdm
from neunet.optim import Adam
import neunet as nnet
import neunet.nn as nn
import numpy as np
import os

from neunet.optim import SGD, Adam


document = [
    "Nice Clothes!",
    "Very good shop for clothes",
    "Amazing clothes",
    "Clothes are good",
    "Superb!",
    "Very bad",
    "Poor quality",
    "not good",
    "clothes fitting bad",
    "Shop not good",
]

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
vocab_size = 40


chars2remove = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

filtered_document = []
for line in document:
    line = "".join([c for c in line if c not in chars2remove])
    filtered_document.append(line)

print(f"Filtered document: {filtered_document}")


words = set([word.lower() for line in filtered_document for word in line.split()])

print(f"Words: {words}")


words_labels = np.random.choice(range(1, vocab_size), len(words), replace=False)
vocab = dict(zip(words, words_labels))

encoded_document = []
for line in filtered_document:
    encoded_line = []
    for word in line.split():
        encoded_line.append(vocab[word.lower()])
    encoded_document.append(encoded_line)

print(f"Encoded document: {encoded_document}")

max_length = len(max(encoded_document, key=len))
print(f"Max length: {max_length}")

padded_document = []
for line in encoded_document:
    if len(line) < max_length:
        padded_line = line + [0] * (max_length - len(line))
    padded_document.append(padded_line)

print("Padded document:", *padded_document, sep="\n")


# class ExtractTensor(nn.Module):
#     def __init__(self, return_sequences):
#         super().__init__()
#         self.return_sequences = return_sequences

#     def forward(self, X):
#         all_states, last_state = X
#         if self.return_sequences:
#             return all_states
#         else:
#             return last_state

# model = nn.Sequential(
#     nn.Embedding(vocab_size, 10),
#     nn.GRU(10, 50),
#     ExtractTensor(return_sequences=True),
#     nn.GRU(50, 50),
#     ExtractTensor(return_sequences=True),
#     nn.GRU(50, 50),
#     ExtractTensor(return_sequences=False),
#     nn.Linear(50, 1),
#     nn.Sigmoid()
# )

model = nn.Sequential(
    nn.Embedding(vocab_size, 10),
    nn.Bidirectional(nn.GRU(10, 50, return_sequences=True), merge_mode="sum"),
    nn.Bidirectional(nn.RNN(50, 50, return_sequences=True, bias=True)),
    nn.Bidirectional(nn.GRU(50, 50, return_sequences=False)),
    nn.Linear(50, 1),
    nn.Sigmoid(),
)


loss_fn = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

padded_document = np.array(padded_document)


labels = np.array(labels).reshape(-1, 1)


loss = []
epochs = 100
tqdm_range = tqdm(range(epochs))
for epoch in tqdm_range:
    for i in range(padded_document.shape[0]):
        optimizer.zero_grad()
        y_pred = model.forward(nnet.tensor(padded_document[i], dtype=nnet.int32))

        loss_ = loss_fn(y_pred, labels[i])
        loss_.backward()
        optimizer.step()
        loss.append(loss_.data)

    tqdm_range.set_description(f"epoch: {epoch + 1}/{epochs}, loss: {loss[-1]:.7f}")


acc = 0
for i in range(padded_document.shape[0]):
    y_pred = model.forward(nnet.tensor(padded_document[i], dtype=nnet.int32))
    if y_pred.data.round() == labels[i]:
        acc += 1

print(f"Accuracy: {acc / padded_document.shape[0] * 100}%")


import matplotlib.pyplot as plt

plt.plot(loss)
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("iterations")
plt.show()
