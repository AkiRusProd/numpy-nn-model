# import sys
# from pathlib import Path
# sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
from tqdm import tqdm
import torch
from torch.nn import Linear, Dropout,  RNNCell, Embedding, RNN, LSTM
from torch.nn import Sigmoid
from torch.nn import Sequential, Module
from torch.nn import MSELoss
from torch.optim import SGD, Adam


document = ['Nice Clothes!', 
            'Very good shop for clothes',
            'Amazing clothes', 
            'Clothes are good', 
            'Superb!', 
            'Very bad', 
            'Poor quality', 
            'not good', 
            'clothes fitting bad', 
            'Shop not good']

labels = [1,1,1,1,1,0,0,0,0,0]
vocab_size = 40


chars2remove = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

filtered_document = []
for line in document:
    line = ''.join([c for c in line if c not in chars2remove])
    filtered_document.append(line)

print(f'Filtered document: {filtered_document}')


words = set([word.lower() for line in filtered_document for word in line.split()])

print(f'Words: {words}')


words_labels = np.random.choice(range(1, vocab_size), len(words), replace=False)
vocab = dict(zip(words, words_labels))

encoded_document = []
for line in filtered_document:
    encoded_line = []
    for word in line.split():
        encoded_line.append(vocab[word.lower()])
    encoded_document.append(encoded_line)

print(f'Encoded document: {encoded_document}')

max_length = len(max(encoded_document, key=len))
print(f'Max length: {max_length}')

padded_document = []
for line in encoded_document:
    if len(line) < max_length:
        padded_line = line + [0] * (max_length - len(line))
    padded_document.append(padded_line)

print('Padded document:', *padded_document, sep = "\n")


# model.add(Embedding(vocab_size, 10, input_length=max_length))
# model.add(RNN(50, return_sequences=True))
# model.add(RNN(50, return_sequences=True))
# model.add(RNN(50, return_sequences=False))
# # model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer='adam', loss='binary_crossentropy')
# loss = model.fit(padded_document, labels, batch_size = 1, epochs = 1000)
# acc = model.predict(padded_document, labels)

# print(model.predict(padded_document[0]))
# print(model.predict(padded_document[9]))


# class ExtractTensor(Module):
#     def __init__(self, return_sequences):
#         super().__init__()
#         self.return_sequences = return_sequences

#     def forward(self, X):
#         all_states, last_state = X
#         # print(all_states.shape, last_state.shape)
#         if self.return_sequences:
#             return all_states
#         else:
#             return last_state

class ExtractTensor(Module):
    def __init__(self, return_sequences):
        super().__init__()
        self.return_sequences = return_sequences

    def forward(self, X):
        all_states, last_state = X
        # print(all_states.shape, last_state.shape)
        if self.return_sequences:
            return all_states
        else:
            return last_state[0]

model = Sequential(
    Embedding(vocab_size, 10),
    LSTM(10, 50),
    ExtractTensor(return_sequences=True),
    LSTM(50, 50),
    ExtractTensor(return_sequences=True),
    LSTM(50, 50),
    ExtractTensor(return_sequences=False),
    Linear(50, 1),
    Sigmoid()
)


loss_fn = MSELoss()
optimizer = Adam(model.parameters(), lr=0.01)


padded_document = torch.tensor(padded_document, dtype=torch.long)#[..., np.newaxis]

labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)

loss = []
for epoch in tqdm(range(1000)):
    for i in range(padded_document.shape[0]):
        optimizer.zero_grad()
        y_pred = model.forward(padded_document[i])
        # print(y_pred.shape, labels[i].shape)
        loss_ = loss_fn(y_pred, labels[i].reshape(y_pred.shape))
        loss_.backward()
        optimizer.step()
        loss.append(loss_.data)

    # optimizer.zero_grad()
    # y_pred = model(padded_document)
    # loss_ = loss_fn(y_pred, labels)
    # loss_.backward()
    # loss.append(loss_.data)

    # optimizer.step()

    tqdm.write(f'Epoch: {epoch}, loss: {loss_.data}')
# print(model.forward(padded_document[0]))
# print(model.forward(padded_document[9]))










import matplotlib.pyplot as plt

plt.plot(loss)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('iterations')
plt.show()


emb = Embedding(vocab_size, 10)
out = emb.forward(padded_document[0])
print(out.shape)