# import sys
# from pathlib import Path
# sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
from tqdm import tqdm

from nn import Linear, Dropout,  RNN, LSTM, Embedding, Bidirectional
from nn import Sigmoid
from nn import Sequential, Module
from nn import MSELoss
from optim import SGD, Adam


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





class ExtractTensor(Module):
    def __init__(self, return_sequences):
        super().__init__()
        self.return_sequences = return_sequences

    def forward(self, X):
        all_states, last_state = X
        if self.return_sequences:
            return all_states
        else:
            return last_state

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

# model = Sequential(
#     Embedding(vocab_size, 10),
#     Bidirectional(LSTM(10, 50, return_sequences=True), merge_mode='sum'),
#     Bidirectional(LSTM(50, 50, return_sequences=True)),
#     Bidirectional(LSTM(50, 50, return_sequences=False)),
#     Linear(50, 1),
#     Sigmoid()
# )



loss_fn = MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# for param in model.parameters():
#     print(param.shape)

padded_document = np.array(padded_document)#[..., np.newaxis]
# print(padded_document.shape)
# print(padded_document[0].shape)

labels = np.array(labels).reshape(-1, 1) #check
# print(labels.shape)

loss = []
epochs = 1000
tqdm_range = tqdm(range(epochs))
for epoch in tqdm_range:
    for i in range(padded_document.shape[0]):
        optimizer.zero_grad()
        y_pred = model.forward(padded_document[i])

        loss_ = loss_fn(y_pred, labels[i])
        loss_.backward()
        optimizer.step()
        loss.append(loss_.data)


    tqdm_range.set_description(f"epoch: {epoch + 1}/{epochs}, loss: {loss[-1].round(10)}")



acc = 0
for i in range(padded_document.shape[0]):
    y_pred = model.forward(padded_document[i])
    if y_pred.data.round() == labels[i]:
        acc += 1

print(f'Accuracy: {acc / padded_document.shape[0] * 100}%')


import matplotlib.pyplot as plt

plt.plot(loss)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('iterations')
plt.show()

emb = Embedding(vocab_size, 10)
out = emb.forward(padded_document[0])
print(out.shape)

