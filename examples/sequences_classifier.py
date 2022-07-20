import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
from tqdm import tqdm

from nnmodel.layers import Dense, BatchNormalization, Dropout, Flatten, MaxPooling2D, AveragePooling2D, UpSampling2D, Activation, RepeatVector, \
TimeDistributed, RNN, LSTM, GRU, Bidirectional, Embedding
from nnmodel import Model
from nnmodel.activations import LeakyReLU
from nnmodel.optimizers import SGD, Adam
from nnmodel.loss_functions import MSE
model = Model()

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


words_labels = np.random.choice(vocab_size, len(words), replace=False)
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


model.add(Embedding(vocab_size, 10, input_length=max_length))
model.add(Bidirectional(LSTM(50, return_sequences=False)))
# model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')
loss = model.fit(padded_document, labels, batch_size = 1, epochs = 100)
acc = model.predict(padded_document, labels)

# print(model.predict(padded_document[0]))
# print(model.predict(padded_document[9]))


import matplotlib.pyplot as plt

plt.plot(loss)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('iterations')
plt.show()
