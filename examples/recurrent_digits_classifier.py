import sys
sys.path.append("../nnmodel")
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from nnmodel.layers import Dense, BatchNormalization, Dropout, Flatten, Reshape, Activation, RepeatVector, GRU, TimeDistributed, Bidirectional
from nnmodel import Model
from nnmodel.activations import LeakyReLU, ReLU

training_data = open('dataset/mnist_train.csv','r').readlines()
test_data = open('dataset/mnist_test.csv','r').readlines()


def prepare_data(data):
    inputs, targets = [], []

    for raw_line in tqdm(data, desc = 'preparing data'):

        line = raw_line.split(',')
    
        inputs.append(np.asfarray(line[1:])/255)
        targets.append(int(line[0]))

    return inputs, targets



training_inputs, training_targets = prepare_data(training_data)
test_inputs, test_targets = prepare_data(test_data)

model = Model()
model.add(Reshape(shape = (28, 28)))
model.add(GRU(256, input_shape=(28, 28), return_sequences=False, cycled_states = True))
model.add(RepeatVector(28))
model.add(TimeDistributed(Dense(50, use_bias=False)))
model.add(TimeDistributed(BatchNormalization()))
model.add(GRU(128, input_shape=(28, 28), cycled_states = True))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer = "adam", loss = "mse")
model.fit(training_inputs,  training_targets, epochs = 5, batch_size = 200)
model.predict(test_inputs, test_targets)


model.save("saved models/recurrent_digits_classifier")

