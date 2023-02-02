from autograd import Tensor
from nn import Linear, Sequential, Sigmoid, Tanh, BCELoss, MSELoss, LeakyReLU, Dropout, BatchNorm1d, Conv2d, Module, BatchNorm2d
from optim import SGD, Adam
from tqdm import tqdm
import numpy as np
import os
from PIL import Image



training_data = open('datasets/mnist/mnist_train.csv','r').readlines()
test_data = open('datasets/mnist/mnist_test.csv','r').readlines()

image_size = (1, 28, 28)

def prepare_data(data):
    inputs, targets = [], []

    for raw_line in tqdm(data, desc = 'preparing data'):

        line = raw_line.split(',')
    
        inputs.append(np.asfarray(line[1:]))
        targets.append(int(line[0]))

    return inputs, targets



mnist_data_path = "datasets/mnist/"

if not os.path.exists(mnist_data_path + "mnist_train.npy") or not os.path.exists(mnist_data_path + "mnist_test.npy"):
    train_inputs, train_targets = prepare_data(training_data)
    train_inputs = np.asfarray(train_inputs)

    test_inputs, test_targets = prepare_data(test_data)
    test_inputs = np.asfarray(test_inputs)

    np.save(mnist_data_path + "mnist_train.npy", train_inputs)
    np.save(mnist_data_path + "mnist_test.npy", test_inputs)

    np.save(mnist_data_path + "mnist_train_targets.npy", train_targets)
    np.save(mnist_data_path + "mnist_test_targets.npy", test_targets)
else:
    train_inputs = np.load(mnist_data_path + "mnist_train.npy")
    test_inputs = np.load(mnist_data_path + "mnist_test.npy")

    train_targets = np.load(mnist_data_path + "mnist_train_targets.npy")
    test_targets = np.load(mnist_data_path + "mnist_test_targets.npy")


dataset = train_inputs / 127.5-1 # normalization: / 255 => [0; 1]  #/ 127.5-1 => [-1; 1]






class Conv2dClassifier(Module):
    def __init__(self):
        super(Conv2dClassifier, self).__init__()

        self.conv1 = Conv2d(1, 4, 3, 1, 2)
        self.conv2 = Conv2d(4, 8, 3, 1, 1)
        self.conv3 = Conv2d(8, 16, 3, 1, 1)

        self.bnorm1 = BatchNorm2d(4)
        self.bnorm2 = BatchNorm2d(8)
        self.bnorm3 = BatchNorm2d(16)

        self.leaky_relu = LeakyReLU()
        
        self.fc1 = Linear(3136, 10)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.sigmoid(x)

        return x

classifier = Conv2dClassifier()


# class LinearClassifier(Module):
#     def __init__(self):
#         super(LinearClassifier, self).__init__()

#         self.fc1 = Linear(784, 512)
#         self.leaky_relu = LeakyReLU()
#         self.fc2 = Linear(512, 10)
#         self.sigmoid = Sigmoid()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.leaky_relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)

#         return x

# classifier = LinearClassifier()

def one_hot_encode(labels):
    one_hot_labels = np.zeros((labels.shape[0], 10))

    for i in range(labels.shape[0]):
        one_hot_labels[i, int(labels[i])] = 1

    return one_hot_labels


loss = MSELoss()
optimizer = Adam(classifier.parameters(), lr = 0.001)

batch_size = 100
epochs = 5

for epoch in range(epochs):
    tqdm_range = tqdm(range(0, len(dataset), batch_size), desc = 'epoch ' + str(epoch))
    for i in tqdm_range:

        batch = dataset[i:i+batch_size]

        batch = batch.reshape(batch.shape[0], image_size[0], image_size[1], image_size[2])

        batch = Tensor(batch)

        labels = one_hot_encode(train_targets[i:i+batch_size])

        optimizer.zero_grad()

        outputs = classifier(batch)

        l = loss(outputs, labels)

        l.backward()

        optimizer.step()

        tqdm_range.set_description('epoch %d, loss: %.7f' % (epoch, l.data))


# evaluate
correct = 0
total = 0


for i in tqdm(range(len(test_inputs)), desc = 'evaluating'):
    img = test_inputs[i]
    img = img.reshape(1, image_size[0], image_size[1], image_size[2])
    img = Tensor(img)

    outputs = classifier(img)
    predicted = np.argmax(outputs.data)

    total += 1
    correct += (predicted == test_targets[i])

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
