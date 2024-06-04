import sys
from pathlib import Path

sys.path[0] = str(Path(sys.path[0]).parent)


from tqdm import tqdm
from neunet.optim import Adam
import neunet as nnet
import neunet.nn as nn
import numpy as np
import os

from data_loader import load_mnist


image_size = (1, 28, 28)

training_dataset, test_dataset, training_targets, test_targets = load_mnist()
training_dataset = (
    training_dataset / 127.5 - 1
)  # normalization: / 255 => [0; 1]  #/ 127.5-1 => [-1; 1]
test_dataset = (
    test_dataset / 127.5 - 1
)  # normalization: / 255 => [0; 1]  #/ 127.5-1 => [-1; 1]


device = "cuda"


class Conv2dClassifier(nn.Module):
    def __init__(self):
        super(Conv2dClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.bnorm = nn.BatchNorm2d(16)
        self.leaky_relu = nn.LeakyReLU()

        self.fc1 = nn.Linear(784, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.maxpool2(x)

        x = self.bnorm(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.sigmoid(x)

        return x


classifier = Conv2dClassifier().to(device)


def one_hot_encode(labels):
    one_hot_labels = np.zeros((labels.shape[0], 10))

    for i in range(labels.shape[0]):
        one_hot_labels[i, int(labels[i])] = 1

    return one_hot_labels


loss_fn = nn.MSELoss()
optimizer = Adam(classifier.parameters(), lr=0.001)

batch_size = 100
epochs = 3

for epoch in range(epochs):
    tqdm_range = tqdm(
        range(0, len(training_dataset), batch_size), desc="epoch " + str(epoch)
    )
    for i in tqdm_range:
        batch = training_dataset[i : i + batch_size]

        batch = batch.reshape(
            batch.shape[0], image_size[0], image_size[1], image_size[2]
        )

        batch = nnet.tensor(batch, device=device)

        labels = nnet.tensor(
            one_hot_encode(training_targets[i : i + batch_size]), device=device
        )

        optimizer.zero_grad()

        outputs = classifier(batch)

        loss = loss_fn(outputs, labels)

        loss.backward()

        optimizer.step()

        tqdm_range.set_description(
            f"epoch: {epoch + 1}/{epochs}, loss: {loss.data:.7f}"
        )


# evaluate
correct = 0
total = 0

classifier.eval()

for i in tqdm(range(len(test_dataset)), desc="evaluating"):
    img = test_dataset[i]
    img = img.reshape(1, image_size[0], image_size[1], image_size[2])
    img = nnet.tensor(img, requires_grad=False, device=device)
    outputs = classifier(img)
    predicted = np.argmax(outputs.data)

    total += 1
    correct += predicted == test_targets[i]

print(
    "Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total)
)
