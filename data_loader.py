import numpy as np
import os
from tqdm import tqdm
from mnist_data_downloader import download_data


def prepare_data(data):
    inputs, targets = [], []

    for raw_line in tqdm(data, desc = 'preparing data'):
        line = raw_line.split(',')
    
        inputs.append(np.asfarray(line[1:]))
        targets.append(int(line[0]))

    return inputs, targets

def load_mnist(path = "datasets/mnist/"):

    if not os.path.exists(path + "mnist_train.csv") or not os.path.exists(path + "mnist_test.csv"):
        train_url = 'https://pjreddie.com/media/files/mnist_train.csv'
        test_url = 'https://pjreddie.com/media/files/mnist_test.csv'

        download_data(train_url, path + "mnist_train.csv")
        download_data(test_url, path + "mnist_test.csv")


    training_data = open(path + "mnist_train.csv", "r").readlines()
    test_data = open(path + "mnist_test.csv", 'r').readlines()


    if not os.path.exists(path + "mnist_train.npy") or not os.path.exists(path + "mnist_test.npy"):

        training_inputs, training_targets = prepare_data(training_data)
        training_inputs = np.asfarray(training_inputs)

        test_inputs, test_targets = prepare_data(test_data)
        test_inputs = np.asfarray(test_inputs)

        np.save(path + "mnist_train.npy", training_inputs)
        np.save(path + "mnist_test.npy", test_inputs)

        np.save(path + "mnist_train_targets.npy", training_targets)
        np.save(path + "mnist_test_targets.npy", test_targets)
    else:
        training_inputs = np.load(path + "mnist_train.npy")
        test_inputs = np.load(path + "mnist_test.npy")

        training_targets = np.load(path + "mnist_train_targets.npy")
        test_targets = np.load(path + "mnist_test_targets.npy")

    training_dataset = training_inputs
    test_dataset = test_inputs

    return training_dataset, test_dataset, training_targets, test_targets