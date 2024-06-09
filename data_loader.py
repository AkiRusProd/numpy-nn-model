import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mnist_data_downloader import download_data


def prepare_mnist_data(data):
    inputs, targets = [], []

    for raw_line in tqdm(data, desc="preparing data"):
        line = raw_line.split(",")

        inputs.append(np.asfarray(line[1:]) / 127.5 - 1)  # normalization: / 255 => [0; 1]  #/ 127.5-1 => [-1; 1]
        targets.append(int(line[0]))

    return inputs, targets


def load_mnist(path="datasets/mnist/"):
    if not (Path(path) / "mnist_train.csv").exists() or not (Path(path) / "mnist_test.csv").exists():
        train_url = "https://pjreddie.com/media/files/mnist_train.csv"
        test_url = "https://pjreddie.com/media/files/mnist_test.csv"

        download_data(train_url, path + "mnist_train.csv")
        download_data(test_url, path + "mnist_test.csv")

    training_data = Path(path).joinpath("mnist_train.csv").open("r").readlines()
    test_data = Path(path).joinpath("mnist_test.csv").open("r").readlines()


    if not (Path(path) / "mnist_train.npy").exists() or not (Path(path) / "mnist_test.npy").exists():
        training_inputs, training_targets = prepare_mnist_data(training_data)
        training_inputs = np.asfarray(training_inputs)

        test_inputs, test_targets = prepare_mnist_data(test_data)
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


import os


def prepare_utkface_data(path, image_size = (3, 32, 32)):
        
    import random

    import numpy as np
    from PIL import Image
    
    images = os.listdir(path)
    random.shuffle(images)
    
    training_inputs = []
    for image in tqdm(images, desc = 'preparing data'):
        image = Image.open(path + "/" + image)
        image = image.resize((image_size[1], image_size[2]))
        image = np.asarray(image)
        image = image.transpose(2, 0, 1)
        image = image / 127.5 - 1
        training_inputs.append(image)

    return np.array(training_inputs)


def load_utkface(path="datasets/utkface/", image_size=(3, 32, 32)):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)

    if not (path / 'UTKFace').exists():
        with zipfile.ZipFile(path / 'archive.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

    save_path = path / 'UTKFace.npy'
    if not save_path.exists():
        training_inputs = prepare_utkface_data(path / 'UTKFace', image_size)
        np.save(save_path, training_inputs)
    else:
        training_inputs = np.load(save_path)

    return training_inputs