from pathlib import Path

import requests
from tqdm import tqdm

train_url = "https://pjreddie.com/media/files/mnist_train.csv"
test_url = "https://pjreddie.com/media/files/mnist_test.csv"

mnist_data_path = "datasets/mnist/"


def download_mnist_data(url, path):
    resp = requests.get(url, stream=True, timeout=10)
    total = int(resp.headers.get("content-length", 0))
    with (
        Path(path).open("wb") as file,
        tqdm(
            desc=f"downloading file to {path = }",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


if __name__ == "__main__":
    download_mnist_data(train_url, mnist_data_path + "mnist_train.csv")
    download_mnist_data(test_url, mnist_data_path + "mnist_test.csv")
