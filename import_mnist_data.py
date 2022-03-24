import requests
from tqdm import tqdm

train_url = 'https://pjreddie.com/media/files/mnist_train.csv'
test_url = 'https://pjreddie.com/media/files/mnist_test.csv'

def download(url, path):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(path, 'wb') as file, tqdm(
            desc = f'downloading file to {path = }',
            total = total,
            unit = 'iB',
            unit_scale = True,
            unit_divisor = 1024,
    ) as bar:
        for data in resp.iter_content(chunk_size = 1024):
            size = file.write(data)
            bar.update(size)

download(train_url, 'dataset/mnist_train.csv')
download(test_url, 'dataset/mnist_test.csv')