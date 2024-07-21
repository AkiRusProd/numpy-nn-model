from pathlib import Path

import requests
from tqdm import tqdm

#References: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/datasets/multi30k.html

def download_multi30k_data(urls, path, filenames):
    for _, (url, filename) in enumerate(zip(urls, filenames, strict=False)):
        resp = requests.get(url, stream=True, verify=False, timeout=10)
        total = int(resp.headers.get('content-length', 0))
        with open(Path(path) / filename, 'wb') as file, tqdm(
                desc = f'downloading {filename = } to {path = }',
                total = total,
                unit = 'iB',
                unit_scale = True,
                unit_divisor = 1024,
        ) as bar:
            for data in resp.iter_content(chunk_size = 1024):
                size = file.write(data)
                bar.update(size)


if __name__ == '__main__':
    urls = {
        "train": r"https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
        "valid": r"https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
        "test": r"https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz",
    }

    path = 'datasets/multi30k/'
    filenames = ["mmt16_task1_test.tar.gz", "training.tar.gz", "validation.tar.gz"]

    download_multi30k_data(urls.values(), path, filenames)
