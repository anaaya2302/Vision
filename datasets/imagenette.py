import os
import tarfile
import urllib.request

from tqdm import tqdm

# Same tarball fastai's untar_data(URLs.IMAGENETTE) fetches, without the fastai dependency
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclassification/imagenette2.tgz"
DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))


def download_imagenette(dest=DATASETS_DIR):
    path = os.path.join(dest, "imagenette2")
    if os.path.isdir(os.path.join(path, "train")):
        print(f"Dataset already present at: {path}")
        return path

    tgz_path = os.path.join(dest, "imagenette2.tgz")
    print(f"Downloading {IMAGENETTE_URL} ...")
    with tqdm(unit="B", unit_scale=True, desc="imagenette2.tgz") as bar:
        def report(blocks, block_size, total):
            bar.total = total
            bar.update(blocks * block_size - bar.n)
        urllib.request.urlretrieve(IMAGENETTE_URL, tgz_path, reporthook=report)

    print("Extracting...")
    with tarfile.open(tgz_path) as tar:
        tar.extractall(dest)
    os.remove(tgz_path)
    return path


if __name__ == "__main__":
    path = download_imagenette()
    print(f"Dataset is at: {path}")
    print(f"Training folder: {path}/train")
    print(f"Validation folder: {path}/val")

    classes = os.listdir(os.path.join(path, "train"))
    print(f"Classes: {classes}")
