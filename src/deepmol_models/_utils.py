import os
import sys
import zipfile
import requests
from scipy.stats import spearmanr
from tqdm import tqdm


def spearman(x, y):
    return spearmanr(x, y)[0]

def _download_and_save(url, zip_file_name, cache_folder_name):
    response = requests.get(url, stream=True)
    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    cache_folder = os.path.join(os.path.expanduser("~"), cache_folder_name)
    os.makedirs(cache_folder, exist_ok=True)
    file_path = os.path.join(cache_folder, zip_file_name)

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(file_path, "wb") as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)

    # unzip pipeline
    print(f"Unzipping models...")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(cache_folder)

    os.remove(file_path)


def download_models():
    url = "https://zenodo.org/records/14573331/files/models.zip?download=1"
    _download_and_save(url, "models.zip", ".deepmol_case_studies")

def download_deployed_models(model, download_url):
    url = download_url
    _download_and_save(url, f"{model}.zip", ".deepmol_models")