import os
import shutil
from zipfile import ZipFile

import dropbox
from hydra import compose, initialize, initialize_config_dir

__all__ = ["get_config"]


def get_config(config_name, overrides=None):
    if overrides is None:
        overrides = []

    with initialize_config_dir(config_dir=f"{os.getcwd()}/config", version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def download_data(download_path, access_token):
    # Download zip file
    print("Downloading data.")
    dbx = dropbox.Dropbox(access_token)
    url = "https://www.dropbox.com/scl/fi/xu08cx3fxmiwpg32yotd7/zalando-hd-resized.zip?rlkey=ks83mdv2pvmrdl2oo2bmmn69w&e=2&dl=0"
    dbx.sharing_get_shared_link_file_to_file(
        download_path=f"{download_path}.zip", url=url
    )
    print("Download complete.")

    # # Extract zip
    print("Unpacking data from zip.")
    with ZipFile(f"{download_path}.zip", "r") as zObject:
        zObject.extractall(path=download_path)
    print("Data unpacked.")

    # Delete zip
    print("Removing zip.")
    os.remove(f"{download_path}.zip")
    print("Zip removed.")
