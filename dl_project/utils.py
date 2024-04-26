import importlib
import os
import shutil
from zipfile import ZipFile

import dropbox
import hydra
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf, open_dict

__all__ = ["get_config", "unpack_cfg", "load_model"]


def unpack_cfg(cfg):
    while "core" not in cfg:
        keys = list(cfg.keys())
        cfg = cfg[keys[0]]
    # cfg = OmegaConf.resolve(cfg)

    return cfg


def load_model(cfg):
    # Import model class
    import_split = cfg.model._target_.split(".")
    model_name = import_split[-1]
    module_path = ".".join(import_split[:-1])
    module = importlib.import_module(module_path)
    model_class = getattr(module, model_name)

    # Pop target from model
    kwargs = cfg.model
    with open_dict(kwargs):
        kwargs.pop("_target_")

    # Instantiate depending on if checkpoint is present
    if cfg.core.ckpt_path is not None:
        model = model_class.load_from_checkpoint(cfg.core.ckpt_path)
    else:
        model = model_class(**kwargs)
    return model


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
