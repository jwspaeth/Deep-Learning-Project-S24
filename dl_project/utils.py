import os

from hydra import compose, initialize, initialize_config_dir

__all__ = ["get_config"]


def get_config(config_name, overrides=None):
    if overrides is None:
        overrides = []

    with initialize_config_dir(config_dir=f"{os.getcwd()}/config", version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg
