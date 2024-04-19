import os

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from dl_project.datasets import VITONDataLoader, VITONDataModule, VITONDataset
from dl_project.models import DenseModel_Lit


@hydra.main(
    version_base=None, config_path=f"{os.getcwd()}/config", config_name="train_dense"
)
def main(cfg: DictConfig):
    # data
    datamodule = VITONDataModule(**cfg.data)

    # model
    model = hydra.utils.instantiate(cfg.model)

    # train model
    trainer = L.Trainer(**cfg.trainer)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
