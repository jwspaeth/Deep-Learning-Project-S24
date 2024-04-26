import os

import hydra
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

import dl_project
from dl_project.datasets import VITONDataModule


@hydra.main(version_base=None, config_name="train_dense")
def main(cfg: DictConfig):
    # data
    datamodule = VITONDataModule(**cfg.data)

    # model
    model = dl_project.utils.load_model(cfg)

    # logger
    logger = TensorBoardLogger(**cfg.logger)

    # train model
    trainer = L.Trainer(**cfg.trainer, logger=logger)
    assert cfg.core.mode in ["train", "val", "test", "predict"]
    if cfg.core.mode == "train":
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.core.ckpt_path)
    elif cfg.core.mode == "test":
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.core.ckpt_path)
    elif cfg.core.mode == "predict":
        trainer.predict(
            model=model, datamodule=datamodule, ckpt_path=cfg.core.ckpt_path
        )


if __name__ == "__main__":
    main()
