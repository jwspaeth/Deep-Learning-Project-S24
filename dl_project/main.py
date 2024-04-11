import os

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from dl_project.datasets import VITONDataLoader, VITONDataset
from dl_project.models import DenseModel_Lit


@hydra.main(
    version_base=None, config_path=f"{os.getcwd()}/config", config_name="default"
)
def main(cfg: DictConfig):
    # dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    # train_loader = DataLoader(dataset)
    dataset = VITONDataset(**cfg.dataset)
    train_loader = VITONDataLoader(**cfg.dataloader)

    # model
    model = DenseModel_Lit(**cfg.model)

    # train model
    trainer = L.Trainer()
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
