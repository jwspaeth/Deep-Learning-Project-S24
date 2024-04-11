import os

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None, config_path=f"{os.getcwd()}/config", config_name="default"
)
def main(cfg: DictConfig):
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset)

    # model
    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    # train model
    trainer = L.Trainer()
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
