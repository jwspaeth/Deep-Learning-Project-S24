import torch

import os

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from dl_project.datasets import VITONDataLoader, VITONDataModule, VITONDataset
from dl_project.models import DenseModel_Lit
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W] 
    return x


@hydra.main(
    version_base=None, config_path=f"{os.getcwd()}/config", config_name="train_dense"
)
def main(cfg: DictConfig):
    # data
    datamodule = VITONDataModule(**cfg.data)

    # Get the first batch of the training dataloader
    datamodule.setup(stage='fit')
    batch = next(iter(datamodule.train_dataloader()))

    image = batch["img_agnostic"]

    # Convert image to tensor
    image = torch.tensor(image)

    # Convert image to patches
    patch_size = 64
    patches = img_to_patch(image, patch_size, flatten_channels=True)

    # Convert patches back to image
    image_reconstructed = patches.reshape(image.shape)

    fig, ax = plt.subplots(image.shape[0], 1, figsize=(14,3))
    fig.suptitle("Images as input sequences of patches")
    for i in range(image.shape[0]):
        img_grid = torchvision.utils.make_grid(patches[i], nrow=192, normalize=True, pad_value=0.9)
        img_grid = img_grid.permute(1, 2, 0)
        ax[i].imshow(img_grid)
        ax[i].axis('off')
    plt.show()
    plt.close()
    

if __name__ == "__main__":
    main()




# Read image from /data/zalando/images/000000_0.jpg

# Convert image to tensor
