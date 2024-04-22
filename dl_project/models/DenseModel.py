import os
from typing import List

import cv2
import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["DenseModel", "DenseModel_Lit"]


class DenseModel(nn.Module):
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        # Build layers
        all_sizes = [input_size] + hidden_layers + [output_size]
        self.layers = nn.ModuleList()
        for i in range(len(all_sizes) - 1):
            input_size = all_sizes[i]
            output_size = all_sizes[i + 1]
            self.layers.append(nn.Linear(input_size, output_size))

            if i != len(all_sizes) - 1:
                self.layers.append(nn.LeakyReLU())

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)

        return output


class DenseModel_Lit(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        lr: int = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr

        self.model = DenseModel(input_size, hidden_layers, output_size)
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        # Define dummy training to verify training works
        image = batch["img"]
        x = image[:, 0, :10, 0]
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Define dummy training to verify training works
        image = batch["img"]
        x = image[:, 0, :10, 0]
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x)
        self.log("loss", loss)

        self.validation_step_outputs.append(image)
        return loss

    def on_validation_epoch_end(self):
        # Example of saving images to tensorboard
        all_preds = torch.cat(self.validation_step_outputs, dim=0)
        self.logger.experiment.add_images(tag="images", img_tensor=all_preds)
        self.validation_step_outputs.clear()  # free memory

        # Example of saving images with cv2
        all_preds = all_preds.permute([0, 2, 3, 1]).cpu().numpy() * 255
        image_dir = f"{self.logger.save_dir}/images/epoch_{self.current_epoch}"
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        for i in range(all_preds.shape[0]):
            pred = cv2.cvtColor(all_preds[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{image_dir}/{i:04d}.png", pred)

    def test_step(self, batch, batch_idx):
        # Define dummy training to verify training works
        image = batch["img"]
        x = image[:, 0, :10, 0]
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x)
        self.log("loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        # Define dummy training to verify training works
        image = batch["img"]
        x = image[:, 0, :10, 0]
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
