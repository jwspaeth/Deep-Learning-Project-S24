from typing import List

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
        self.lr = lr

        self.model = DenseModel(input_size, hidden_layers, output_size)

    def training_step(self, batch, batch_idx):
        # Define dummy training to verify training works
        pose = batch["pose"]
        x = pose[:, 0, :10, 0]
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Define dummy training to verify training works
        pose = batch["pose"]
        x = pose[:, 0, :10, 0]
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x)
        self.log("loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # Define dummy training to verify training works
        pose = batch["pose"]
        x = pose[:, 0, :10, 0]
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x)
        self.log("loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        # Define dummy training to verify training works
        pose = batch["pose"]
        x = pose[:, 0, :10, 0]
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
