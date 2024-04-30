import os

import cv2
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .AutoencoderModel import unet_model


class Disciminator(nn.Module):
    def __init__(
        self,
        in_channels=3,
    ):
        super().__init__()

        # The VITON U-Net contains 6 convolutional layers
        self.conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    64,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    bias=True,
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    64,
                    128,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    bias=True,
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    128,
                    128,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    bias=True,
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    128,
                    256,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    bias=True,
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    256,
                    256,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    bias=True,
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    256,
                    256,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    bias=True,
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ]
        )

        self.dense = nn.ModuleList(
            [
                nn.Linear(8960, 100),
                nn.ReLU(),
                nn.Linear(100, 1),
            ]
        )

    def forward(self, ins):
        output = ins
        for i, layer in enumerate(self.conv):
            output = layer(output)

        output = torch.flatten(output, start_dim=1)

        for i, layer in enumerate(self.dense):
            output = layer(output)

        return output


def bce_loss(input, target, use_sigmoid=True):
    """
    Numerically stable version of the binary cross-entropy loss function in PyTorch.

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """

    def log_clamp(x):
        x = torch.log(x)
        x = torch.clamp(x, min=-100)
        return x

    if use_sigmoid:
        input = nn.functional.sigmoid(input)
    entries = target * log_clamp(input) + (1 - target) * log_clamp(1 - input)
    loss = -torch.mean(entries, dim=0)

    return loss


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None

    ones = torch.ones_like(logits_real)
    zeros = torch.zeros_like(logits_fake)
    real_loss = bce_loss(logits_real, ones, use_sigmoid=True)
    fake_loss = bce_loss(logits_fake, zeros, use_sigmoid=True)
    loss = real_loss + fake_loss

    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None

    ones = torch.ones_like(logits_fake)
    fake_loss = bce_loss(logits_fake, ones)
    loss = fake_loss

    return loss


class GAN_Lit(L.LightningModule):
    def __init__(
        self,
        lr: int = 1e-3,
        validation_cache_limit=None,
        automatic_optimization=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = automatic_optimization
        self.validation_cache_limit = validation_cache_limit
        self.lr = lr

        self.generator = unet_model(**kwargs)
        self.discriminator = Disciminator()

        self.validation_step_outputs_cr = []
        self.validation_step_outputs_cm = []

    @property
    def noise(self):
        # batch_size = batch_pose.shape[0]
        # noise = (torch.rand(size=(batch_size, dim)) * 2) - 1
        noise = (torch.rand(size=(8, 512, 32, 24)) * 2) - 1
        noise = noise.to(self.device)
        # noise = None
        return noise

    def training_step(self, batch, batch_idx):
        # Model inputs
        batch_pose = batch["pose"]
        batch_img_agnostic = batch["img_agnostic"]
        batch_parse_agnostic = batch["parse_agnostic"]
        batch_cloth = batch["cloth"]["unpaired"]
        cat_input = torch.cat(
            (batch_pose, batch_img_agnostic, batch_parse_agnostic, batch_cloth), dim=1
        )
        im_real = batch["img"]

        # Optimizers
        g_optim, d_optim = self.optimizers()

        # Model
        self.toggle_optimizer(g_optim)
        im_fake = self.generator(cat_input, noise=self.noise)
        pred_fake = self.discriminator(im_fake)
        pred_real = self.discriminator(im_real)

        # Loss estimation
        loss = F.mse_loss(im_fake, im_real)
        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # Model inputs
        batch_pose = batch["pose"]
        batch_img_agnostic = batch["img_agnostic"]
        batch_parse_agnostic = batch["parse_agnostic"]
        batch_cloth = batch["cloth"]["unpaired"]
        cat_input = torch.cat(
            (batch_pose, batch_img_agnostic, batch_parse_agnostic, batch_cloth), dim=1
        )
        im_real = batch["img"]

        # Model
        im_fake = self.generator(cat_input, noise=self.noise)
        # pred_fake = self.discriminator(im_fake)
        # pred_real = self.discriminator(im_real)

        # Loss estimation
        loss = F.mse_loss(im_fake, im_real)
        self.log("loss", loss)

        # Track output image
        if (
            self.validation_cache_limit is not None
            and self.cache_size < self.validation_cache_limit
        ):
            remainder = self.validation_cache_limit - self.cache_size
            # Target image
            image = im_real
            self.validation_step_outputs_cr.append(image[:remainder])

            # Model output image
            image = im_real
            self.validation_step_outputs_cm.append(im_fake[:remainder])

        return loss

    @property
    def cache_size(self):
        _cache_size = 0
        for entry in self.validation_step_outputs_cm:
            _cache_size += entry.shape[0]
        return _cache_size

    def on_validation_epoch_end(self):
        # Example of saving images to tensorboard
        if len(self.validation_step_outputs_cr) == 0:
            return

        all_preds_cr = torch.cat(self.validation_step_outputs_cr, dim=0)
        all_preds_cm = torch.cat(self.validation_step_outputs_cm, dim=0)
        self.logger.experiment.add_images(tag="images", img_tensor=all_preds_cr)
        self.logger.experiment.add_images(tag="images", img_tensor=all_preds_cm)
        self.validation_step_outputs_cr.clear()  # free memory
        self.validation_step_outputs_cm.clear()  # free memory

        # Example of saving images with cv2
        all_preds_cr = all_preds_cr.permute([0, 2, 3, 1]).cpu().numpy() * 255
        all_preds_cm = all_preds_cm.permute([0, 2, 3, 1]).cpu().numpy() * 255
        image_dir = f"{self.logger.save_dir}/images/epoch_{self.current_epoch}"
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        for i in range(all_preds_cr.shape[0]):
            pred = cv2.cvtColor(all_preds_cr[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{image_dir}/{i:04d}_target_img.png", pred)
        for i in range(all_preds_cm.shape[0]):
            pred = cv2.cvtColor(all_preds_cm[i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{image_dir}/{i:04d}_model_output.png", pred)

    def test_step(self, batch, batch_idx):
        # Model inputs
        batch_pose = batch["pose"]
        batch_img_agnostic = batch["img_agnostic"]
        batch_parse_agnostic = batch["parse_agnostic"]
        batch_cloth = batch["cloth"]["unpaired"]
        cat_input = torch.cat(
            (batch_pose, batch_img_agnostic, batch_parse_agnostic, batch_cloth), dim=1
        )
        im_real = batch["img"]

        # Model
        im_fake = self.generator(cat_input, noise=self.noise)
        # pred_fake = self.discriminator(im_fake)
        # pred_real = self.discriminator(im_real)

        # Loss estimation
        loss = F.mse_loss(im_fake, im_real)
        self.log("loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        # Model inputs
        batch_pose = batch["pose"]
        batch_img_agnostic = batch["img_agnostic"]
        batch_parse_agnostic = batch["parse_agnostic"]
        batch_cloth = batch["cloth"]["unpaired"]
        cat_input = torch.cat(
            (batch_pose, batch_img_agnostic, batch_parse_agnostic, batch_cloth), dim=1
        )
        im_real = batch["img"]

        # Model
        im_fake = self.generator(cat_input, noise=self.noise)
        # pred_fake = self.discriminator(im_fake)
        # pred_real = self.discriminator(im_real)

        # Loss estimation
        loss = F.mse_loss(im_fake, im_real)
        self.log("loss", loss)

        return loss

    def configure_optimizers(self):
        g_optim = torch.optim.Adam(
            self.generator.parameters(), lr=8e-4, betas=[0.5, 0.999]
        )
        d_optim = torch.optim.Adam(
            self.discriminator.parameters(), lr=8e-4, betas=[0.5, 0.999]
        )
        return [g_optim, d_optim], []
