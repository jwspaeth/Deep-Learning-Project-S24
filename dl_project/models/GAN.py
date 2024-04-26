import os

import cv2
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dl_project.models import unet_model


class Disciminator(nn.Module):
    def __init__():
        super().__init__()


class GAN_Lit(L.LightningModule):
    def __init__(self, lr: int = 1e-3, validation_cache_limit=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.validation_cache_limit = validation_cache_limit
        self.lr = lr

        self.generator = unet_model(**kwargs)
        self.discriminator = Disciminator()

        self.validation_step_outputs_cr = []
        self.validation_step_outputs_cm = []

    def training_step(self, batch, batch_idx):
        # Model inputs
        batch_pose = batch["pose"]
        batch_img_agnostic = batch["img_agnostic"]
        batch_parse_agnostic = batch["parse_agnostic"]
        batch_cloth = batch["cloth"]["unpaired"]
        cat_input = torch.cat(
            (batch_pose, batch_img_agnostic, batch_parse_agnostic, batch_cloth), dim=1
        )

        # Model
        model_output = self.model(cat_input)
        # coarse_result = model_output[:, :3, :, :]
        # cloth_mask = model_output[:, 3:4, :, :]

        # Loss estimation
        target_img = batch["img"]
        # target_cloth = batch["cloth_warped_mask"]["unpaired"]

        # loss = F.mse_loss(coarse_result, target_img) + F.l1_loss(
        #     cloth_mask, target_cloth
        # )
        loss = F.mse_loss(model_output, target_img)
        self.log("loss", loss)

        # Track output image
        # image = model_output[:, :3, :, :]
        # image = batch["cloth_warped_mask"]["unpaired"]
        # self.validation_step_outputs_cr.append(model_output)
        # image = model_output[:, 3:4, :, :]
        # image = batch_cloth

        # Target image
        # image = target_img
        # self.validation_step_outputs_cr.append(image)

        # # Model output image
        # image = target_img
        # self.validation_step_outputs_cm.append(model_output)

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

        # Model
        model_output = self.model(cat_input)
        # coarse_result = model_output[:, :3, :, :]
        # cloth_mask = model_output[:, 3:4, :, :]

        # Loss estimation
        target_img = batch["img"]
        # target_cloth = batch["cloth_warped_mask"]["unpaired"]

        # loss = F.mse_loss(coarse_result, target_img) + F.l1_loss(
        #     cloth_mask, target_cloth
        # )
        loss = F.mse_loss(model_output, target_img)
        self.log("loss", loss)

        # Track output image
        # image = model_output[:, :3, :, :]
        # image = batch["cloth_warped_mask"]["unpaired"]
        # self.validation_step_outputs_cr.append(model_output)
        # image = model_output[:, 3:4, :, :]
        # image = batch_cloth
        if (
            self.validation_cache_limit is not None
            and self.cache_size < self.validation_cache_limit
        ):
            remainder = self.validation_cache_limit - self.cache_size
            # Target image
            image = target_img
            self.validation_step_outputs_cr.append(image[:remainder])

            # Model output image
            image = target_img
            self.validation_step_outputs_cm.append(model_output[:remainder])

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

        # Model
        model_output = self.model(cat_input)
        # coarse_result = model_output[:, :3, :, :]
        # cloth_mask = model_output[:, 3:4, :, :]

        # Loss estimation
        target_img = batch["img"]
        # target_cloth = batch["cloth_warped_mask"]["unpaired"]

        # loss = F.mse_loss(coarse_result, target_img) + F.l1_loss(
        #     cloth_mask, target_cloth
        # )
        loss = F.mse_loss(model_output, target_img)
        self.log("loss", loss)

        # Track output image
        # image = model_output[:, :3, :, :]
        # image = batch["cloth_warped_mask"]["unpaired"]
        # image = target_img
        # self.validation_step_outputs_cr.append(image)
        # self.validation_step_outputs_cr.append(model_output)
        # image = model_output[:, 3:4, :, :]
        # image = batch_cloth
        # self.validation_step_outputs_cm.append(image)

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

        # Model
        model_output = self.model(cat_input)
        # coarse_result = model_output[:, :3, :, :]
        # cloth_mask = model_output[:, 3:4, :, :]

        # Loss estimation
        target_img = batch["img"]
        # target_cloth = batch["cloth_warped_mask"]["unpaired"]

        # loss = F.mse_loss(coarse_result, target_img) + F.l1_loss(
        #     cloth_mask, target_cloth
        # )
        loss = F.mse_loss(model_output, target_img)
        self.log("loss", loss)

        # Track output image
        # image = model_output[:, :3, :, :]
        # image = batch["cloth_warped_mask"]["unpaired"]
        # # image = target_img
        # # self.validation_step_outputs_cr.append(image)
        # self.validation_step_outputs_cr.append(model_output)
        # # image = model_output[:, 3:4, :, :]
        # # image = batch_cloth
        # # self.validation_step_outputs_cm.append(image)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
