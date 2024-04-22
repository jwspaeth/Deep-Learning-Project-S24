import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_val, stride_val, padding_val):
        super().__init__()

        # The VITON U-Net contains 6 convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_val,
                stride=stride_val,
                padding=padding_val,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_val,
                stride=stride_val,
                padding=padding_val,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class unet_model(nn.Module):
    def __init__(
        self,
        in_channels=22,
        out_channels=4,
        encoder_channels=[64, 128, 256, 512],
        decoder_channels=[512, 256, 128, 64],
        kernel_val=3,
        encoder_stride=1,
        decoder_stride=1,
        padding_val=1,
    ):
        super(unet_model, self).__init__()

        # Encoder convolution
        self.enc_conv1 = conv_layer(
            in_channels, encoder_channels[0], kernel_val, encoder_stride, padding_val
        )
        self.enc_conv2 = conv_layer(
            encoder_channels[0],
            encoder_channels[1],
            kernel_val,
            encoder_stride,
            padding_val,
        )
        self.enc_conv3 = conv_layer(
            encoder_channels[1],
            encoder_channels[2],
            kernel_val,
            encoder_stride,
            padding_val,
        )
        self.enc_conv4 = conv_layer(
            encoder_channels[2],
            encoder_channels[3],
            kernel_val,
            encoder_stride,
            padding_val,
        )

        # Pooling and bottleneck
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.bottleneck = conv_layer(
            encoder_channels[3],
            encoder_channels[3] * 2,
            kernel_val,
            encoder_stride,
            padding_val,
        )

        # Decoder convolution
        self.dec_trans1 = nn.ConvTranspose2d(
            encoder_channels[3] * 2,
            decoder_channels[0],
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.dec_conv1 = conv_layer(
            encoder_channels[3] * 2,
            decoder_channels[0],
            kernel_val,
            decoder_stride,
            padding_val,
        )

        self.dec_trans2 = nn.ConvTranspose2d(
            decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2, padding=0
        )
        self.dec_conv2 = conv_layer(
            decoder_channels[0],
            decoder_channels[1],
            kernel_val,
            decoder_stride,
            padding_val,
        )

        self.dec_trans3 = nn.ConvTranspose2d(
            decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2, padding=0
        )
        self.dec_conv3 = conv_layer(
            decoder_channels[1],
            decoder_channels[2],
            kernel_val,
            decoder_stride,
            padding_val,
        )

        self.dec_trans4 = nn.ConvTranspose2d(
            decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2, padding=0
        )
        self.dec_conv4 = conv_layer(
            decoder_channels[2],
            decoder_channels[3],
            kernel_val,
            decoder_stride,
            padding_val,
        )

        # Final output
        self.conv_out = nn.Conv2d(
            decoder_channels[3], out_channels, kernel_size=1, padding=0
        )

    def forward(self, x):
        skip_connections = []

        s1 = self.enc_conv1.forward(x)
        p1 = self.pool(s1)
        skip_connections.append(s1)
        # print(f"S1: {s1.shape}. P1: {p1.shape}")

        s2 = self.enc_conv2.forward(p1)
        p2 = self.pool(s2)
        skip_connections.append(s2)
        # print(f"S2: {s2.shape}. P2: {p2.shape}")

        s3 = self.enc_conv3.forward(p2)
        p3 = self.pool(s3)
        skip_connections.append(s3)
        # print(f"S3: {s3.shape}. P3: {p3.shape}")

        s4 = self.enc_conv4.forward(p3)
        p4 = self.pool(s4)
        skip_connections.append(s4)
        # print(f"S4: {s4.shape}. P4: {p4.shape}")

        # Bottleneck
        bn = self.bottleneck(p4)
        # print(f"BN: {bn.shape}.")

        # Decoder
        s5 = self.dec_trans1(bn)
        cat1 = torch.cat([s5, skip_connections[3]], axis=1)
        s6 = self.dec_conv1.forward(cat1)
        # print(f"S5: {s5.shape}. Cat1: {cat1.shape}. S6: {s6.shape}")

        s7 = self.dec_trans2(s6)
        cat2 = torch.cat([s7, skip_connections[2]], axis=1)
        s8 = self.dec_conv2.forward(cat2)
        # print(f"S7: {s7.shape}. Cat2: {cat2.shape}. S8: {s8.shape}")

        s9 = self.dec_trans3(s8)
        cat3 = torch.cat([s9, skip_connections[1]], axis=1)
        s10 = self.dec_conv3.forward(cat3)
        # print(f"S9: {s9.shape}. Cat3: {cat3.shape}. S10: {s10.shape}")

        s11 = self.dec_trans4(s10)
        _, _, s11_h, s11_w = s11.shape
        skip_resize = skip_connections[0][:, :, :s11_h, :s11_w]
        cat4 = torch.cat([s11, skip_resize], axis=1)
        s12 = self.dec_conv4.forward(cat4)
        # print(f"S11: {s11.shape}. Cat4: {cat4.shape}. S12: {s12.shape}")

        # Final convolution
        x = self.conv_out(s12)

        return x


class UNetModel_Lit(L.LightningModule):
    def __init__(self, lr: int = 1e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr

        self.model = unet_model(**kwargs)

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
        coarse_result = model_output[:, :3, :, :]
        cloth_mask = model_output[:, 3:4, :, :]

        # Loss estimation
        target_img = batch["img"]
        target_cloth = batch["cloth_mask"]["unpaired"]

        loss = F.mse_loss(coarse_result, target_img) + F.l1_loss(
            cloth_mask, target_cloth
        )
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

        # Model
        model_output = self.model(cat_input)
        coarse_result = model_output[:, :3, :, :]
        cloth_mask = model_output[:, 3:4, :, :]

        # Loss estimation
        target_img = batch["img"]
        target_cloth = batch["cloth_mask"]["unpaired"]

        loss = F.mse_loss(coarse_result, target_img) + F.l1_loss(
            cloth_mask, target_cloth
        )
        self.log("loss", loss)

        return loss

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
        coarse_result = model_output[:, :3, :, :]
        cloth_mask = model_output[:, 3:4, :, :]

        # Loss estimation
        target_img = batch["img"]
        target_cloth = batch["cloth_mask"]["unpaired"]

        loss = F.mse_loss(coarse_result, target_img) + F.l1_loss(
            cloth_mask, target_cloth
        )
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

        # Model
        model_output = self.model(cat_input)
        coarse_result = model_output[:, :3, :, :]
        cloth_mask = model_output[:, 3:4, :, :]

        # Loss estimation
        target_img = batch["img"]
        target_cloth = batch["cloth_mask"]["unpaired"]

        loss = F.mse_loss(coarse_result, target_img) + F.l1_loss(
            cloth_mask, target_cloth
        )
        self.log("loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
