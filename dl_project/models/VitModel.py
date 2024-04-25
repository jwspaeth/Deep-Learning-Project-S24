import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np


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

class AttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, 
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B, emb_size, H/P, W/P
        x = x.flatten(2)  # B, emb_size, N (N is number of patches)
        x = x.transpose(1, 2)  # B, N, emb_size
        return x

class VisionTransformer(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and 
                      on the input encoding
        """
        super().__init__()
        
        self.patch_size = patch_size

        self.patch_embedding = PatchEmbedding(num_channels, patch_size, embed_dim)
        
        # Layers/Networks
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        
        """ self.to_pixels = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, patch_size*patch_size*3)
        ) """
        
        # Position Embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1,num_patches,embed_dim))

        self.initial_projection = nn.Linear(256, 256 * 4)  # Upsample feature dimension

        self.conv_transpose1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsample to 24x32
        self.conv_transpose2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)   # Upsample to 48x64
        self.conv_transpose3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)    # Upsample to 96x128
        self.conv_transpose4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)    # Upsample to 192x256

        # Final convolution to get to 3 channels for an RGB image
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)
    
    
    def forward(self, x):
        
        # Preprocess input
        print("Original Image:", x.shape)
        x = self.patch_embedding(x)    
        print("Patches:", x.shape)    
        B, T, _ = x.shape
        x += self.pos_embedding
        
        # Apply Transforrmer
        #x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        print("Output transformer:", x.shape)

        
        # Upsample
        # TODO: Change for batchsize
        x = x.view(-1, 256)
        print(x.shape)
        x = self.initial_projection(x)  # Map features to a higher dimension
        x = x.view(-1, 12, 16, 256)  # Reshape to spatial dimensions (assuming the patches were 12x16 grid), with batch handling
        
        # Apply transposed convolutions
        x = F.relu(self.conv_transpose1(x))
        x = F.relu(self.conv_transpose2(x))
        x = F.relu(self.conv_transpose3(x))
        x = F.relu(self.conv_transpose4(x))

        # Final convolution to get the RGB image
        out = self.final_conv(x)

        return out

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
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_val,
                stride=stride_val,
                padding=padding_val,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Vit_Lit(L.LightningModule):
    
    def __init__(self, lr: int = 1e-3, **kwargs):
        super().__init__()
        self.lr = lr
        print(kwargs)
        self.model = VisionTransformer(**kwargs)
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        return [optimizer], [lr_scheduler]   
    
    def _calculate_loss(self, batch, mode="train"):
        batch_pose = batch["pose"]
        batch_img_agnostic = batch["img_agnostic"]
        batch_parse_agnostic = batch["parse_agnostic"]
        batch_cloth = batch["cloth"]["unpaired"]
        cat_input = torch.cat((batch_pose,batch_img_agnostic,batch_parse_agnostic,batch_cloth),dim=1)
        
        # Model
        model_output = self.model(batch["img_agnostic"])
        coarse_result = model_output[:, :3, :, :]
        cloth_mask = model_output[:,3:4,:,:]
        
        # Loss estimation
        target_img = batch["img"]
        target_cloth = batch["cloth_mask"]["unpaired"]

        loss = F.mse_loss(coarse_result, target_img) + F.l1_loss(cloth_mask, target_cloth)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
