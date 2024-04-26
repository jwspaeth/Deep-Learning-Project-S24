import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from einops.layers.torch import Rearrange

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

def patchify(imgs):
    """
    imgs: (N, 3, H, W) x: (N, L, patch_size**2 *3)
    """
    p = 32 #self.vit.embeddings.patch_embeddings.patch_size[0]
    #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    #h = w = imgs.shape[2] // p
    h = imgs.shape[2] // p
    w = imgs.shape[3] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

def unpatchify(x):
    """
    x: (N, L, patch_size**2 *3) imgs: (N, 3, H, W)
    """
    p = 64
    """ h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1] """
    h = 1 # height // 64
    w = 1 # width // 64

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

class VisionTransformer(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, in_channels,out_channels, num_heads, num_layers, patch_size, num_patches, dropout=0.0):
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        self.patch_embedding = PatchEmbedding(self.in_channels, patch_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        # Layers/Networks
        #self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        #self.decoder = nn.Linear(embed_dim, num_channels * patch_size * patch_size)

        #self.upsample1 = nn.Linear(embed_dim, 512)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=self.out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        
        # Preprocess input
        #print()
        H = x.shape[2] // self.patch_size
        W = x.shape[3] // self.patch_size
        #print("Original Image:", x.shape)
        x = self.patch_embedding(x)    
        #x = patchify(x)
        #print("Patches:", x.shape)    
        B, T, _ = x.shape
        x += self.pos_embedding
        
        # Apply Transforrmer
        #x = self.dropout(x)
        x = x.transpose(0, 1)
        #print("Transformer Input:", x.shape)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        #print("Output transformer:", x.shape) # (N, B, E) (192,1,256)

        # Reshape (N,B,E) -> (N, L, h,w)
        x = x.reshape(B, self.embed_dim,H,W)
        #print("Reshaped:", x.shape)
        # Upsample
        out = self.upsample(x)
        #print("Output upsample:", out.shape)    
        

        # TODO: Make more complex decoder
        #x = self.decoder(x)
        #out = x.view(B, self.num_channels, 512, 512)

        return out

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
        model_output = self.model(cat_input)
        #print(model_output.shape)
        coarse_result = model_output #[:, :3, :, :]
        #cloth_mask = model_output[:,3:4,:,:]
        
        # Loss estimation
        target_img = batch["img"]
        #target_cloth = batch["cloth_mask"]["unpaired"]

        loss = F.mse_loss(coarse_result, target_img) #+ F.l1_loss(cloth_mask, target_cloth)

        self.log(f"{mode}_loss", loss)  

        #print(coarse_result.shape)
        if mode == "val":
            reconstructed_img_np = torchvision.utils.make_grid(coarse_result, normalize=True).cpu().detach().numpy()
            original_img_np = torchvision.utils.make_grid(target_img, normalize=True).cpu().detach().numpy()
            self.logger.experiment.add_image(f'{mode}_reconstructed_images', reconstructed_img_np, global_step=self.current_epoch)
            self.logger.experiment.add_image(f'{mode}_original_images', original_img_np, global_step=self.current_epoch)


        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
