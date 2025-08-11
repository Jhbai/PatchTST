import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import math

class PatchTST(nn.Module):
    def __init__(self, n_channels=7, seq_len=512, patch_len=12, d_model=128, n_heads=16, n_layers=3, d_ff=256, dropout=0.2, masking_ratio=0.4):
        super().__init__()
        
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.masking_ratio = masking_ratio
        
        # Non-Overlapping Patch
        self.stride = patch_len
        
        # Truncate the time series if needed
        self.n_patches = seq_len // patch_len
        self.truncated_len = self.n_patches * self.patch_len
        
        # Instance Normalization
        self.instance_norm = nn.InstanceNorm1d(self.truncated_len)
        
        # Patch -> d_model
        self.projection = nn.Linear(patch_len, d_model)
        
        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu', # Paper used GELU
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Mask Reconstruction Head
        self.reconstruction_head = nn.Linear(d_model, patch_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in):
        # x_in: (B, M, L)
        B, M, L = x_in.shape
        
        # x: (B, M, L_trunc)
        x = x_in[:, :, :self.truncated_len]

        # Instance Normalization, x_norm: (B * M, 1, L_trunc)
        x = x.reshape(B * M, 1, self.truncated_len)
        x_norm = self.instance_norm(x)
        
        # x_norm: (B * M, L_trunc)
        x_norm = x_norm.squeeze(1)
        # patches: (B * M, N, P)
        patches = x_norm.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Projection to d_model, patches_proj: (B * M, N, D), x_encoded: (B * M, N, D)
        patches_proj = self.projection(patches)
        x_encoded = self.dropout(patches_proj + self.pos_embedding)

        
        # Masking
        mask = None
        if self.training:
            n_masked_patches = int(self.masking_ratio * self.n_patches)
            # Use Random Noise to determine which patches to mask, noise: (B * M, N)
            noise = torch.rand(B * M, self.n_patches, device=x_in.device)
            masked_indices = torch.argsort(noise, dim=-1)[:, :n_masked_patches]
            mask = torch.zeros((B * M, self.n_patches), dtype=torch.bool, device=x_in.device)
            mask.scatter_(1, masked_indices, True) # Let the masked indices be True, mask: (B * M, N)
            x_encoded[mask] = 0

        # z: (B * M, N, D)
        z = self.transformer_encoder(x_encoded)

        # reconstructed_patches: (B * M, N, P)
        reconstructed_patches = self.reconstruction_head(z)

        return reconstructed_patches, mask