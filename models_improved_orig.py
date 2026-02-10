"""
Improved Model Architectures for 2D Wave Spectra Reconstruction

This module contains several advanced neural network architectures for reconstructing
2D wave spectra from mean parameters (Hs, Tm, Dir for wind waves and swells).

Input: N-dimensional vector (default 10: 9 wave parameters + depth)
Output: 2D spectra (frequency x direction space, default 32x24)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import random

# Set seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================================
# Base Model Class
# ============================================================================

class BaseSpectralModel(nn.Module):
    """Base class for all spectral reconstruction models"""
    
    def __init__(self, input_dim, output_shape):
        super(BaseSpectralModel, self).__init__()
        self.input_dim = input_dim
        self.output_shape = output_shape
        self.output_dim = output_shape[0] * output_shape[1]

    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")


# ============================================================================
# 1. Enhanced Fully Connected Network with Attention
# ============================================================================

class AttentionFFNN(BaseSpectralModel):
    """
    Enhanced FFNN with self-attention mechanism and residual connections.
    Good baseline model with moderate complexity.
    """
    
    def __init__(self, input_dim, reshape_size,
                 hidden_dims=[512, 512, 256, 256], dropout=0.1):

        super(AttentionFFNN, self).__init__(input_dim, reshape_size)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        # Self-attention layer
        self.attention = SelfAttention(hidden_dims[0])
        
        # Deep residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout)
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.output_dim),
            nn.ReLU()  # Ensure positive output
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.attention(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        x = self.output_proj(x)
        x = x.view(-1, 1, self.output_shape[0], self.output_shape[1])
        return x


# ============================================================================
# 2. Transformer-Based Model
# ============================================================================

class SpectralTransformer(BaseSpectralModel):
    """
    Transformer-based architecture that treats the input parameters as a sequence.
    Excellent for capturing long-range dependencies between parameters.
    """
    
    def __init__(self,input_dim, reshape_size,
                 d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super(SpectralTransformer, self).__init__(input_dim, reshape_size)
        
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=input_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder to 2D spectra
        self.decoder = nn.Sequential(
            nn.Linear(d_model * input_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, self.output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape to (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # Embed each input feature
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Flatten and decode
        x = x.view(batch_size, -1)
        x = self.decoder(x)
        x = x.view(-1, 1, self.output_shape[0], self.output_shape[1])
        
        return x


# ============================================================================
# 3. U-Net Style Architecture
# ============================================================================
class SpectralUNet(BaseSpectralModel):
    """
    UNet completa per ricostruzione di spettri 2D (k, theta)
    Input  : vettore di statistiche [B, input_dim]
    Output : spettro EF [B, 1, Nk, Ntheta]
    """

    def __init__(
        self,
        input_dim,
        reshape_size,            # (Nk, Ntheta)
        hidden_dim=256,
        channels=(16, 32, 64, 128),num_blocks=3,
    ):
        super().__init__(input_dim, reshape_size)

        self.Nk, self.Ntheta = reshape_size

        # ================= Input projection =================
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, self.Nk * self.Ntheta),
        )

        # ================= Encoder =================
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = 1
        for ch in channels:
            self.enc_blocks.append(self._conv_block(in_ch, ch))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = ch

        # ================= Bottleneck =================
        self.bottleneck = self._conv_block(channels[-1], channels[-1] * 2)

        # ================= Decoder =================
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        in_ch = channels[-1] * 2
        for ch in reversed(channels):
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, ch, kernel_size=2, stride=2)
            )
            self.dec_blocks.append(
                self._conv_block(ch * 2, ch)
            )
            in_ch = ch

        # ================= Output =================
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0], 1, kernel_size=1),
            nn.ReLU()   # EF >= 0
        )

    # ------------------------------------------------------
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    # ------------------------------------------------------
    def forward(self, x):
        B = x.size(0)

        # ---- Project input to 2D ----
        x = self.input_proj(x)
        x = x.view(B, 1, self.Nk, self.Ntheta)

        # ---- Encoder ----
        skips = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # ---- Bottleneck ----
        x = self.bottleneck(x)

        # ---- Decoder ----
        skips = skips[::-1]
        for up, dec, skip in zip(self.upconvs, self.dec_blocks, skips):
            x = up(x)
            x = self._match_size(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        # ---- Output ----
        x = self.final_conv(x)
        return x

    # ------------------------------------------------------
    def _match_size(self, x, ref):
        """Crop or pad x to match ref spatial size"""
        dh = ref.size(2) - x.size(2)
        dw = ref.size(3) - x.size(3)

        if dh != 0 or dw != 0:
            x = F.pad(
                x,
                [
                    dw // 2, dw - dw // 2,
                    dh // 2, dh - dh // 2
                ]
            )
        return x
# ============================================================================
# 4. Enhanced ResNet-Style Architecture
# ============================================================================

class SpectralResNet(BaseSpectralModel):
    """
    Deep residual network with improved skip connections.
    Best for very deep networks and complex patterns.
    """
    
    def __init__(self, input_dim, reshape_size,
                 hidden_dim=512, num_res_blocks=8, num_conv_blocks=4):

        super(SpectralResNet, self).__init__(input_dim, reshape_size)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Deep residual blocks
        self.res_blocks = nn.ModuleList([
            ImprovedResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        
        # Transition to 2D
        self.to_2d = nn.Sequential(
            nn.Linear(hidden_dim, self.output_dim),
            nn.LeakyReLU(0.2)
        )
        
        # 2D convolutional refinement
        self.conv_blocks = nn.ModuleList()
        channels = [1, 32, 64, 32, 16][:num_conv_blocks+1]
        for i in range(len(channels) - 1):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], 3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.LeakyReLU(0.2)
            ))
        
        # Final output
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels[-1], 1, 3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Input processing
        x = self.input_proj(x)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # To 2D
        x = self.to_2d(x)
        x = x.view(-1, 1, self.output_shape[0], self.output_shape[1])
        
        # 2D refinement
        for block in self.conv_blocks:
            x = block(x)
        
        x = self.output_conv(x)
        return x


# ============================================================================
# 5. Hybrid CNN-Attention Model
# ============================================================================

class HybridCNNAttention(BaseSpectralModel):
    """
    Combines CNN for spatial processing with attention for feature importance.
    Balanced approach with good performance.
    """
    
    def __init__(self, input_dim, reshape_size,
                 hidden_dim=512, num_res_blocks=4):

        super(HybridCNNAttention, self).__init__(input_dim, reshape_size)
        
        # Input processing with attention
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        self.attention = SelfAttention(hidden_dim)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ImprovedResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        
        # Spatial attention
        self.spatial_attention = SpatialAttention()
        
        # To 2D
        self.to_2d = nn.Linear(hidden_dim, self.output_dim)
        
        # 2D processing
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 1, 3, padding=1),
        )
        
        self.output_activation = nn.ReLU()
    
    def forward(self, x):
        # Input with attention
        x = self.input_proj(x)
        x = self.attention(x)
        
        # Residual processing
        for block in self.res_blocks:
            x = block(x)
        
        # To 2D
        x = self.to_2d(x)
        x = x.view(-1, 1, self.output_shape[0], self.output_shape[1])
        
        # Spatial attention
        x = self.spatial_attention(x)
        
        # 2D refinement
        x = self.conv_blocks(x)
        x = self.output_activation(x)
        
        return x


# ============================================================================
# 6. Lightweight Fast Model
# ============================================================================

class LightweightSpectralNet(BaseSpectralModel):
    """
    Efficient lightweight model for fast inference.
    Good for deployment and real-time applications.
    """
    
    def __init__(self, input_dim, reshape_size, hidden_dim=256):

        super(LightweightSpectralNet, self).__init__(input_dim, reshape_size)
        
        self.network = nn.Sequential(
            # Efficient input processing
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            
            # Depthwise separable conv equivalent
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim // 2, self.output_dim),
        )
        
        # Lightweight 2D refinement
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.network(x)
        x = x.view(-1, 1, self.output_shape[0], self.output_shape[1])
        x = self.refine(x)
        return x


# ============================================================================
# Helper Modules
# ============================================================================

class SelfAttention(nn.Module):
    """Self-attention mechanism for feature importance"""
    
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Attention scores
        if len(x.shape) == 2:
            Q = Q.unsqueeze(1)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        
        if len(x.shape) == 2:
            out = out.squeeze(1)
        
        return out


class SpatialAttention(nn.Module):
    """Spatial attention for 2D feature maps"""
    
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
    
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        attention = torch.sigmoid(self.conv(pool))
        return x * attention


class ResidualBlock(nn.Module):
    """Residual block with projection"""
    
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)
        
        # Projection for skip connection if dimensions differ
        self.projection = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
    
    def forward(self, x):
        identity = self.projection(x) if self.projection else x
        
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.norm2(out)
        
        out = out + identity
        out = self.activation(out)
        
        return out


class ImprovedResidualBlock(nn.Module):
    """Improved residual block with better gradient flow"""
    
    def __init__(self, hidden_dim, dropout=0.1):
        super(ImprovedResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        identity = x
        
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.dropout(out)
        
        return self.activation(out + identity)


class UNetBlock(nn.Module):
    """Basic U-Net convolutional block"""
    
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# Model Factory
# ============================================================================

def get_model(model_name, input_dim, reshape_size, **kwargs):
    """
    Factory function to create models by name.
    
    Args:
        model_name: Name of the model architecture
        X_train_scaled: Training data for shape inference
        reshape_size: Output shape (k_bins, theta_bins)
        scalerX, scalerY: Data scalers
        **kwargs: Additional model-specific parameters
    
    Returns:
        Initialized model
    """
    models = {
        'attention_ffnn': AttentionFFNN,
        'transformer': SpectralTransformer,
        'unet': SpectralUNet,
        'resnet': SpectralResNet,
        'hybrid': HybridCNNAttention,
        'lightweight': LightweightSpectralNet,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    model_class = models[model_name]
    return model_class(input_dim, reshape_size , **kwargs)
