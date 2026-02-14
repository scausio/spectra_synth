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

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetBlock(nn.Module):
    """Enhanced U-Net block with dropout and batch norm options"""

    def __init__(self, in_channels, out_channels, dropout=0.0, use_batchnorm=True):
        super().__init__()
        layers = []

        # First conv
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        # Second conv
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SpatialAttention(nn.Module):
    """Spatial attention for bottleneck"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        return x * attention


class SpectralUNet(nn.Module):
    """
    U-Net for 2D Wave Spectra Reconstruction

    Parameters:
    -----------
    input_dim : int
        Number of input features (wind-wave parameters)
    reshape_size : tuple
        Output shape (k_bins, theta_bins)
    hidden_dim : int
        Base number of channels (default: 128)
    num_blocks : int
        Number of encoder/decoder blocks (default: 4)
    channels : list, optional
        Custom channel progression [c1, c2, c3, c4, ...]
        If None, uses [hidden_dim, hidden_dim*2, hidden_dim*4, ...]
    dropout : float
        Dropout rate (default: 0.1)
    use_batchnorm : bool
        Use batch normalization (default: True)
    use_attention : bool
        Add spatial attention in bottleneck (default: True)
    skip_connection_mode : str
        'concat' (default) or 'add' for skip connections

    Example:
    --------
    # Deep model with custom channels
    model = SpectralUNet(
        input_dim=9,
        reshape_size=(32, 24),
        hidden_dim=512,
        num_blocks=4,
        channels=[64, 128, 256, 512],
        dropout=0.1,
        use_attention=True
    )
    """

    def __init__(
            self,
            input_dim,
            reshape_size,
            hidden_dim=128,
            num_blocks=4,
            channels=None,
            dropout=0.1,
            use_batchnorm=True,
            use_attention=True,
            skip_connection_mode='concat'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.reshape_size = reshape_size
        self.num_blocks = num_blocks
        self.skip_connection_mode = skip_connection_mode

        # Determine channel progression
        if channels is None:
            # Default: [hidden_dim, hidden_dim*2, hidden_dim*4, ...]
            self.channels = [hidden_dim * (2 ** i) for i in range(num_blocks)]
        else:
            if len(channels) < num_blocks:
                raise ValueError(f"channels list must have at least {num_blocks} elements")
            self.channels = channels[:num_blocks]

        # Initial projection: input_dim -> reshape_size with first channel
        initial_features = reshape_size[0] * reshape_size[1]
        self.fc_initial = nn.Sequential(
            nn.Linear(input_dim, initial_features * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(initial_features * 2, initial_features)
        )

        # Input conv: 1 channel -> first encoder channel
        self.input_conv = nn.Conv2d(1, self.channels[0], kernel_size=3, padding=1)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()

        for i in range(num_blocks):
            in_ch = self.channels[i] if i == 0 else self.channels[i - 1]
            out_ch = self.channels[i]

            self.encoder_blocks.append(
                UNetBlock(in_ch, out_ch, dropout, use_batchnorm)
            )

            if i < num_blocks - 1:  # No pooling after last encoder
                self.encoder_pools.append(nn.MaxPool2d(2))

        # Bottleneck with optional attention
        self.bottleneck = UNetBlock(
            self.channels[-1],
            self.channels[-1] * 2,
            dropout,
            use_batchnorm
        )

        if use_attention:
            self.attention = SpatialAttention(self.channels[-1] * 2)
        else:
            self.attention = None

        # Decoder blocks
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in range(num_blocks - 1, -1, -1):
            # Upsample
            in_ch = self.channels[-1] * 2 if i == num_blocks - 1 else self.channels[i + 1]
            out_ch = self.channels[i]

            self.decoder_upsamples.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )

            # Decoder block
            if skip_connection_mode == 'concat':
                decoder_in_ch = out_ch * 2  # Concatenated skip connection
            else:  # 'add'
                decoder_in_ch = out_ch

            self.decoder_blocks.append(
                UNetBlock(decoder_in_ch, out_ch, dropout, use_batchnorm)
            )

        # Output conv: first channel -> 1 channel
        self.output_conv = nn.Conv2d(self.channels[0], 1, kernel_size=1)

    def forward(self, x):
        # Initial projection: (batch, input_dim) -> (batch, k*theta)
        x = self.fc_initial(x)

        # Reshape to 2D: (batch, 1, k, theta)
        x = x.view(-1, 1, self.reshape_size[0], self.reshape_size[1])

        # Input conv
        x = self.input_conv(x)

        # Encoder with skip connections
        skip_connections = []

        for i in range(self.num_blocks):
            x = self.encoder_blocks[i](x)
            skip_connections.append(x)          # ✅ Save ALL encoder outputs
            
            if i < self.num_blocks - 1:
                x = self.encoder_pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        if self.attention is not None:
            x = self.attention(x)
            
        # Decoder with skip connections
        for i in range(self.num_blocks):  # ✅ Changed: was num_blocks-1
            # Upsample
            x = self.decoder_upsamples[i](x)
        
            # Get corresponding skip connection
            skip = skip_connections[-(i + 1)]
        
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
            # Combine with skip connection
            if self.skip_connection_mode == 'concat':
                x = torch.cat([x, skip], dim=1)
            else:  # 'add'
                x = x + skip
        
            # Decoder block
            x = self.decoder_blocks[i](x)
        
        # ✅ Final size adjustment (if needed)
        if x.shape[2:] != self.reshape_size:
            x = F.interpolate(x, size=self.reshape_size, mode='bilinear', align_corners=False)
        
        # Output conv
        x = self.output_conv(x)
        
        # Apply ReLU to ensure non-negative spectrum
        x = F.relu(x)
        
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
# 7. AUTOENCODER-BASED MODELS
# ============================================================================

class SpectralAutoencoder(nn.Module):
    """
    Standard Autoencoder for 2D Wave Spectra
    
    First trains on spectra reconstruction, then statistics are mapped to latent space.
    This two-stage approach ensures realistic outputs.
    
    Architecture:
    - Encoder: 2D Spectra -> Latent (64-128D)
    - Decoder: Latent -> 2D Spectra
    
    Use: Train in two stages:
      1. Train autoencoder to reconstruct spectra (unsupervised)
      2. Train mapper to predict latent from statistics (supervised)
    """
    
    def __init__(self,
                 spectra_shape=(32, 24),
                 latent_dim=64,
                 encoder_channels=[16, 32, 64, 128],
                 decoder_channels=[128, 64, 32, 16],
                 use_skip_connections=True):
        
        super(SpectralAutoencoder, self).__init__()
        
        self.spectra_shape = spectra_shape
        self.latent_dim = latent_dim
        self.use_skip_connections = use_skip_connections
        
        # =============== ENCODER ===============
        self.encoder = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        
        in_ch = 1
        for out_ch in encoder_channels:
            self.encoder.append(self._conv_block(in_ch, out_ch))
            self.encoder_pools.append(nn.MaxPool2d(2))
            in_ch = out_ch
        
        # Calculate flattened size
        h, w = spectra_shape
        for _ in encoder_channels:
            h = h // 2
            w = w // 2
        
        self.flat_size = encoder_channels[-1] * h * w
        
        # Single latent projection
        self.fc_encode = nn.Sequential(
            nn.Linear(self.flat_size, latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()  # Normalize latent space
        )
        
        # =============== DECODER ===============
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim * 2, self.flat_size),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        
        in_ch = encoder_channels[-1]
        for out_ch in decoder_channels:
            self.decoder_upsamples.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            dec_in_ch = out_ch * 2 if use_skip_connections and out_ch != decoder_channels[-1] else out_ch
            self.decoder.append(self._conv_block(dec_in_ch, out_ch))
            in_ch = out_ch
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 1, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def encode(self, x):
        """Encode to latent space"""
        skip_connections = []
        
        for conv, pool in zip(self.encoder, self.encoder_pools):
            x = conv(x)
            skip_connections.append(x)
            x = pool(x)
        
        x = x.view(x.size(0), -1)
        z = self.fc_encode(x)
        
        if self.use_skip_connections:
            return z, skip_connections
        else:
            return z, None
    
    def decode(self, z, skip_connections=None):
        """Decode from latent space"""
        x = self.fc_decode(z)
        
        batch_size = z.size(0)
        h, w = self.spectra_shape
        for _ in self.encoder:
            h = h // 2
            w = w // 2
        
        x = x.view(batch_size, -1, h, w)
        
        for i, (upsample, conv) in enumerate(zip(self.decoder_upsamples, self.decoder)):
            x = upsample(x)
            
            if skip_connections is not None and i < len(skip_connections) - 1:
                skip_idx = -(i + 2)
                skip = skip_connections[skip_idx]
                
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                
                x = torch.cat([x, skip], dim=1)
            
            x = conv(x)
        
        x = self.final_conv(x)
        
        if x.shape[2:] != self.spectra_shape:
            x = F.interpolate(x, size=self.spectra_shape, mode='bilinear', align_corners=False)
        
        return x
    
    def forward(self, x):
        """Full autoencoder forward pass"""
        z, skip_connections = self.encode(x)
        x_recon = self.decode(z, skip_connections)
        return x_recon
    
    def get_latent(self, x):
        """Get latent representation"""
        z, _ = self.encode(x)
        return z


class StatisticsToLatentMapper(nn.Module):
    """
    Maps wave statistics to autoencoder latent space
    
    Input: Wave parameters (Hs, Tm, Dir, etc.)
    Output: Latent vector compatible with trained autoencoder
    
    Training: Train AFTER autoencoder is trained and frozen
    """
    
    def __init__(self, 
                 input_dim=10,
                 latent_dim=64,
                 hidden_dims=[256, 512, 512, 256],
                 dropout=0.1,
                 use_attention=True):
        
        super(StatisticsToLatentMapper, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        # Optional attention
        if use_attention:
            self.attention = SelfAttention(hidden_dims[0])
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout)
            )
        
        # Output to latent space
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        z = self.output_proj(x)
        
        return z


class AutoencoderPipeline(BaseSpectralModel):
    """
    Complete Statistics -> Spectra pipeline using autoencoder
    
    This integrates with your existing workflow. Training procedure:
    
    Stage 1: Train autoencoder on spectra reconstruction
        model = SpectralAutoencoder(...)
        # Train with spectra only, no statistics needed
    
    Stage 2: Train mapper with frozen autoencoder
        mapper = StatisticsToLatentMapper(...)
        # Extract latent codes from autoencoder
        # Train mapper: statistics -> latent codes
    
    Stage 3: Create pipeline for inference
        pipeline = AutoencoderPipeline(autoencoder, mapper)
        # Now: statistics -> spectra end-to-end
    
    This model fits your existing workflow:
    - Input: [batch, input_dim] statistics
    - Output: [batch, 1, H, W] spectra
    """
    
    def __init__(self, input_dim, reshape_size, 
                 latent_dim=64, 
                 autoencoder_channels=[16, 32, 64, 128],
                 mapper_hidden=[256, 512, 512, 256],
                 pretrained_autoencoder=None,
                 pretrained_mapper=None):
        """
        Args:
            input_dim: Number of input statistics
            reshape_size: Output shape (H, W)
            latent_dim: Latent space dimension
            autoencoder_channels: Encoder/decoder channels
            mapper_hidden: Mapper hidden dimensions
            pretrained_autoencoder: Optional pre-trained autoencoder
            pretrained_mapper: Optional pre-trained mapper
        """
        super(AutoencoderPipeline, self).__init__(input_dim, reshape_size)
        
        # Create or load autoencoder
        if pretrained_autoencoder is not None:
            self.autoencoder = pretrained_autoencoder
        else:
            self.autoencoder = SpectralAutoencoder(
                spectra_shape=reshape_size,
                latent_dim=latent_dim,
                encoder_channels=autoencoder_channels,
                use_skip_connections=True
            )
        
        # Create or load mapper
        if pretrained_mapper is not None:
            self.mapper = pretrained_mapper
        else:
            self.mapper = StatisticsToLatentMapper(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=mapper_hidden
            )
    
    def forward(self, statistics):
        """
        Statistics -> Spectra
        
        Args:
            statistics: [batch, input_dim]
        Returns:
            spectra: [batch, 1, H, W]
        """
        # Map to latent
        z = self.mapper(statistics)
        
        # Decode to spectra
        spectra = self.autoencoder.decode(z)
        
        return spectra
    
    def freeze_autoencoder(self):
        """Freeze autoencoder for mapper training"""
        for param in self.autoencoder.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze for fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True


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
        'autoencoder': AutoencoderPipeline,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    model_class = models[model_name]
    return model_class(input_dim, reshape_size , **kwargs)
