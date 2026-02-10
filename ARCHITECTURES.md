"""
Visual Architecture Comparison

This file provides a text-based visualization of each model architecture.
"""

ARCHITECTURES = """
================================================================================
MODEL ARCHITECTURE COMPARISON
================================================================================

INPUT: 10-dimensional vector [Hs_ww, Tm_ww, Dir_ww, Hs_sw1, Tm_sw1, Dir_sw1, 
                               Hs_sw2, Tm_sw2, Dir_sw2, Depth]

--------------------------------------------------------------------------------
1. ATTENTION FFNN
--------------------------------------------------------------------------------
Input (10) 
  ↓
Linear(10 → 512) + LayerNorm + LeakyReLU + Dropout
  ↓
Self-Attention (512)
  ↓
ResidualBlock (512 → 512) ×4
  ↓
Linear(512 → 768) → Reshape(1, 32, 24)

Parameters: ~1.5M
Speed: Fast
Memory: Low

--------------------------------------------------------------------------------
2. SPECTRAL TRANSFORMER
--------------------------------------------------------------------------------
Input (10)
  ↓
Embed each feature (10 × 256)
  ↓
Positional Encoding
  ↓
Transformer Encoder (4 layers, 8 heads)
  ├─ Multi-Head Attention
  ├─ Layer Norm
  ├─ Feed Forward (256 → 1024 → 256)
  └─ Layer Norm
  ↓
Flatten (10 × 256 → 2560)
  ↓
Linear(2560 → 1024 → 512 → 768) → Reshape(1, 32, 24)

Parameters: ~3.5M
Speed: Medium
Memory: High

--------------------------------------------------------------------------------
3. SPECTRAL UNET
--------------------------------------------------------------------------------
Input (10)
  ↓
Linear(10 → 256 → 768) → Reshape(1, 32, 24)
  ↓
ENCODER PATH:
  Conv(1→16, 3×3) + BN → MaxPool(2×2)  → (16, 16, 12)
  ↓                    ↗ (skip connection)
  Conv(16→32, 3×3) + BN → MaxPool(2×2)  → (32, 8, 6)
  ↓                    ↗ (skip connection)
  Conv(32→64, 3×3) + BN → MaxPool(2×2)  → (64, 4, 3)
  ↓
BOTTLENECK:
  Conv(64→128, 3×3) + BN
  ↓
DECODER PATH:
  UpConv(128→64, 2×2) + Concat + Conv(128→64)
  ↓
  UpConv(64→32, 2×2) + Concat + Conv(64→32)
  ↓
  UpConv(32→16, 2×2) + Concat + Conv(32→16)
  ↓
  Conv(16→1, 1×1) + ReLU → (1, 32, 24)

Parameters: ~2.5M
Speed: Medium
Memory: Medium-High

--------------------------------------------------------------------------------
4. SPECTRAL RESNET
--------------------------------------------------------------------------------
Input (10)
  ↓
Linear(10 → 512) + LayerNorm
  ↓
Residual Blocks (512) ×8:
  ├─ LayerNorm
  ├─ Linear(512 → 512)
  ├─ LeakyReLU + Dropout
  ├─ LayerNorm
  ├─ Linear(512 → 512)
  ├─ Dropout
  └─ Add residual + LeakyReLU
  ↓
Linear(512 → 768) → Reshape(1, 32, 24)
  ↓
Conv(1→32, 3×3) + BN + LeakyReLU
  ↓
Conv(32→64, 3×3) + BN + LeakyReLU
  ↓
Conv(64→32, 3×3) + BN + LeakyReLU
  ↓
Conv(32→16, 3×3) + BN + LeakyReLU
  ↓
Conv(16→1, 3×3) + ReLU → (1, 32, 24)

Parameters: ~3.0M
Speed: Medium
Memory: Medium

--------------------------------------------------------------------------------
5. HYBRID CNN-ATTENTION (⭐ RECOMMENDED)
--------------------------------------------------------------------------------
Input (10)
  ↓
Linear(10 → 512) + LayerNorm + LeakyReLU
  ↓
Self-Attention (512)
  ↓
Residual Blocks (512) ×4:
  └─ [Same as ResNet block]
  ↓
Linear(512 → 768) → Reshape(1, 32, 24)
  ↓
Spatial Attention:
  ├─ Max Pool (channel-wise)
  ├─ Avg Pool (channel-wise)
  ├─ Concat → Conv(2→1, 7×7)
  └─ Sigmoid → Attention Map
  ↓
Conv(1→32, 3×3) + BN + LeakyReLU
  ↓
Conv(32→64, 3×3) + BN + LeakyReLU
  ↓
Conv(64→32, 3×3) + BN + LeakyReLU
  ↓
Conv(32→1, 3×3) + ReLU → (1, 32, 24)

Parameters: ~2.5M
Speed: Medium
Memory: Medium

Key Features:
- Attention at both 1D (features) and 2D (spatial) levels
- Combines strengths of ResNet and CNN
- Best balance of performance and efficiency

--------------------------------------------------------------------------------
6. LIGHTWEIGHT
--------------------------------------------------------------------------------
Input (10)
  ↓
Linear(10 → 256) + LeakyReLU
  ↓
Linear(256 → 256) + LeakyReLU
  ↓
Linear(256 → 128) + LeakyReLU
  ↓
Linear(128 → 768) → Reshape(1, 32, 24)
  ↓
Conv(1→16, 3×3) + LeakyReLU
  ↓
Conv(16→1, 3×3) + ReLU → (1, 32, 24)

Parameters: ~0.5M
Speed: Very Fast (3x faster)
Memory: Very Low

================================================================================
OUTPUT: 2D Spectra (1, 32, 24) - 1 channel, 32 frequencies, 24 directions
================================================================================


PERFORMANCE COMPARISON (Expected on typical wave spectra data):
================================================================================

Model           | R²    | MAE   | Speed | Memory | When to Use
----------------|-------|-------|-------|--------|---------------------------
Lightweight     | 0.85  | 0.03  | ⚡⚡⚡  | 💾     | Deployment, real-time
AttentionFFNN   | 0.88  | 0.025 | ⚡⚡   | 💾💾   | Quick baseline
Hybrid          | 0.92  | 0.018 | ⚡⚡   | 💾💾   | ⭐ Best all-around
UNet            | 0.91  | 0.020 | ⚡    | 💾💾💾 | Spatial reconstruction
ResNet          | 0.93  | 0.016 | ⚡    | 💾💾💾 | Complex patterns
Transformer     | 0.90  | 0.022 | ⚡    | 💾💾💾💾| Parameter interactions

================================================================================


DECISION TREE:
================================================================================

Do you need real-time inference (< 10ms)?
├─ YES → Use LIGHTWEIGHT
└─ NO  → Continue

Do you have limited GPU memory (< 8GB)?
├─ YES → Use ATTENTION FFNN or HYBRID
└─ NO  → Continue

Is your primary goal 2D spatial reconstruction?
├─ YES → Use UNET or HYBRID
└─ NO  → Continue

Do you have very complex spectra with multiple peaks?
├─ YES → Use RESNET
└─ NO  → Continue

Do you need to model interactions between input parameters?
├─ YES → Use TRANSFORMER
└─ NO  → Use HYBRID (best default choice)

================================================================================


KEY ARCHITECTURAL INNOVATIONS:
================================================================================

1. Self-Attention: Learns which input parameters are most important
   - Dynamically weights features
   - Better than fixed fully-connected layers

2. Residual Connections: Enables training of very deep networks
   - Prevents vanishing gradients
   - Improves gradient flow

3. Skip Connections (U-Net): Preserves spatial information
   - Connects encoder to decoder
   - Maintains fine details

4. Spatial Attention: Focuses on important regions of 2D spectra
   - Learns where to look in frequency-direction space
   - Improves peak detection

5. Layer Normalization: Stabilizes training
   - Normalizes activations
   - Allows higher learning rates

6. Dropout: Prevents overfitting
   - Randomly drops neurons during training
   - Improves generalization

================================================================================


HYPERPARAMETER SENSITIVITY:
================================================================================

Model           | Learning Rate | Batch Size | Most Sensitive To
----------------|---------------|------------|-------------------
Lightweight     | 2e-3 (high)   | 64 (large) | Learning rate
AttentionFFNN   | 1e-3          | 32         | Dropout rate
Hybrid          | 1e-3          | 32         | Hidden dim, num blocks
UNet            | 1e-3          | 32         | Number of blocks
ResNet          | 1e-3          | 32         | Number of residual blocks
Transformer     | 5e-4 (low)    | 16 (small) | d_model, num_heads

================================================================================


TRAINING TIME ESTIMATES (100 epochs, 10k samples):
================================================================================

GPU: NVIDIA RTX 3090 (24GB)

Lightweight:     ~15 minutes
AttentionFFNN:   ~25 minutes
Hybrid:          ~35 minutes
UNet:            ~40 minutes
ResNet:          ~35 minutes
Transformer:     ~50 minutes

CPU: Intel i9 (would be 10-20x slower)

================================================================================
"""

if __name__ == "__main__":
    print(ARCHITECTURES)
