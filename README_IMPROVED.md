# Improved 2D Wave Spectra Reconstruction

This repository contains enhanced neural network architectures for reconstructing 2D wave spectra from mean parameters of wind waves and swells.

## Overview

**Input**: N-dimensional vector (default 10 parameters):
- Wind waves (ww): Hs, Tm, Dir
- Swell 1 (sw1): Hs, Tm, Dir  
- Swell 2 (sw2): Hs, Tm, Dir
- Depth

**Output**: 2D wave spectra in frequency-direction space (default 32×24 grid)

## New Features

### 🚀 Multiple Model Architectures

Six different architectures are now available:

1. **AttentionFFNN** - Enhanced fully connected network with self-attention mechanism
   - Good baseline with moderate complexity
   - Self-attention for feature importance
   - Residual connections for better gradient flow

2. **SpectralTransformer** - Transformer-based architecture
   - Excellent for capturing long-range dependencies
   - Positional encoding for sequence information
   - Multi-head attention mechanism

3. **SpectralUNet** - U-Net style with encoder-decoder structure
   - Excellent for spatial reconstruction tasks
   - Skip connections preserve fine details
   - Multi-scale feature processing

4. **SpectralResNet** - Deep residual network
   - Best for very deep networks
   - Multiple residual blocks for complex patterns
   - 2D convolutional refinement

5. **HybridCNNAttention** - Combined CNN and attention
   - Balanced approach with good performance
   - Spatial and channel attention
   - Both 1D feature and 2D spatial processing

6. **LightweightSpectralNet** - Fast and efficient model
   - Good for deployment and real-time applications
   - Minimal parameters while maintaining performance
   - Efficient inference

### 📊 Enhanced Training Features

- **Better Logging**: Comprehensive metrics tracking (MSE, MAE, R², peak error, integral error)
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Early Stopping**: Automatic stopping when validation loss plateaus
- **Gradient Clipping**: Prevents exploding gradients
- **Checkpointing**: Regular model saves with best model tracking
- **Visualization**: Automatic plotting of training curves

### 🔧 Easy Configuration

Predefined experiments in `config_experiments.py`:
- `baseline_ffnn` - Baseline with attention mechanism
- `transformer_large` - Large transformer model
- `unet_deep` - Deep U-Net architecture
- `resnet_deep` - Deep ResNet with many residual blocks
- `resnet_wide` - Wide ResNet with large capacity
- `hybrid_balanced` - Balanced CNN-Attention hybrid
- `lightweight_fast` - Fast lightweight model

## Installation

```bash
# Install dependencies
pip install torch numpy scikit-learn matplotlib seaborn pandas scipy xarray tqdm
```

## Quick Start

### 1. Run a Single Experiment

```bash
python run_experiments.py --experiment hybrid_balanced
```

### 2. Run All Experiments

```bash
python run_experiments.py --all
```

### 3. List Available Experiments

```bash
python run_experiments.py --list
```

### 4. Custom Training

```python
from train import main
from config_experiments import get_experiment_config

# Get configuration
config = get_experiment_config('hybrid_balanced')

# Modify if needed
config['epochs'] = 150
config['batch_size'] = 64

# Run training
main()
```

## Model Evaluation

### Evaluate a Single Model

```python
from evaluate_models import evaluate_single_model

data_config = {
    'spc_path': 'path/to/wave_spectra.nc',
    'stats_path': 'path/to/wave_stats.nc',
    'mp': True,
    'wind': False,
    'add_coords': False,
    'decimate_input': 100
}

evaluate_single_model('output_hybrid/best_model_full.pt', data_config)
```

This generates:
- `metrics.json` - All evaluation metrics
- `sample_predictions.png` - Visual comparison of predictions
- `error_distribution.png` - Error analysis plots
- `spectral_statistics.png` - Spectral characteristic plots

### Compare Multiple Models

```python
from evaluate_models import compare_models

model_paths = {
    'hybrid': 'output_hybrid_balanced/best_model_full.pt',
    'transformer': 'output_transformer_large/best_model_full.pt',
    'resnet': 'output_resnet_deep/best_model_full.pt',
}

compare_models(model_paths, data_config)
```

This creates:
- `model_comparison.csv` - Metrics table
- `model_comparison.png` - Visual comparison

## Project Structure

```
.
├── models_improved.py          # New model architectures
├── train_improved.py           # Enhanced training script
├── config_experiments.py       # Experiment configurations
├── run_experiments.py          # Experiment runner
├── evaluate_models.py          # Evaluation and comparison tools
├── loss_functions.py           # Loss functions (original)
├── reader.py                   # Data reader (original)
├── shaper.py                   # Data shaper (original)
├── utils.py                    # Utilities (original)
└── README_IMPROVED.md          # This file
```

## Evaluation Metrics

The improved evaluation includes:

1. **Basic Metrics**
   - MSE, RMSE, MAE
   - MSLE (Mean Squared Logarithmic Error)
   - R² score
   - Correlation coefficient

2. **Physical Metrics**
   - **Peak Error**: Accuracy of maximum energy prediction
   - **Integral Error**: Energy conservation
   - **Direction Error**: Peak direction accuracy (in degrees)
   - **Frequency Error**: Peak frequency accuracy

3. **Visualizations**
   - Sample predictions vs ground truth
   - Error distributions (absolute and relative)
   - Scatter plots (predicted vs true)
   - Frequency-wise and direction-wise error analysis
   - Average spectra comparisons

## Key Improvements Over Original Code

### Code Quality
- ✅ Modular design with base classes
- ✅ Comprehensive documentation
- ✅ Type hints and clear naming
- ✅ Factory pattern for model creation
- ✅ Configuration management

### Training
- ✅ Advanced optimization (AdamW, LR scheduling)
- ✅ Early stopping to prevent overfitting
- ✅ Gradient clipping for stability
- ✅ Better checkpoint management
- ✅ Comprehensive logging

### Evaluation
- ✅ Multiple evaluation metrics
- ✅ Physical constraint validation
- ✅ Rich visualizations
- ✅ Model comparison tools
- ✅ Statistical analysis

### Flexibility
- ✅ Easy to add new models
- ✅ Configurable experiments
- ✅ Flexible input dimensions
- ✅ Flexible output sizes
- ✅ Command-line interface

## Model Selection Guide

| Model | Speed | Memory | Performance | Use Case |
|-------|-------|--------|-------------|----------|
| Lightweight | ⚡⚡⚡ | 💾 | ⭐⭐⭐ | Deployment, real-time |
| AttentionFFNN | ⚡⚡ | 💾💾 | ⭐⭐⭐⭐ | Good baseline |
| Hybrid | ⚡⚡ | 💾💾 | ⭐⭐⭐⭐⭐ | Best balanced choice |
| UNet | ⚡ | 💾💾💾 | ⭐⭐⭐⭐⭐ | Spatial reconstruction |
| ResNet | ⚡ | 💾💾💾 | ⭐⭐⭐⭐⭐ | Complex patterns |
| Transformer | ⚡ | 💾💾💾💾 | ⭐⭐⭐⭐⭐ | Long-range dependencies |

## Tips for Best Results

1. **Start with Hybrid or AttentionFFNN** - Good balance of performance and speed
2. **Use Early Stopping** - Prevents overfitting (already enabled by default)
3. **Tune Learning Rate** - Start with 1e-3, reduce if training is unstable
4. **Monitor Multiple Metrics** - Don't just look at loss, check R² and physical metrics
5. **Compare Models** - Run several architectures and pick the best for your data
6. **Adjust Model Capacity** - Increase hidden_dim or num_blocks if underfitting

## Advanced Customization

### Create Your Own Model

```python
from models_improved import BaseSpectralModel

class MyCustomModel(BaseSpectralModel):
    def __init__(self, X_train_scaled, reshape_size, scalerX, scalerY):
        input_dim = X_train_scaled.shape[1]
        super().__init__(input_dim, reshape_size, scalerX, scalerY)
        
        # Define your architecture
        self.network = nn.Sequential(
            # ... your layers
        )
    
    def forward(self, x):
        # Implement forward pass
        return self.network(x)
```

### Add Custom Loss Function

Edit `train_improved.py` to add your custom loss:

```python
class MyCustomLoss(nn.Module):
    def forward(self, y_pred, y_true):
        # Your loss computation
        return loss

# Use it in config
config['loss_function'] = 'custom'
criterion = MyCustomLoss()
```

## Citation

If you use this code, please cite:
```
[Your citation information here]
```

## License

[Your license information]

## Contact

[Your contact information]

---

**Note**: This is an improved version of the original wave spectra reconstruction code with enhanced architecture options, better training procedures, and comprehensive evaluation tools.
