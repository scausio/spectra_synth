"""
Quick Reference Guide for Wave Spectra Reconstruction Models

This file provides quick examples and tips for using the improved models.
"""

# ============================================================================
# QUICK START EXAMPLES
# ============================================================================

# -----------------------------------------------------------------------------
# Example 1: Train a single model with default settings
# -----------------------------------------------------------------------------

from train import main
from config_experiments import get_experiment_config

# Get a predefined configuration
config = get_experiment_config('hybrid_balanced')

# Modify paths if needed
config['stats_path'] = 'path/to/your/wave_stats.nc'
config['spc_path'] = 'path/to/your/wave_spectra.nc'

# Run training
main()


# -----------------------------------------------------------------------------
# Example 2: Create and train a custom model
# -----------------------------------------------------------------------------

from models_improved import get_model
from reader import Reader_NC
from shaper import Ds_Conv
import torch.optim as optim

# Load data
ds = Reader_NC(spc_path, stats_path, "FFNN0", decimate_input=100, mp=True)
ml_ds = Ds_Conv(ds, reshape_size=(32, 24), mp=True)

# Create model
model = get_model(
    'resnet',  # Choose: 'attention_ffnn', 'transformer', 'unet', 'resnet', 'hybrid', 'lightweight'
    ml_ds.X_train_scaled,
    (32, 24),
    ml_ds.scaler_X,
    ml_ds.scaler_Y,
    hidden_dim=512,
    num_res_blocks=8
)

# Train
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
# ... training loop


# -----------------------------------------------------------------------------
# Example 3: Evaluate a trained model
# -----------------------------------------------------------------------------

from evaluate_models import evaluate_single_model

data_config = {
    'spc_path': 'path/to/wave_spectra.nc',
    'stats_path': 'path/to/wave_stats.nc',
    'mp': True,
    'wind': False,
    'add_coords': False,
    'decimate_input': 100
}

# Comprehensive evaluation with plots
evaluate_single_model('output_hybrid/best_model_full.pt', data_config, 'results/')


# -----------------------------------------------------------------------------
# Example 4: Compare multiple models
# -----------------------------------------------------------------------------

from evaluate_models import compare_models

model_paths = {
    'Hybrid': 'output_hybrid_balanced/best_model_full.pt',
    'Transformer': 'output_transformer_large/best_model_full.pt',
    'ResNet': 'output_resnet_deep/best_model_full.pt',
    'Lightweight': 'output_lightweight_fast/best_model_full.pt',
}

# Creates comparison table and plots
comparison_df = compare_models(model_paths, data_config, 'comparison/')


# -----------------------------------------------------------------------------
# Example 5: Use a trained model for inference
# -----------------------------------------------------------------------------

import torch
import joblib

# Load model
model = torch.load('output_hybrid/best_model_full.pt')
model.eval()

# Load scalers
scalers = joblib.load('output_hybrid/scaler.pkl')
scaler_X = scalers['X']
scaler_Y = scalers['Y']

# Prepare input (example with 9 parameters + depth)
input_params = np.array([[
    1.5,   # Hs_ww
    5.0,   # Tm_ww
    180.0, # Dir_ww
    0.8,   # Hs_sw1
    8.0,   # Tm_sw1
    270.0, # Dir_sw1
    0.3,   # Hs_sw2
    10.0,  # Tm_sw2
    90.0,  # Dir_sw2
    # Add more if using wind or coords
]])

# Scale input
input_scaled = scaler_X.transform(input_params)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# Predict
with torch.no_grad():
    output_scaled = model(input_tensor)
    output_numpy = output_scaled.cpu().numpy()

# Inverse transform if needed
spectra_2d = output_numpy.reshape(32, 24)  # or use scaler_Y.inverse_transform


# ============================================================================
# MODEL HYPERPARAMETER SUGGESTIONS
# ============================================================================

HYPERPARAMETERS = {
    'attention_ffnn': {
        'hidden_dims': [512, 512, 256, 256],  # Can try [1024, 512, 256]
        'dropout': 0.1,                        # Range: 0.0-0.3
        'learning_rate': 1e-3,
        'batch_size': 32,
        'note': 'Good general-purpose baseline'
    },
    
    'transformer': {
        'd_model': 256,           # Can try 128, 512
        'nhead': 8,               # Must divide d_model evenly
        'num_layers': 4,          # Range: 2-6
        'dropout': 0.1,
        'learning_rate': 5e-4,    # Lower LR for transformers
        'batch_size': 16,         # Smaller batch for memory
        'note': 'Best for capturing parameter interactions'
    },
    
    'unet': {
        'hidden_dim': 256,        # Can try 128, 512
        'num_blocks': 3,          # Range: 2-4 (limited by spatial dims)
        'learning_rate': 1e-3,
        'batch_size': 32,
        'note': 'Excellent for 2D spatial reconstruction'
    },
    
    'resnet': {
        'hidden_dim': 512,        # Can try 256, 1024
        'num_res_blocks': 8,      # Range: 4-16
        'num_conv_blocks': 4,     # Range: 2-6
        'learning_rate': 1e-3,
        'batch_size': 32,
        'note': 'Can go very deep, good for complex patterns'
    },
    
    'hybrid': {
        'hidden_dim': 512,        # Can try 256, 1024
        'num_res_blocks': 4,      # Range: 2-8
        'learning_rate': 1e-3,
        'batch_size': 32,
        'note': 'Best all-around choice'
    },
    
    'lightweight': {
        'hidden_dim': 256,        # Keep small: 128-512
        'learning_rate': 2e-3,    # Can use higher LR
        'batch_size': 64,         # Can use larger batches
        'note': 'Fast inference, good for deployment'
    }
}


# ============================================================================
# TROUBLESHOOTING TIPS
# ============================================================================

TROUBLESHOOTING = {
    'Training loss not decreasing': [
        'Reduce learning rate (try 1e-4 or 5e-4)',
        'Check data normalization',
        'Try simpler model first',
        'Check for NaN values in data',
    ],
    
    'Validation loss increasing': [
        'Reduce model capacity (smaller hidden_dim)',
        'Increase dropout',
        'Use early stopping (already enabled)',
        'Add weight decay (increase from 1e-5 to 1e-4)',
    ],
    
    'Poor spectral reconstruction': [
        'Try U-Net or Hybrid model (better for 2D)',
        'Check if input parameters are informative',
        'Try constrained loss function (MSLELossContraint)',
        'Increase model capacity',
    ],
    
    'Training too slow': [
        'Use Lightweight model',
        'Reduce batch size if GPU memory limited',
        'Reduce model complexity',
        'Use fewer residual blocks',
    ],
    
    'Out of memory': [
        'Reduce batch_size (try 16 or 8)',
        'Use Lightweight model',
        'Reduce hidden_dim',
        'Reduce number of layers/blocks',
    ],
}


# ============================================================================
# EXPERIMENT WORKFLOW
# ============================================================================

WORKFLOW = """
1. START WITH BASELINE
   python run_experiments.py --experiment baseline_ffnn
   
2. TRY RECOMMENDED MODELS
   python run_experiments.py --experiment hybrid_balanced
   python run_experiments.py --experiment unet_deep
   
3. EVALUATE AND COMPARE
   from evaluate_models import compare_models
   # Compare the best models
   
4. FINE-TUNE BEST MODEL
   - Adjust hyperparameters in config_experiments.py
   - Try different loss functions
   - Experiment with learning rate
   
5. FINAL EVALUATION
   - Use evaluate_single_model for comprehensive analysis
   - Check all metrics: R², MAE, peak error, integral error
   - Visualize predictions
   
6. DEPLOY
   - Use best_model_full.pt for inference
   - Consider Lightweight model if speed is critical
"""


# ============================================================================
# LOSS FUNCTION GUIDE
# ============================================================================

LOSS_FUNCTIONS = {
    'mse': {
        'use_when': 'Spectra values are roughly same scale',
        'pros': 'Simple, fast, well-understood',
        'cons': 'Can be dominated by large values',
    },
    
    'msle': {
        'use_when': 'Spectra has wide range of values (recommended)',
        'pros': 'Handles different scales well, focuses on relative errors',
        'cons': 'Slightly more complex',
    },
    
    'msle_constraint': {
        'use_when': 'Need to enforce physical constraints',
        'pros': 'Can preserve peak, integral, and peak position',
        'cons': 'More hyperparameters to tune (alpha, beta, gamma)',
        'tune': {
            'alpha': 'Controls integral/energy conservation (try 1e-4)',
            'beta': 'Controls peak value preservation (try 1e-4)',
            'gamma': 'Controls peak position (try 0 or 1e-4)',
        }
    }
}


# ============================================================================
# PERFORMANCE EXPECTATIONS
# ============================================================================

EXPECTED_METRICS = """
Good Model Performance:
- R² > 0.85
- MAE < 0.05 (depends on data scale)
- Peak Error < 0.1 (10%)
- Integral Error < 0.05 (5%)
- Direction Error < 20 degrees

Excellent Model Performance:
- R² > 0.92
- MAE < 0.02
- Peak Error < 0.05 (5%)
- Integral Error < 0.02 (2%)
- Direction Error < 10 degrees
"""


if __name__ == "__main__":
    print("Quick Reference Guide")
    print("=" * 80)
    print("\nThis file contains examples and tips for using the models.")
    print("Import specific functions or refer to code snippets above.")
    print("\nRecommended reading order:")
    print("1. QUICK START EXAMPLES")
    print("2. MODEL HYPERPARAMETER SUGGESTIONS")
    print("3. EXPERIMENT WORKFLOW")
    print("4. TROUBLESHOOTING TIPS")
    print("\n" + "=" * 80)
