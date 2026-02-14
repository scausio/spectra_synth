"""
Configuration file for model training experiments

This file allows easy switching between different model configurations
and hyperparameters for experimentation.
"""

# Available models:
# - 'attention_ffnn': Enhanced FFNN with self-attention
# - 'transformer': Transformer-based architecture
# - 'unet': U-Net style with skip connections
# - 'resnet': Deep residual network
# - 'hybrid': Hybrid CNN-Attention model
# - 'lightweight': Fast and efficient model
import os
import xarray as xr
EXPERIMENTS = {
    'baseline_ffnn': {
        'model_name': 'attention_ffnn',
        'model_params': {
            'hidden_dims': [512, 512, 256, 256],
            'dropout': 0.1
        },
        'learning_rate': 1e-3,
        'batch_size': 4,
        'epochs': 100,
        'description': 'Baseline with attention mechanism'
    },

    'transformer_large': {
        'model_name': 'transformer',
        'model_params': {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 4,
            'dropout': 0.1
        },
        'learning_rate': 5e-4,
        'batch_size': 16,
        'epochs': 150,
        'description': 'Transformer for long-range dependencies'
    },

    'unet_deep': {
        'model_name': 'unet',
        'model_params': {
            'hidden_dim': 256,
            'num_blocks': 3
        },
        'learning_rate': 1e-3,
        'batch_size': 8,
        'epochs': 300,
        'description': 'U-Net for spatial reconstruction'
    },

    'resnet_deep': {
        'model_name': 'resnet',
        'model_params': {
            'hidden_dim': 512,
            'num_res_blocks': 12,
            'num_conv_blocks': 4
        },
        'learning_rate': 1e-3,
        'batch_size': 32,
        'epochs': 150,
        'description': 'Deep ResNet with many residual blocks'
    },

    'hybrid_balanced': {
        'model_name': 'hybrid',
        'model_params': {
            'hidden_dim': 512,
            'num_res_blocks': 4
        },
        'learning_rate': 1e-3,
        'description': 'Balanced CNN-Attention hybrid'
    },

    'lightweight_fast': {
        'model_name': 'lightweight',
        'model_params': {
            'hidden_dim': 256
        },
        'learning_rate': 2e-3,
        'batch_size': 64,
        'epochs': 80,
        'description': 'Fast lightweight model for deployment'
    },

    'resnet_wide': {
        'model_name': 'resnet',
        'model_params': {
            'hidden_dim': 1024,
            'num_res_blocks': 6,
            'num_conv_blocks': 5
        },
        'learning_rate': 5e-4,
        'batch_size': 16,
        'epochs': 120,
        'description': 'Wide ResNet with large capacity'
    },
    "autoencoder":{
        'model_name': 'autoencoder',
        'batch_size': 32,
        'epochs': 120,
        'description': 'VAE'
    },
    'unet_sparse_optimized': {
        'model_name': 'unet',
        'model_params': {
            'hidden_dim': 512,
            'channels': [64, 128, 256, 512],
            'num_blocks': 4,
            'dropout': 0.1
        },
        'learning_rate': 5e-4,
        'batch_size': 32,
        'epochs': 300,
        'loss_function': 'msle_constraint_weighted',
        'alpha': 0.15,
        'beta': 0.15,
        'gamma': 0.02,
        'nonzero_weight': 20.0,
        'zero_weight': 1.0,
        'description': 'U-Net optimized for sparse spectral data'
    },

    'unet_log_scale': {
        'model_name': 'unet',
        'model_params': {
            'hidden_dim': 512,
            'channels': [64, 128, 256, 512],
            'num_blocks': 4,
            'dropout': 0.1
        },
        'learning_rate': 8e-4,
        'batch_size': 32,
        'epochs': 300,
        'loss_function': 'log_scale',
        'description': 'U-Net with pure log-scale loss',
        'early_stopping_patience': 100,
        'decimate_input': 10,
        'auto_resume': True,
    },
    'unet_msle': {
        'model_name': 'unet',
        'model_params': {
            'hidden_dim': 512,
            'channels': [64, 128, 256, 512],
            'num_blocks': 4,
            'dropout': 0.1
        },
        'learning_rate': 8e-4,
        'batch_size': 8,
        'epochs': 300,
        'loss_function': 'msle',
        'description': 'U-Net with pure log-scale loss',
        'early_stopping_patience': 1000,
        'decimate_input': 10,
        'auto_resume': False,
    }

}

# Default configuration (used when no experiment is specified)
DEFAULT_CONFIG = {
    # Training parameters
    'outdir': 'output_unet',
    'batch_size': 8,
    'epochs': 300,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'checkpoint_interval': 5,
    'init_epoch': 0,
    'auto_resume': True,

    # Data parameters
    'wind': False,  # Include wind components
    'add_coords': False,  # Include lat/lon coordinates
    'decimate_input': 10,  # Decimation factor for input data

    # Model selection
    'model_name': 'hybrid',
    'model_params': {
        'hidden_dim': 512,
        'num_res_blocks': 4,
    },

    # Loss function
    'loss_function': 'combined',  # Options: 'mse', 'msle', 'msle_constraint', 'combined'
    'alpha': 0.0001,  # For constrained loss
    'beta': 0.0001,   # For constrained loss
    'gamma': 0,       # For constrained loss
    'alpha_intensity': 1.0,# For combined loss
    "alpha_position":  0.3,# For combined loss
    "alpha_moment":  0.1,# For combined loss
    "alpha_peak": 0.1,# For combined loss
    # Data paths
    'local_machine': False,
    'base':'/work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026',
    'stats_path': '/work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/stats_grid',
    'spc_path': '/work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/spcs_grid',
    'scaler': '/work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/scalers/scalers_partitions.json',
    'depth':'/work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/stats_grid/dpt.nc'

    }


def get_experiment_config(experiment_name):
    """
    Get configuration for a specific experiment.

    Args:
        experiment_name: Name of the experiment from EXPERIMENTS dict

    Returns:
        Complete configuration dictionary
    """
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}. "
                        f"Available: {list(EXPERIMENTS.keys())}")

    # Start with default config
    config = DEFAULT_CONFIG.copy()

    # Update with experiment-specific settings
    experiment = EXPERIMENTS[experiment_name]
    config.update(experiment)

    # Update output directory to include experiment name
    config['outdir'] = f"output_{experiment_name}"

    return config


def get_all_experiments():
    """Get list of all available experiment names"""
    return list(EXPERIMENTS.keys())


def print_experiment_info():
    """Print information about all available experiments"""
    print("\n" + "="*80)
    print("Available Experiments")
    print("="*80)

    for name, exp in EXPERIMENTS.items():
        print(f"\n{name}:")
        print(f"  Model: {exp['model_name']}")
        print(f"  Description: {exp['description']}")
        print(f"  Learning Rate: {exp['learning_rate']}")
        print(f"  Batch Size: {exp['batch_size']}")
        print(f"  Epochs: {exp['epochs']}")
        if 'model_params' in exp:
            print(f"  Model Params: {exp['model_params']}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print_experiment_info()
