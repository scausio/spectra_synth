"""
Enhanced configuration with sparse spectral data improvements

This extends your existing config_experiments.py with new loss functions
and optimizations for sparse data, while maintaining full backward compatibility.
"""
import os

# ============================================================================
# NEW: Sparse Data Optimized Experiments
# ============================================================================

SPARSE_DATA_EXPERIMENTS = {
    'unet_sparse_optimized': {
        'model_name': 'unet',
        'model_params': {
            'hidden_dim': 512,        # Increased from 256
            'channels': [64, 128, 256, 512],  # Deeper hierarchy
            'num_blocks': 4,          # Increased from 3
            'dropout': 0.1
        },
        'learning_rate': 5e-4,        # Lower for stability
        'batch_size': 32,
        'epochs': 300,                # Longer training
        'loss_function': 'msle_constraint_weighted',  # NEW
        'loss_params': {              # NEW
            'alpha': 0.15,
            'beta': 0.15,
            'gamma': 0.02,
            'nonzero_weight': 20.0,
            'zero_weight': 1.0
        },
        'description': 'U-Net optimized for sparse spectral data'
    },

    'hybrid_sparse_optimized': {
        'model_name': 'hybrid',
        'model_params': {
            'hidden_dim': 512,
            'num_res_blocks': 6,      # Increased from 4
            'dropout': 0.1
        },
        'learning_rate': 5e-4,
        'batch_size': 32,
        'epochs': 300,
        'loss_function': 'msle_constraint_weighted',
        'loss_params': {
            'alpha': 0.15,
            'beta': 0.15,
            'gamma': 0.02,
            'nonzero_weight': 20.0,
            'zero_weight': 1.0
        },
        'description': 'Hybrid CNN-Attention optimized for sparse data'
    },

    'resnet_sparse_optimized': {
        'model_name': 'resnet',
        'model_params': {
            'hidden_dim': 512,
            'num_res_blocks': 12,
            'num_conv_blocks': 4,
            'dropout': 0.1
        },
        'learning_rate': 5e-4,
        'batch_size': 32,
        'epochs': 300,
        'loss_function': 'msle_constraint_weighted',
        'loss_params': {
            'alpha': 0.15,
            'beta': 0.15,
            'gamma': 0.02,
            'nonzero_weight': 20.0,
            'zero_weight': 1.0
        },
        'description': 'ResNet optimized for sparse data'
    },

    # Alternative loss functions for experimentation
    'unet_log_scale': {
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
        'loss_function': 'log_scale',  # Pure log-scale loss
        'loss_params': {
            'epsilon': 1e-8
        },
        'description': 'U-Net with pure log-scale loss (simpler alternative)'
    },

    'unet_weighted_mse': {
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
        'loss_function': 'weighted_mse',  # Weighted MSE
        'loss_params': {
            'nonzero_weight': 30.0,
            'zero_weight': 1.0
        },
        'description': 'U-Net with weighted MSE (aggressive weighting)'
    },
}

# ============================================================================
# ORIGINAL EXPERIMENTS (from your config_experiments.py)
# ============================================================================

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
        'batch_size': 32,
        'epochs': 100,
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
        'batch_size': 32,
        'epochs': 100,
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

    'autoencoder': {
        'model_name': 'autoencoder',
        'batch_size': 32,
        'epochs': 120,
        'description': 'VAE'
    }
}

# Merge sparse data experiments into main experiments dict
EXPERIMENTS.update(SPARSE_DATA_EXPERIMENTS)

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    # Training parameters
    'outdir': 'output_improved',
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'checkpoint_interval': 1,
    'early_stopping_patience': 100,
    'scheduler_patience': 5,  # NEW: For ReduceLROnPlateau
    'init_epoch': 0,
    'auto_resume': True,
    'num_workers': 4,  # NEW: For DataLoader

    # Data parameters
    'wind': False,
    'add_coords': False,
    'decimate_input': 10,
    'fname': '*2025*.zarr',  # NEW: File pattern for zarr files

    # Model selection
    'model_name': 'hybrid',
    'model_params': {
        'hidden_dim': 512,
        'num_res_blocks': 4,
    },

    # Loss function (backward compatible)
    'loss_function': 'combined',
    
    # Legacy loss parameters (for backward compatibility)
    'alpha': 0.0001,
    'beta': 0.0001,
    'gamma': 0,
    'alpha_intensity': 1.0,
    'alpha_position': 0.3,
    'alpha_moment': 0.1,
    'alpha_peak': 0.1,
    
    # NEW: Modern loss parameters (used by new loss functions)
    # Only used when loss_function is one of the new types
    'loss_params': {
        'alpha': 0.15,
        'beta': 0.15,
        'gamma': 0.02,
        'nonzero_weight': 20.0,
        'zero_weight': 1.0,
        'epsilon': 1e-8
    },

    # Data paths
    'local_machine': False,
    'base': '/work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026',
    'stats_dir': '/work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/stats_grid',
    'spc_dir': '/work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/spcs_grid',
    'scaler': '/work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/scalers/scalers_partitions.json',
    'depth': '/work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/stats_grid/dpt.nc'
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
    
    # Deep copy model_params and loss_params if they exist
    if 'model_params' in experiment:
        config['model_params'] = experiment['model_params'].copy()
    
    if 'loss_params' in experiment:
        config['loss_params'] = experiment['loss_params'].copy()
    
    # Update other experiment settings
    for key, value in experiment.items():
        if key not in ['model_params', 'loss_params']:
            config[key] = value
    
    # Update output directory to include experiment name
    config['outdir'] = f"output_{experiment_name}"
    
    return config


def get_all_experiments():
    """Get list of all available experiment names"""
    return list(EXPERIMENTS.keys())


def get_sparse_experiments():
    """Get list of sparse data optimized experiments"""
    return list(SPARSE_DATA_EXPERIMENTS.keys())


def print_experiment_info(sparse_only=False):
    """
    Print information about all available experiments
    
    Args:
        sparse_only: If True, only print sparse data optimized experiments
    """
    experiments_to_show = SPARSE_DATA_EXPERIMENTS if sparse_only else EXPERIMENTS
    
    print("\n" + "="*80)
    if sparse_only:
        print("Sparse Data Optimized Experiments (NEW)")
    else:
        print("Available Experiments")
    print("="*80)
    
    for name, exp in experiments_to_show.items():
        print(f"\n{name}:")
        print(f"  Model: {exp['model_name']}")
        print(f"  Description: {exp['description']}")
        print(f"  Learning Rate: {exp.get('learning_rate', 'default')}")
        print(f"  Batch Size: {exp.get('batch_size', 'default')}")
        print(f"  Epochs: {exp.get('epochs', 'default')}")
        
        if 'loss_function' in exp:
            print(f"  Loss Function: {exp['loss_function']}")
            if 'loss_params' in exp:
                print(f"  Loss Params: {exp['loss_params']}")
        
        if 'model_params' in exp:
            print(f"  Model Params: {exp['model_params']}")
    
    print("\n" + "="*80 + "\n")


def create_quick_test_config(experiment_name, quick_test=True):
    """
    Create a quick test configuration from an experiment
    
    Args:
        experiment_name: Name of the experiment
        quick_test: If True, reduce epochs and use small data subset
    
    Returns:
        Configuration for quick testing
    """
    config = get_experiment_config(experiment_name)
    
    if quick_test:
        config['epochs'] = 20
        config['fname'] = '*202501*.zarr'  # Only January 2025
        config['outdir'] = f"output_{experiment_name}_quicktest"
        config['checkpoint_interval'] = 5
        print(f"\n⚡ Quick test mode enabled:")
        print(f"   - Epochs reduced to 20")
        print(f"   - Using January 2025 data only")
        print(f"   - Output: {config['outdir']}")
    
    return config


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--sparse':
        print_experiment_info(sparse_only=True)
    else:
        print_experiment_info(sparse_only=False)
        print("\n💡 Tip: Run with --sparse flag to see only sparse data optimized experiments")
        print("   Example: python config_experiments.py --sparse")
