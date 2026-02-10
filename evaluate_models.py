"""
Model Evaluation and Comparison Script

This script helps evaluate and compare different trained models, providing
comprehensive metrics and visualizations.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reader import Reader
from shaper import Ds_Conv
from loss_functions import MSLELoss
import torch.nn as nn
import json
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model_path, data_config):
        """
        Initialize evaluator with model and data
        
        Args:
            model_path: Path to saved model
            data_config: Dictionary with data configuration
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Load data
        self._load_data(data_config)
    
    def _load_data(self, config):
        """Load and prepare test data"""
        ds = Reader(
            config['spc_path'],
            config['stats_path'],
            "FFNN0",
            config.get('decimate_input', 1),
            wind=config.get('wind', False),
            mp=config.get('mp', True)
        )
        
        ml_ds = Ds_Conv(
            ds,
            reshape_size=(ds.kappa_bins, ds.theta_bins),
            wind=config.get('wind', False),
            coords=config.get('add_coords', False),
            mp=config.get('mp', True)
        )
        
        self.X_test = ml_ds.X_test_scaled.to(self.device)
        self.Y_test = ml_ds.Y_test.to(self.device)
        self.scaler_X = ml_ds.scaler_X
        self.scaler_Y = ml_ds.scaler_Y
        self.freqs = ds.freqs
        self.dires = ds.dires
    
    def compute_metrics(self):
        """Compute comprehensive evaluation metrics"""
        with torch.no_grad():
            Y_pred = self.model(self.X_test)
        
        # Convert to numpy
        y_true = self.Y_test.cpu().numpy()
        y_pred = Y_pred.cpu().numpy()
        
        metrics = {}
        
        # Overall metrics
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        
        # Log scale metrics
        y_true_log = np.log1p(np.maximum(y_true, 1e-7))
        y_pred_log = np.log1p(np.maximum(y_pred, 1e-7))
        metrics['msle'] = np.mean((y_true_log - y_pred_log) ** 2)
        
        # R-squared
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Correlation
        metrics['correlation'], _ = pearsonr(y_true_flat, y_pred_flat)
        
        # Peak preservation
        y_true_max = np.max(y_true, axis=(2, 3)).mean()
        y_pred_max = np.max(y_pred, axis=(2, 3)).mean()
        metrics['peak_error'] = abs(y_true_max - y_pred_max) / y_true_max
        metrics['peak_relative_error'] = (y_pred_max - y_true_max) / y_true_max
        
        # Integral/Energy preservation
        y_true_sum = np.sum(y_true, axis=(2, 3)).mean()
        y_pred_sum = np.sum(y_pred, axis=(2, 3)).mean()
        metrics['integral_error'] = abs(y_true_sum - y_pred_sum) / y_true_sum
        metrics['integral_relative_error'] = (y_pred_sum - y_true_sum) / y_true_sum
        
        # Directional accuracy (peak direction)
        y_true_argmax_dir = np.argmax(np.sum(y_true, axis=2), axis=2).flatten()
        y_pred_argmax_dir = np.argmax(np.sum(y_pred, axis=2), axis=2).flatten()
        dir_diff = np.abs(y_true_argmax_dir - y_pred_argmax_dir)
        # Handle circular difference
        dir_diff = np.minimum(dir_diff, len(self.dires) - dir_diff)
        metrics['direction_error_bins'] = dir_diff.mean()
        metrics['direction_error_degrees'] = dir_diff.mean() * (360 / len(self.dires))
        
        # Frequency accuracy (peak frequency)
        y_true_argmax_freq = np.argmax(np.sum(y_true, axis=3), axis=2).flatten()
        y_pred_argmax_freq = np.argmax(np.sum(y_pred, axis=3), axis=2).flatten()
        metrics['frequency_error_bins'] = np.abs(y_true_argmax_freq - y_pred_argmax_freq).mean()
        
        return metrics, Y_pred
    
    def plot_sample_predictions(self, num_samples=6, save_path=None):
        """Plot sample predictions vs ground truth"""
        with torch.no_grad():
            Y_pred = self.model(self.X_test)
        
        y_true = self.Y_test.cpu().numpy()
        y_pred = Y_pred.cpu().numpy()
        
        # Select random samples
        indices = np.random.choice(len(y_true), num_samples, replace=False)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            # Ground truth
            im0 = axes[i, 0].imshow(y_true[idx, 0], aspect='auto', cmap='viridis', origin='lower')
            axes[i, 0].set_title(f'Ground Truth (Sample {idx})')
            axes[i, 0].set_xlabel('Direction')
            axes[i, 0].set_ylabel('Frequency')
            plt.colorbar(im0, ax=axes[i, 0])
            
            # Prediction
            im1 = axes[i, 1].imshow(y_pred[idx, 0], aspect='auto', cmap='viridis', origin='lower')
            axes[i, 1].set_title(f'Prediction (Sample {idx})')
            axes[i, 1].set_xlabel('Direction')
            axes[i, 1].set_ylabel('Frequency')
            plt.colorbar(im1, ax=axes[i, 1])
            
            # Difference
            diff = y_pred[idx, 0] - y_true[idx, 0]
            im2 = axes[i, 2].imshow(diff, aspect='auto', cmap='RdBu_r', origin='lower',
                                   vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
            axes[i, 2].set_title(f'Difference (Sample {idx})')
            axes[i, 2].set_xlabel('Direction')
            axes[i, 2].set_ylabel('Frequency')
            plt.colorbar(im2, ax=axes[i, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_error_distribution(self, save_path=None):
        """Plot distribution of prediction errors"""
        with torch.no_grad():
            Y_pred = self.model(self.X_test)
        
        y_true = self.Y_test.cpu().numpy()
        y_pred = Y_pred.cpu().numpy()
        
        errors = (y_pred - y_true).flatten()
        relative_errors = ((y_pred - y_true) / (y_true + 1e-7)).flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Absolute error histogram
        axes[0, 0].hist(errors, bins=100, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Absolute Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Absolute Errors')
        axes[0, 0].axvline(0, color='red', linestyle='--', label='Zero Error')
        axes[0, 0].legend()
        
        # Relative error histogram
        axes[0, 1].hist(relative_errors[np.abs(relative_errors) < 5], bins=100, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Relative Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Relative Errors (clipped at ±5)')
        axes[0, 1].axvline(0, color='red', linestyle='--', label='Zero Error')
        axes[0, 1].legend()
        
        # Scatter plot
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        axes[1, 0].hexbin(y_true_flat, y_pred_flat, gridsize=50, cmap='Blues', mincnt=1)
        axes[1, 0].plot([y_true_flat.min(), y_true_flat.max()],
                       [y_true_flat.min(), y_true_flat.max()],
                       'r--', label='Perfect Prediction')
        axes[1, 0].set_xlabel('True Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].set_title('Prediction vs Truth')
        axes[1, 0].legend()
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Errors')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_spectral_statistics(self, save_path=None):
        """Plot statistics of spectral reconstruction"""
        with torch.no_grad():
            Y_pred = self.model(self.X_test)
        
        y_true = self.Y_test.cpu().numpy()
        y_pred = Y_pred.cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Frequency-wise MAE
        freq_mae = np.mean(np.abs(y_pred - y_true), axis=(0, 1, 3))
        axes[0, 0].plot(self.freqs, freq_mae, marker='o')
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].set_title('MAE vs Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Direction-wise MAE
        dir_mae = np.mean(np.abs(y_pred - y_true), axis=(0, 1, 2))
        axes[0, 1].plot(self.dires, dir_mae, marker='o')
        axes[0, 1].set_xlabel('Direction (degrees)')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('MAE vs Direction')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average spectra comparison (frequency)
        avg_true_freq = np.mean(np.sum(y_true, axis=3), axis=(0, 1))
        avg_pred_freq = np.mean(np.sum(y_pred, axis=3), axis=(0, 1))
        axes[1, 0].plot(self.freqs, avg_true_freq, label='True', marker='o')
        axes[1, 0].plot(self.freqs, avg_pred_freq, label='Predicted', marker='x')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Average Energy')
        axes[1, 0].set_title('Average Frequency Spectrum')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Average spectra comparison (direction)
        avg_true_dir = np.mean(np.sum(y_true, axis=2), axis=(0, 1))
        avg_pred_dir = np.mean(np.sum(y_pred, axis=2), axis=(0, 1))
        axes[1, 1].plot(self.dires, avg_true_dir, label='True', marker='o')
        axes[1, 1].plot(self.dires, avg_pred_dir, label='Predicted', marker='x')
        axes[1, 1].set_xlabel('Direction (degrees)')
        axes[1, 1].set_ylabel('Average Energy')
        axes[1, 1].set_title('Average Directional Spectrum')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def compare_models(model_paths, data_config, output_dir='comparison_results'):
    """
    Compare multiple trained models
    
    Args:
        model_paths: Dictionary of {model_name: model_path}
        data_config: Data configuration dictionary
        output_dir: Directory to save comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    print("Evaluating models...")
    for name, path in model_paths.items():
        print(f"\nEvaluating {name}...")
        evaluator = ModelEvaluator(path, data_config)
        metrics, _ = evaluator.compute_metrics()
        results[name] = metrics
        print(f"  R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}")
    
    # Create comparison DataFrame
    df = pd.DataFrame(results).T
    df = df.round(6)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(csv_path)
    print(f"\nComparison saved to {csv_path}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Comparison', fontsize=16)
    
    metrics_to_plot = ['r2', 'rmse', 'mae', 'msle', 'peak_error', 'integral_error']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        values = [results[name][metric] for name in model_paths.keys()]
        ax.bar(range(len(model_paths)), values)
        ax.set_xticks(range(len(model_paths)))
        ax.set_xticklabels(model_paths.keys(), rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return df


def evaluate_single_model(model_path, data_config, output_dir='evaluation_results'):
    """
    Comprehensive evaluation of a single model
    
    Args:
        model_path: Path to the trained model
        data_config: Data configuration dictionary
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model and data...")
    evaluator = ModelEvaluator(model_path, data_config)
    
    print("Computing metrics...")
    metrics, _ = evaluator.compute_metrics()
    
    # Print metrics
    print("\n" + "="*80)
    print("Evaluation Metrics")
    print("="*80)
    for key, value in metrics.items():
        print(f"{key:30s}: {value:.6f}")
    print("="*80 + "\n")
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate plots
    print("Generating visualizations...")
    evaluator.plot_sample_predictions(
        num_samples=6,
        save_path=os.path.join(output_dir, 'sample_predictions.png')
    )
    
    evaluator.plot_error_distribution(
        save_path=os.path.join(output_dir, 'error_distribution.png')
    )
    
    evaluator.plot_spectral_statistics(
        save_path=os.path.join(output_dir, 'spectral_statistics.png')
    )
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    
    # Data configuration
    data_config = {
        'spc_path': '/work/cmcc/ww3_cst-dev/work/ML/data/SSdaily_dataset/wave_spectra_Ita_red2.nc',
        'stats_path': '/work/cmcc/ww3_cst-dev/work/ML/data/SSdaily_dataset/wave_stats_Ita_red2.nc',
        'mp': True,
        'wind': False,
        'add_coords': False,
        'decimate_input': 100
    }
    
    # Evaluate single model
    # model_path = 'output_improved/best_model_full.pt'
    # evaluate_single_model(model_path, data_config)
    
    # Compare multiple models
    model_paths = {
        'hybrid': 'output_hybrid_balanced/best_model_full.pt',
        'transformer': 'output_transformer_large/best_model_full.pt',
        'resnet': 'output_resnet_deep/best_model_full.pt',
        # Add more models as needed
    }
    # compare_models(model_paths, data_config)
    
    print("Use this script by calling:")
    print("  - evaluate_single_model(model_path, data_config)")
    print("  - compare_models(model_paths_dict, data_config)")
