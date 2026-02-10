"""
Inference and Visualization Script for Wave Spectra Reconstruction

This script:
- Loads a trained model
- Performs inference on test data
- Compares predictions with ground truth and Yamaguchi approximation
- Generates comprehensive visualizations
- Computes quantitative metrics

Updated to work with:
- New zarr-based data structure
- New Reader class with build_file_pairs
- New model architectures from models_improved.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from glob import glob
from natsort import natsorted
import argparse
from pathlib import Path
from tqdm import tqdm
import random

from yamaguchi import JONSWAP
from reader import Reader, build_file_pairs, CreateDataset
from utils import fixBCdir
from models_improved import get_model

# Set seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_model_and_data(model_path, data_config):
    """Load trained model and prepare test data"""
    print(f"Loading model from: {model_path}")

    # Load device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Load scaler configuration
    with open(data_config['scaler'], 'r') as f:
        scaler_dict = json.load(f)

    print(f"✓ Scaler configuration loaded")

    # Build file pairs for test data
    print("Loading test data...")
    file_pairs = build_file_pairs( data_config['stats_dir'],data_config['spc_dir'],fname="*202501*.zarr")
    print(f"Found {len(file_pairs)} file pairs")

    # Create dataset
    dataset = CreateDataset(file_pairs, Reader, data_config)
    # Get dimensions from first sample
    X_sample, Y_sample = dataset[0]
    #plt.imshow(Y_sample[0])
    #plt.show()
    #exit()
    input_dim = X_sample.shape[0]
    theta_bins = Y_sample.shape[2]
    k_bins = Y_sample.shape[1]

    print(f"Input dimension: {input_dim}")
    print(f"Output shape: ({theta_bins}, {k_bins})")

    # Initialize model with same architecture used in training
    model = get_model(
        data_config['model_name'],
        input_dim,
        (theta_bins, k_bins),
        **data_config.get('model_params', {})
    )

    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        # If checkpoint is the full model
        model = checkpoint
        print(f"✓ Full model loaded")

    model.to(device)
    model.eval()
    print(f"✓ Model ready on {device}")

    # Get frequency and direction bins from a sample file
    import xarray as xr
    sample_spc_file = file_pairs[0][1]
    ds_spc = xr.open_zarr(sample_spc_file)
    freqs = ds_spc.frequency.values
    dires = ds_spc.direction.values

    print(f"✓ Data loaded: {len(dataset)} test samples")
    print(f"  Frequencies: {len(freqs)} bins")
    print(f"  Directions: {len(dires)} bins")

    return model, dataset, scaler_dict, freqs, dires, k_bins, theta_bins, device


def inverse_scale_spectra(scaled_spectra, scaler_dict):
    """
    Inverse transform scaled spectra back to original units

    Args:
        scaled_spectra: Scaled spectra array (N, 1, theta, k) or (N, theta, k)
        scaler_dict: Dictionary with scale and offset for 'EF'

    Returns:
        unscaled_spectra: Spectra in original units
    """
    scale = scaler_dict['EF']['scale']
    offset = scaler_dict['EF']['offset']

    # Inverse: original = (scaled - offset) / scale
    unscaled = (scaled_spectra - offset) / scale
    print ('EF scaled min ',np.nanmin(scaled_spectra))
    print ('EF scaled max ',np.nanmax(scaled_spectra))
    print ('EF unscaled min ',np.nanmin(unscaled))
    print ('EF unscaled max ',np.nanmax(unscaled))
    return unscaled


def inverse_scale_inputs(scaled_inputs, scaler_dict, feature_names):
    """
    Inverse transform scaled input features

    Args:
        scaled_inputs: Scaled input array (N, n_features)
        scaler_dict: Dictionary with scale and offset for each feature
        feature_names: List of feature names in order

    Returns:
        unscaled_inputs: Inputs in original units
    """
    unscaled = np.zeros_like(scaled_inputs)

    for i, name in enumerate(feature_names):
        if name in scaler_dict:
            scale = scaler_dict[name]['scale']
            offset = scaler_dict[name]['offset']
            unscaled[:, i] = (scaled_inputs[:, i] - offset) / scale
            print (f'{name} scaled min ',np.nanmin(scaled_inputs[:, i]))
            print (f'{name} scaled max ',np.nanmax(scaled_inputs[:, i]))
            print (f'{name} unscaled min ',np.nanmin(unscaled[:, i]))
            print (f'{name} unscaled max ',np.nanmax(unscaled[:, i]))
        else:
            print(f"Warning: {name} not found in scaler_dict, using scaled values")
            unscaled[:, i] = scaled_inputs[:, i]

    return unscaled


def compute_yamaguchi_spectra(X_test, freqs, theta_bins, mp=True):
    """Compute Yamaguchi (JONSWAP) spectra for comparison"""
    print("Computing Yamaguchi approximations...")

    # Validate input values are in physical units
    print(f"  X_test range for Yamaguchi: [{X_test.min():.4f}, {X_test.max():.4f}]")

    yamaguchi_spectra = []
    n_samples = X_test.shape[0]

    for i in range(n_samples):
        if mp:
            # For multi-partition: use only wind wave parameters
            hs = X_test[i, 0]
            tp = X_test[i, 1]
            dir_deg = X_test[i, 2]

            # Debug first sample
            if i == 0:
                print(f"  First sample Hs: {hs:.4f} m, Tp: {tp:.4f} s, Dir: {dir_deg:.4f} deg")
        else:
            # For single partition
            hs = X_test[i, 0]
            tp = X_test[i, 1]
            dir_deg = X_test[i, 3]  # MWD

        # Apply direction fix
        dir_deg = fixBCdir(dir_deg)

        # Compute JONSWAP spectrum
        _, dimSpec = JONSWAP(hs, tp, dir_deg, theta_bins, freqs).main()
        yamaguchi_spectra.append(dimSpec.T[:, ::-1])

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    print("✓ Yamaguchi spectra computed")
    return np.array(yamaguchi_spectra)


def compute_metrics(y_true, y_pred, y_yamaguchi=None):
    """Compute comprehensive metrics"""
    metrics = {}

    # Model metrics
    metrics['model_mse'] = np.mean((y_true - y_pred) ** 2)
    metrics['model_rmse'] = np.sqrt(metrics['model_mse'])
    metrics['model_mae'] = np.mean(np.abs(y_true - y_pred))

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics['model_r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Peak metrics
    y_true_max = np.max(y_true, axis=(1, 2)).mean()
    y_pred_max = np.max(y_pred, axis=(1, 2)).mean()
    metrics['model_peak_error'] = abs(y_true_max - y_pred_max) / y_true_max

    # Integral metrics
    y_true_sum = np.sum(y_true, axis=(1, 2)).mean()
    y_pred_sum = np.sum(y_pred, axis=(1, 2)).mean()
    metrics['model_integral_error'] = abs(y_true_sum - y_pred_sum) / y_true_sum

    # Yamaguchi metrics (if provided)
    if y_yamaguchi is not None:
        metrics['yamaguchi_mse'] = np.mean((y_true - y_yamaguchi) ** 2)
        metrics['yamaguchi_rmse'] = np.sqrt(metrics['yamaguchi_mse'])
        metrics['yamaguchi_mae'] = np.mean(np.abs(y_true - y_yamaguchi))

        ss_res_yam = np.sum((y_true - y_yamaguchi) ** 2)
        metrics['yamaguchi_r2'] = 1 - (ss_res_yam / ss_tot) if ss_tot > 0 else 0

        y_yam_max = np.max(y_yamaguchi, axis=(1, 2)).mean()
        metrics['yamaguchi_peak_error'] = abs(y_true_max - y_yam_max) / y_true_max

        y_yam_sum = np.sum(y_yamaguchi, axis=(1, 2)).mean()
        metrics['yamaguchi_integral_error'] = abs(y_true_sum - y_yam_sum) / y_true_sum

        # Improvement over Yamaguchi
        metrics['improvement_rmse'] = (metrics['yamaguchi_rmse'] - metrics['model_rmse']) / metrics['yamaguchi_rmse'] * 100
        metrics['improvement_mae'] = (metrics['yamaguchi_mae'] - metrics['model_mae']) / metrics['yamaguchi_mae'] * 100
        metrics['improvement_r2'] = (metrics['model_r2'] - metrics['yamaguchi_r2']) * 100

    return metrics


def plot_comparison(idx, y_true, y_pred, y_yamaguchi, X_test, freqs, dires,
                   k_bins, theta_bins, save_path, mp=True):
    """Create comparison plot for a single sample"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    vmin = 0
    vmax = np.nanmax(y_true)

    # Ground truth
    #im1 = axes[0, 0].imshow(y_true, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    im1 = axes[0, 0].imshow(y_true, aspect='auto', cmap='viridis', origin='lower')
    axes[0, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Frequency bins')
    axes[0, 0].set_ylabel('Direction bins')
    plt.colorbar(im1, ax=axes[0, 0], label='E [m²/Hz/deg]')

    # Model prediction
    #im2 = axes[0, 1].imshow(y_pred, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    im2 = axes[0, 1].imshow(y_pred, aspect='auto', cmap='viridis', origin='lower')
    axes[0, 1].set_title('Model Prediction', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Frequency bins')
    axes[0, 1].set_ylabel('Direction bins')
    plt.colorbar(im2, ax=axes[0, 1], label='E [m²/Hz/deg]')

    # Yamaguchi approximation
    if mp:
        hs = X_test[idx, 0]
        tp = X_test[idx, 1]
        dir_deg = X_test[idx, 2]
    else:
        hs = X_test[idx, 0]
        tp = X_test[idx, 1]
        dir_deg = X_test[idx, 3]

    #im3 = axes[0, 2].imshow(y_yamaguchi, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    im3 = axes[0, 2].imshow(y_yamaguchi, aspect='auto', cmap='viridis',  origin='lower')
    axes[0, 2].set_title(f'Yamaguchi (JONSWAP)\nHs={hs:.2f}m, Tp={tp:.2f}s, Dir={dir_deg:.0f}°',
                        fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Frequency bins')
    axes[0, 2].set_ylabel('Direction bins')
    plt.colorbar(im3, ax=axes[0, 2], label='E [m²/Hz/deg]')

    # Error maps
    error_model = y_pred - y_true
    error_yam = y_yamaguchi - y_true
    error_max = max(abs(error_model).max(), abs(error_yam).max())

    im4 = axes[1, 0].imshow(error_model, aspect='auto', cmap='RdBu_r',
                           vmin=-error_max, vmax=error_max, origin='lower')
    axes[1, 0].set_title('Model Error (Pred - Truth)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Frequency bins')
    axes[1, 0].set_ylabel('Direction bins')
    plt.colorbar(im4, ax=axes[1, 0], label='Error')

    im5 = axes[1, 1].imshow(error_yam, aspect='auto', cmap='RdBu_r',
                           vmin=-error_max, vmax=error_max, origin='lower')
    axes[1, 1].set_title('Yamaguchi Error (Pred - Truth)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Frequency bins')
    axes[1, 1].set_ylabel('Direction bins')
    plt.colorbar(im5, ax=axes[1, 1], label='Error')

    # Integrated spectra comparison
    ax = axes[1, 2]

    # Frequency spectrum (integrated over directions)
    freq_true = y_true.sum(axis=0)
    freq_pred = y_pred.sum(axis=0)
    freq_yam = y_yamaguchi.sum(axis=0)

    ax.plot(freqs, freq_true, 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(freqs, freq_pred, 'b--', linewidth=2, label='Model', alpha=0.8)
    ax.plot(freqs, freq_yam, 'r:', linewidth=2, label='Yamaguchi', alpha=0.8)

    ax.set_xlabel('Frequency [Hz]', fontsize=11)
    ax.set_ylabel('Energy', fontsize=11)
    ax.set_title('Frequency Spectrum (integrated over directions)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add metrics text
    mse_model = np.mean((y_true - y_pred) ** 2)
    mse_yam = np.mean((y_true - y_yamaguchi) ** 2)

    metrics_text = f"Sample {idx}\n"
    metrics_text += f"Model MSE: {mse_model:.6f}\n"
    metrics_text += f"Yamaguchi MSE: {mse_yam:.6f}\n"
    metrics_text += f"Improvement: {(mse_yam - mse_model) / mse_yam * 100:.1f}%"

    ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    """Main inference function"""
    print("\n" + "="*80)
    print("WAVE SPECTRA RECONSTRUCTION - INFERENCE AND COMPARISON")
    print("="*80 + "\n")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(args.spc_dir)

    # Configuration for data loading
    data_config = {
        'stats_dir': args.stats_dir,
        'spc_dir': args.spc_dir,
        'depth': args.depth,
        'scaler': args.scaler,
        'decimate_input': args.decimate,
        'wind': args.wind,
        'add_coords': args.coords,
        'model_name': args.model_name,
        'model_params': args.model_params if hasattr(args, 'model_params') else {}
    }

    # Load model and data
    model, dataset, scaler_dict, freqs, dires, k_bins, theta_bins, device = \
        load_model_and_data(args.model_path, data_config)
    print()

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Run inference
    print("Running inference on test data...")
    X_test_scaled = []
    Y_test_scaled = []
    Y_pred_scaled = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc="Processing batches"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)

            X_test_scaled.append(X_batch.cpu().numpy())
            Y_test_scaled.append(y_batch.cpu().numpy())
            Y_pred_scaled.append(y_pred.cpu().numpy())

    # Concatenate all batches
    X_test_scaled = np.concatenate(X_test_scaled, axis=0)  # (N, input_dim)
    Y_test_scaled = np.concatenate(Y_test_scaled, axis=0)  # (N, 1, theta_bins, k_bins)
    Y_pred_scaled = np.concatenate(Y_pred_scaled, axis=0)  # (N, 1, theta_bins, k_bins)
    #Y_test_scaled= np.transpose(Y_test_scaled, (0, 2, 1))
    print(f"✓ Inference complete")
    print(f"  Predictions shape: {Y_pred_scaled.shape}")
    print()

    # Remove channel dimension if present
    if len(Y_test_scaled.shape) == 4:
        Y_test_scaled = Y_test_scaled.squeeze(1)
    if len(Y_pred_scaled.shape) == 4:
        Y_pred_scaled = Y_pred_scaled.squeeze(1)

    # Inverse transform to physical units
    print("Converting to physical units...")

    # Define feature names for input variables
    feature_names = [
        'VHM0_WW', 'VTM01_WW', 'VMDR_WW',
        'VHM0_SW1', 'VTM01_SW1', 'VMDR_SW1',
        'VHM0_SW2', 'VTM01_SW2', 'VMDR_SW2'
    ]

    # Inverse scale inputs
    X_test = inverse_scale_inputs(X_test_scaled, scaler_dict, feature_names)

    # Inverse scale spectra
    Y_test = inverse_scale_spectra(Y_test_scaled, scaler_dict)
    Y_test = np.transpose(Y_test, (0, 2, 1))
    Y_pred = inverse_scale_spectra(Y_pred_scaled, scaler_dict)

    print(f"  X_test range: [{X_test.min():.4f}, {X_test.max():.4f}]")
    print(f"  Y_test range: [{Y_test.min():.4f}, {Y_test.max():.4f}]")
    print(f"  Y_pred range: [{Y_pred.min():.4f}, {Y_pred.max():.4f}]")
    print()

    # Compute Yamaguchi spectra
    if args.compute_yamaguchi:
        Y_yamaguchi = compute_yamaguchi_spectra(X_test, freqs, theta_bins, mp=args.mp)
        Y_yamaguchi = np.transpose(Y_yamaguchi, (0, 2, 1))
    else:
        Y_yamaguchi = None
        print("Skipping Yamaguchi computation (--compute_yamaguchi not set)")
    print()

    # Compute metrics
    print("Computing metrics...")
    print (Y_test.shape,Y_pred.shape, Y_yamaguchi.shape)
    metrics = compute_metrics(Y_test, Y_pred, Y_yamaguchi)
    print()

    # Print metrics
    print("="*80)
    print("QUANTITATIVE METRICS")
    print("="*80)
    print()
    print("Model Performance:")
    print(f"  RMSE:           {metrics['model_rmse']:.6f}")
    print(f"  MAE:            {metrics['model_mae']:.6f}")
    print(f"  R²:             {metrics['model_r2']:.4f}")
    print(f"  Peak Error:     {metrics['model_peak_error']*100:.2f}%")
    print(f"  Integral Error: {metrics['model_integral_error']*100:.2f}%")
    print()

    if Y_yamaguchi is not None:
        print("Yamaguchi (JONSWAP) Performance:")
        print(f"  RMSE:           {metrics['yamaguchi_rmse']:.6f}")
        print(f"  MAE:            {metrics['yamaguchi_mae']:.6f}")
        print(f"  R²:             {metrics['yamaguchi_r2']:.4f}")
        print(f"  Peak Error:     {metrics['yamaguchi_peak_error']*100:.2f}%")
        print(f"  Integral Error: {metrics['yamaguchi_integral_error']*100:.2f}%")
        print()

        print("Improvement over Yamaguchi:")
        print(f"  RMSE:           {metrics['improvement_rmse']:+.1f}%")
        print(f"  MAE:            {metrics['improvement_mae']:+.1f}%")
        print(f"  R² improvement: {metrics['improvement_r2']:+.1f} points")
        print()

    print("="*80)
    print()

    # Save metrics
    metrics_file = output_dir / 'metrics_comparison.txt'
    with open(metrics_file, 'w') as f:
        f.write("WAVE SPECTRA RECONSTRUCTION - METRICS COMPARISON\n")
        f.write("="*80 + "\n\n")
        f.write("Model Performance:\n")
        for key in ['model_rmse', 'model_mae', 'model_r2', 'model_peak_error', 'model_integral_error']:
            f.write(f"  {key}: {metrics[key]}\n")

        if Y_yamaguchi is not None:
            f.write("\nYamaguchi Performance:\n")
            for key in ['yamaguchi_rmse', 'yamaguchi_mae', 'yamaguchi_r2', 'yamaguchi_peak_error', 'yamaguchi_integral_error']:
                f.write(f"  {key}: {metrics[key]}\n")
            f.write("\nImprovement:\n")
            for key in ['improvement_rmse', 'improvement_mae', 'improvement_r2']:
                f.write(f"  {key}: {metrics[key]}\n")

    print(f"✓ Metrics saved to: {metrics_file}")

    # Save predictions as numpy arrays
    np.save(output_dir / 'Y_test.npy', Y_test)
    np.save(output_dir / 'Y_pred.npy', Y_pred)
    np.save(output_dir / 'X_test.npy', X_test)
    if Y_yamaguchi is not None:
        np.save(output_dir / 'Y_yamaguchi.npy', Y_yamaguchi)
    print(f"✓ Arrays saved to: {output_dir}")
    print()

    # Generate visualizations
    if args.num_samples > 0:
        print(f"Generating visualizations for {args.num_samples} samples...")

        # Select samples (random or sequential)
        n_test = Y_test.shape[0]
        if args.random_samples:
            indices = np.random.choice(n_test, min(args.num_samples, n_test), replace=False)
        else:
            indices = range(min(args.num_samples, n_test))

        for i, idx in enumerate(tqdm(list(indices), desc="Creating plots")):
            save_path = output_dir / f'comparison_sample_{idx:04d}.png'

            y_yam = Y_yamaguchi[idx] if Y_yamaguchi is not None else np.zeros_like(Y_test[idx])

            plot_comparison(
                idx, Y_test[idx], Y_pred[idx], y_yam,
                X_test, freqs, dires, k_bins, theta_bins,
                save_path, mp=args.mp
            )

        print(f"✓ Visualizations saved to: {output_dir}")
        print()

    # Plot summary statistics
    if Y_yamaguchi is not None:
        print("Generating summary plots...")

        # Error distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        model_errors = (Y_test - Y_pred).flatten()
        yam_errors = (Y_test - Y_yamaguchi).flatten()

        axes[0].hist(model_errors, bins=100, alpha=0.7, label='Model', color='blue')
        axes[0].hist(yam_errors, bins=100, alpha=0.7, label='Yamaguchi', color='orange')
        axes[0].set_xlabel('Error (Truth - Prediction)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # RMSE per sample
        model_rmse_per_sample = np.sqrt(np.mean((Y_test - Y_pred) ** 2, axis=(1, 2)))
        yam_rmse_per_sample = np.sqrt(np.mean((Y_test - Y_yamaguchi) ** 2, axis=(1, 2)))

        axes[1].scatter(yam_rmse_per_sample, model_rmse_per_sample, alpha=0.5, s=10)
        axes[1].plot([0, yam_rmse_per_sample.max()], [0, yam_rmse_per_sample.max()],
                    'r--', label='Equal performance')
        axes[1].set_xlabel('Yamaguchi RMSE')
        axes[1].set_ylabel('Model RMSE')
        axes[1].set_title('Per-Sample RMSE Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'summary_statistics.png', dpi=150)
        plt.close()

        print(f"✓ Summary plots saved")
        print()

    print("="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print(f"Results saved in: {output_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference and comparison for wave spectra reconstruction')

    # Model and data paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--stats_dir', type=str, required=True,
                       help='Directory containing wave stats zarr files')
    parser.add_argument('--spc_dir', type=str, required=True,
                       help='Directory containing wave spectra zarr files')
    parser.add_argument('--depth', type=str, required=True,
                       help='Path to depth NetCDF file')
    parser.add_argument('--scaler', type=str, required=True,
                       help='Path to scaler configuration JSON file')

    # Model configuration
    parser.add_argument('--model_name', type=str, default='hybrid',
                       help='Model architecture name')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for inference')

    # Data configuration
    parser.add_argument('--mp', action='store_true', default=True,
                       help='Use multi-partition mode (wind waves + swells)')
    parser.add_argument('--wind', action='store_true', default=False,
                       help='Include wind components')
    parser.add_argument('--coords', action='store_true', default=False,
                       help='Include coordinates')
    parser.add_argument('--decimate', type=int, default=100,
                       help='Decimation factor for input data')

    # Analysis options
    parser.add_argument('--compute_yamaguchi', action='store_true', default=True,
                       help='Compute Yamaguchi (JONSWAP) approximation for comparison')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to visualize (0 to skip visualization)')
    parser.add_argument('--random_samples', action='store_true', default=False,
                       help='Select random samples instead of sequential')

    args = parser.parse_args()

    # Set model_params based on model_name (you can customize this)
    if args.model_name == 'hybrid':
        args.model_params = {'hidden_dim': 512, 'num_res_blocks': 4}
    elif args.model_name == 'transformer':
        args.model_params = {'d_model': 256, 'nhead': 8, 'num_layers': 4}
    elif args.model_name == 'unet':
        args.model_params = {'hidden_dim': 256, 'channels': (16, 32, 64, 128)}
    else:
        args.model_params = {}

    main(args)
