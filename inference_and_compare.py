"""
COMPLETE FAST INFERENCE with Yamaguchi Comparison and Visualization

This version includes:
✓ Fast batch processing (GPU/CPU)
✓ Yamaguchi (JONSWAP) comparison
✓ Visualization plots
✓ Comprehensive metrics
✓ All your original features + speed optimization

Usage:
    python fast_inference_complete.py \
        --model_path output_unet_sparse_optimized/best_model.pt \
        --stats_dir /path/to/stats \
        --spc_dir /path/to/spcs \
        --depth /path/to/depth.nc \
        --scaler /path/to/scaler.json \
        --output_dir inference_results \
        --batch_size 128 \
        --num_samples 20 \
        --compute_yamaguchi
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import random
import xarray as xr

from yamaguchi import JONSWAP
from reader import Reader, build_file_pairs, CreateDataset
from utils import fixBCdir
from models_improved import get_model
from config_experiments import EXPERIMENTS

# Set seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_model(model_path, input_dim, k_bins, theta_bins, device):
    """Load model with correct configuration"""
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract experiment name
    path_parts = Path(model_path).parts
    exp_name = None
    for part in path_parts:
        if part.startswith('output_'):
            exp_name = part.replace('output_', '')
            break
    
    if exp_name is None:
        exp_name = 'unet_sparse_optimized'
    
    # Get config
    if exp_name in EXPERIMENTS:
        exp_config = EXPERIMENTS[exp_name]
        print(f"✓ Using config: {exp_name}")
    else:
        print(f"⚠️  Config '{exp_name}' not found, using default")
        exp_config = {
            'model_name': 'unet',
            'model_params': {
                'hidden_dim': 512,
                'num_blocks': 4,
                'channels': [64, 128, 256, 512],
                'dropout': 0.1,
                'use_batchnorm': True,
                'use_attention': True
            }
        }
    
    # Build model
    model_name = exp_config.get('model_name', 'unet')
    model_params = exp_config.get('model_params', {})
    
    print(f"Building model: {model_name}")
    print(f"  Params: {model_params}")
    
    model = get_model(
        model_name=model_name,
        input_dim=input_dim,
        reshape_size=(k_bins, theta_bins),
        **model_params
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
    else:
        model = checkpoint
        epoch = 'unknown'
    
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (epoch {epoch}) on {device}")
    return model


def inverse_scale_spectra(scaled_spectra, scaler_dict):
    """Inverse transform scaled spectra back to original units"""
    scale = scaler_dict['EF']['scale']
    offset = scaler_dict['EF']['offset']
    unscaled = (scaled_spectra - offset) / scale
    return unscaled


def inverse_scale_inputs(scaled_inputs, scaler_dict, feature_names):
    """Inverse transform scaled input features"""
    unscaled = np.zeros_like(scaled_inputs)
    
    for i, name in enumerate(feature_names):
        if name in scaler_dict:
            scale = scaler_dict[name]['scale']
            offset = scaler_dict[name]['offset']
            unscaled[:, i] = (scaled_inputs[:, i] - offset) / scale
        else:
            print(f"Warning: {name} not found in scaler_dict, using scaled values")
            unscaled[:, i] = scaled_inputs[:, i]
    
    return unscaled


def compute_yamaguchi_spectra(X_test, freqs, theta_bins, mp=True):
    """Compute Yamaguchi (JONSWAP) spectra for comparison"""
    print("\nComputing Yamaguchi approximations...")
    
    yamaguchi_spectra = []
    n_samples = X_test.shape[0]
    
    for i in tqdm(range(n_samples), desc="Yamaguchi computation"):
        if mp:
            hs = X_test[i, 0]
            tp = X_test[i, 1]
            dir_deg = X_test[i, 2]
        else:
            hs = X_test[i, 0]
            tp = X_test[i, 1]
            dir_deg = X_test[i, 3]
        
        dir_deg = fixBCdir(dir_deg)
        
        _, dimSpec = JONSWAP(hs, tp, dir_deg, theta_bins, freqs).main()
        yamaguchi_spectra.append(dimSpec.T[:, ::-1])
    
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
    metrics['model_peak_error'] = abs(y_true_max - y_pred_max) / y_true_max if y_true_max > 0 else 0
    
    # Integral metrics
    y_true_sum = np.sum(y_true, axis=(1, 2)).mean()
    y_pred_sum = np.sum(y_pred, axis=(1, 2)).mean()
    metrics['model_integral_error'] = abs(y_true_sum - y_pred_sum) / y_true_sum if y_true_sum > 0 else 0
    
    # Yamaguchi metrics
    if y_yamaguchi is not None:
        metrics['yamaguchi_mse'] = np.mean((y_true - y_yamaguchi) ** 2)
        metrics['yamaguchi_rmse'] = np.sqrt(metrics['yamaguchi_mse'])
        metrics['yamaguchi_mae'] = np.mean(np.abs(y_true - y_yamaguchi))
        
        ss_res_yam = np.sum((y_true - y_yamaguchi) ** 2)
        metrics['yamaguchi_r2'] = 1 - (ss_res_yam / ss_tot) if ss_tot > 0 else 0
        
        y_yam_max = np.max(y_yamaguchi, axis=(1, 2)).mean()
        metrics['yamaguchi_peak_error'] = abs(y_true_max - y_yam_max) / y_true_max if y_true_max > 0 else 0
        
        y_yam_sum = np.sum(y_yamaguchi, axis=(1, 2)).mean()
        metrics['yamaguchi_integral_error'] = abs(y_true_sum - y_yam_sum) / y_true_sum if y_true_sum > 0 else 0
        
        # Improvement
        if metrics['yamaguchi_rmse'] > 0:
            metrics['improvement_rmse'] = (metrics['yamaguchi_rmse'] - metrics['model_rmse']) / metrics['yamaguchi_rmse'] * 100
        else:
            metrics['improvement_rmse'] = 0
            
        if metrics['yamaguchi_mae'] > 0:
            metrics['improvement_mae'] = (metrics['yamaguchi_mae'] - metrics['model_mae']) / metrics['yamaguchi_mae'] * 100
        else:
            metrics['improvement_mae'] = 0
            
        metrics['improvement_r2'] = (metrics['model_r2'] - metrics['yamaguchi_r2']) * 100
    
    return metrics


def plot_comparison(idx, y_true, y_pred, y_yamaguchi, X_test, freqs, dires,
                   k_bins, theta_bins, save_path, mp=True):
    """Create comparison plot for a single sample"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Ground truth
    im1 = axes[0, 0].imshow(y_true, aspect='auto', cmap='viridis', origin='lower')
    axes[0, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Frequency bins')
    axes[0, 0].set_ylabel('Direction bins')
    plt.colorbar(im1, ax=axes[0, 0], label='E [m²/Hz/deg]')
    
    # Model prediction
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
    
    im3 = axes[0, 2].imshow(y_yamaguchi, aspect='auto', cmap='viridis', origin='lower')
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
    
    # Metrics text
    mse_model = np.mean((y_true - y_pred) ** 2)
    mse_yam = np.mean((y_true - y_yamaguchi) ** 2)
    
    metrics_text = f"Sample {idx}\n"
    metrics_text += f"Model MSE: {mse_model:.6f}\n"
    metrics_text += f"Yamaguchi MSE: {mse_yam:.6f}\n"
    if mse_yam > 0:
        metrics_text += f"Improvement: {(mse_yam - mse_model) / mse_yam * 100:.1f}%"
    
    ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_statistics(Y_test, Y_pred, Y_yamaguchi, output_dir):
    """Generate summary plots"""
    print("\nGenerating summary plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error distribution
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


def run_fast_inference(args):
    """Fast batch inference with all original features"""
    
    print("\n" + "="*80)
    print("COMPLETE FAST INFERENCE - Speed + Yamaguchi + Visualization")
    print("="*80 + "\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("⚠️  WARNING: Running on CPU. For faster inference, use GPU!")
        print("   Submit with: sbatch run_inference_gpu.sh\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Load scaler
    print("Loading scaler...")
    with open(args.scaler, 'r') as f:
        scaler_dict = json.load(f)
    print(f"✓ Scaler loaded\n")
    
    # Build file pairs
    print("Finding files...")
    file_pairs = build_file_pairs(args.stats_dir, args.spc_dir, fname="*202501*.zarr")
    n_files = len(file_pairs)
    print(f"✓ Found {n_files} file pairs\n")
    
    if n_files == 0:
        raise ValueError("No files found!")
    
    # Get dimensions
    print("Loading data dimensions...")
    sample_spc = xr.open_zarr(file_pairs[0][1])
    freqs = sample_spc.frequency.values
    dires = sample_spc.direction.values
    k_bins = len(freqs)
    theta_bins = len(dires)
    input_dim = 9
    print(f"✓ Dimensions: input={input_dim}, output=({k_bins}, {theta_bins})")
    print(f"  Frequencies: {k_bins} bins")
    print(f"  Directions: {theta_bins} bins\n")
    
    # Load model
    model = load_model(args.model_path, input_dim, k_bins, theta_bins, device)
    print()
    
    # Create dataset and dataloader
    print("Creating dataset...")
    config = {
        'depth': args.depth,
        'scaler': args.scaler,
        'wind': args.wind,
        'add_coords': args.coords,
        'decimate_input': args.decimate
    }
    
    dataset = CreateDataset(file_pairs, Reader, config)
    print(f"✓ Dataset created: {len(dataset)} samples\n")
    
    # DataLoader with multiple workers for speed
    num_workers = min(4, os.cpu_count() or 1)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    print(f"Running inference...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Device: {device}\n")
    
    # Run inference
    predictions = []
    ground_truth = []
    inputs = []
    
    start_time = time.time()
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, (X_batch, Y_batch) in enumerate(tqdm(dataloader, desc="Inference")):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            # Predict
            Y_pred = model(X_batch)
            
            # Store results
            predictions.append(Y_pred.cpu().numpy())
            ground_truth.append(Y_batch.cpu().numpy())
            inputs.append(X_batch.cpu().numpy())
            
            samples_processed += X_batch.shape[0]
    
    # Concatenate results
    X_test_scaled = np.concatenate(inputs, axis=0)
    Y_test_scaled = np.concatenate(ground_truth, axis=0)
    Y_pred_scaled = np.concatenate(predictions, axis=0)
    
    total_time = time.time() - start_time
    avg_speed = samples_processed / total_time
    
    print(f"\n✓ Inference complete!")
    print(f"  Total samples: {samples_processed}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average speed: {avg_speed:.1f} samples/sec\n")
    
    # Remove channel dimension if present
    if len(Y_test_scaled.shape) == 4:
        Y_test_scaled = Y_test_scaled.squeeze(1)
    if len(Y_pred_scaled.shape) == 4:
        Y_pred_scaled = Y_pred_scaled.squeeze(1)
    
    # Inverse transform to physical units
    print("Converting to physical units...")
    
    feature_names = [
        'VHM0_WW', 'VTM01_WW', 'VMDR_WW',
        'VHM0_SW1', 'VTM01_SW1', 'VMDR_SW1',
        'VHM0_SW2', 'VTM01_SW2', 'VMDR_SW2'
    ]
    
    X_test = inverse_scale_inputs(X_test_scaled, scaler_dict, feature_names)
    Y_test = inverse_scale_spectra(Y_test_scaled, scaler_dict)
    Y_pred = inverse_scale_spectra(Y_pred_scaled, scaler_dict)
    
    # Transpose Y_test to match expected shape
    #Y_test = np.transpose(Y_test, (0, 2, 1))
    print(f"  X_test range: [{X_test.min():.4f}, {X_test.max():.4f}]")
    print(f"  Y_test range: [{Y_test.min():.4f}, {Y_test.max():.4f}]")
    print(f"  Y_pred range: [{Y_pred.min():.4f}, {Y_pred.max():.4f}]")
    
    # Compute Yamaguchi spectra
    if args.compute_yamaguchi:
        Y_yamaguchi = compute_yamaguchi_spectra(X_test, freqs, theta_bins, mp=args.mp)
        #Y_yamaguchi = np.transpose(Y_yamaguchi, (0, 2, 1))
    else:
        Y_yamaguchi = None
        print("\nSkipping Yamaguchi computation")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(Y_test, Y_pred, Y_yamaguchi)
    
    # Print metrics
    print("\n" + "="*80)
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
    
    print("="*80 + "\n")
    
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
    
    # Save arrays
    np.save(output_dir / 'Y_test.npy', Y_test)
    np.save(output_dir / 'Y_pred.npy', Y_pred)
    np.save(output_dir / 'X_test.npy', X_test)
    if Y_yamaguchi is not None:
        np.save(output_dir / 'Y_yamaguchi.npy', Y_yamaguchi)
    print(f"✓ Arrays saved to: {output_dir}")
    
    # Generate visualizations
    if args.num_samples > 0 and Y_yamaguchi is not None:
        print(f"\nGenerating visualizations for {args.num_samples} samples...")
        
        n_test = Y_test.shape[0]
        if args.random_samples:
            indices = np.random.choice(n_test, min(args.num_samples, n_test), replace=False)
        else:
            indices = range(min(args.num_samples, n_test))
        
        for idx in tqdm(list(indices), desc="Creating plots"):
            save_path = output_dir / f'comparison_sample_{idx:04d}.png'
            
            plot_comparison(
                idx, Y_test[idx], Y_pred[idx], Y_yamaguchi[idx],
                X_test, freqs, dires, k_bins, theta_bins,
                save_path, mp=args.mp
            )
        
        print(f"✓ Visualizations saved to: {output_dir}")
    
    # Summary plots
    if Y_yamaguchi is not None:
        plot_summary_statistics(Y_test, Y_pred, Y_yamaguchi, output_dir)
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print(f"Results saved in: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fast inference with Yamaguchi and visualization')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--stats_dir', type=str, required=True,
                       help='Stats directory')
    parser.add_argument('--spc_dir', type=str, required=True,
                       help='Spectra directory')
    parser.add_argument('--depth', type=str, required=True,
                       help='Depth file')
    parser.add_argument('--scaler', type=str, required=True,
                       help='Scaler JSON')
    
    # Data configuration
    parser.add_argument('--mp', action='store_true', default=True,
                       help='Use multi-partition mode')
    parser.add_argument('--wind', action='store_true', default=False,
                       help='Include wind components')
    parser.add_argument('--coords', action='store_true', default=False,
                       help='Include coordinates')
    parser.add_argument('--decimate', type=int, default=1,
                       help='Decimation factor')
    
    # Analysis options
    parser.add_argument('--compute_yamaguchi', action='store_true', default=True,
                       help='Compute Yamaguchi approximation')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (larger = faster on GPU)')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to visualize (0 to skip)')
    parser.add_argument('--random_samples', action='store_true', default=False,
                       help='Select random samples')
    
    args = parser.parse_args()
    
    run_fast_inference(args)
