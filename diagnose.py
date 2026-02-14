#!/usr/bin/env python3
"""
Diagnostic script to identify training issues
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from reader import Reader, build_file_pairs, CreateDataset

def diagnose_training(config):
    print("="*70)
    print("TRAINING DIAGNOSTICS")
    print("="*70)
    
    # 1. Check scalers
    print("\n1. Checking scalers...")
    with open(config['scaler'], 'r') as f:
        scaler = json.load(f)
    
    print(f"   EF scale: {scaler['EF']['scale']:.6e}")
    print(f"   EF offset: {scaler['EF']['offset']:.6e}")
    
    if abs(scaler['EF']['scale']) < 1e-10:
        print("   ⚠️  WARNING: EF scale is too small!")
    
    # 2. Load training data
    print("\n2. Loading training data...")
    pairs = build_file_pairs(config['stats_path'], config['spc_path'], 
                             fname='*2025*.zarr')
    dataset = CreateDataset(pairs[:10], Reader, config)
    
    print(f"   Found {len(dataset)} samples")
    
    # 3. Check data quality
    print("\n3. Checking data quality...")
    X_samples = []
    Y_samples = []
    
    for i in range(min(100, len(dataset))):
        X, Y = dataset[i]
        X_samples.append(X)
        Y_samples.append(Y)
    
    X_samples = np.array(X_samples)
    Y_samples = np.array(Y_samples)
    
    print(f"   X shape: {X_samples.shape}")
    print(f"   X range: [{X_samples.min():.4f}, {X_samples.max():.4f}]")
    print(f"   X mean: {X_samples.mean():.4f}")
    print(f"   X has NaN: {np.isnan(X_samples).any()}")
    
    print(f"   Y shape: {Y_samples.shape}")
    print(f"   Y range: [{Y_samples.min():.6e}, {Y_samples.max():.6e}]")
    print(f"   Y mean: {Y_samples.mean():.6e}")
    print(f"   Y has NaN: {np.isnan(Y_samples).any()}")
    
    # 4. Visualize samples
    print("\n4. Creating visualizations...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(6):
        ax = axes[i//3, i%3]
        ax.imshow(Y_samples[i, 0], origin='lower')
        ax.set_title(f'Training Sample {i}')
        ax.set_xlabel('Frequency bins')
        ax.set_ylabel('Direction bins')
    plt.tight_layout()
    plt.savefig('training_samples_diagnostic.png', dpi=150)
    print("   Saved: training_samples_diagnostic.png")
    
    # 5. Check spectral statistics
    print("\n5. Spectral statistics:")
    Y_max = Y_samples.max(axis=(1,2,3))
    Y_sum = Y_samples.sum(axis=(1,2,3))
    
    print(f"   Peak energy range: [{Y_max.min():.4e}, {Y_max.max():.4e}]")
    print(f"   Total energy range: [{Y_sum.min():.4e}, {Y_sum.max():.4e}]")
    
    # 6. Recommendations
    print("\n6. Recommendations:")
    if Y_max.max() < 1e-6:
        print("   ⚠️  Spectra values are very small - check scaling!")
    if Y_max.max() > 1e6:
        print("   ⚠️  Spectra values are very large - check scaling!")
    if np.isnan(X_samples).any() or np.isnan(Y_samples).any():
        print("   ❌ NaN values detected - clean your data!")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    config = {
    'stats_path': '/work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/stats_grid',
    'spc_path': '/work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/spcs_grid',
    'scaler': '/work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/scalers/scalers_partitions.json',
    'depth':'/work/cmcc/ww3_cst-dev/work/ML/preprocessing/data/SS_2026/stats_grid/dpt.nc',
    'decimate_input': 10,
    'wind': False,
    'add_coords': False,
    }
    
    diagnose_training(config)

