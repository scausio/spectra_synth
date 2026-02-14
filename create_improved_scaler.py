#!/usr/bin/env python3
"""
Create improved scalers that map sparse spectral data to [0.2, 0.8] range
"""
import json
import numpy as np
import sys
import os

def create_improved_scaler(original_scaler_path='scalers_partitions.json', 
                          output_path='scalers_partitions_improved.json',
                          target_min=0.2, target_max=0.8):
    """
    Create improved scaler that maps EF to [target_min, target_max]
    
    Current issue: EF max = 1.663182e-02, mean = 6.29e-05
    These small values give poor gradient signal with MSE
    
    Solution: Map to [0.2, 0.8] range for better training dynamics
    """
    
    print("=" * 80)
    print("Creating Improved Scaler for Sparse Spectral Data")
    print("=" * 80)
    
    # Load original scaler
    if not os.path.exists(original_scaler_path):
        print(f"❌ Original scaler not found: {original_scaler_path}")
        print("Using default values from diagnostic results...")
        original_scale = 2.854352e-03
        original_offset = 0.0
        # From diagnostics: Y range [0.000000e+00, 1.663182e-02]
        data_min = 0.0
        data_max = 1.663182e-02
    else:
        with open(original_scaler_path, 'r') as f:
            scalers = json.load(f)
        
        original_scale = scalers.get('EF', {}).get('scale', 2.854352e-03)
        original_offset = scalers.get('EF', {}).get('offset', 0.0)
        
        # Compute data range from original scaler
        # scaled = (raw - offset) * scale
        # If scaled_max ≈ 1.663182e-02, then:
        # raw_max = scaled_max / scale + offset
        scaled_max = 1.663182e-02
        data_max = scaled_max / original_scale + original_offset
        data_min = original_offset
    
    print(f"\n📊 Original Data Statistics:")
    print(f"   Data range: [{data_min:.6e}, {data_max:.6e}]")
    print(f"   Original scale: {original_scale:.6e}")
    print(f"   Original offset: {original_offset:.6e}")
    
    # Compute new scaler to map [data_min, data_max] → [target_min, target_max]
    # Formula: scaled = (raw - offset) * scale
    # We want: target_min when raw = data_min
    #          target_max when raw = data_max
    
    # Solve:
    # target_min = (data_min - offset) * scale
    # target_max = (data_max - offset) * scale
    
    # Setting offset = data_min:
    new_offset = data_min
    new_scale = (target_max - target_min) / (data_max - data_min)
    
    print(f"\n🎯 New Scaler Parameters:")
    print(f"   Target range: [{target_min}, {target_max}]")
    print(f"   New scale: {new_scale:.6e}")
    print(f"   New offset: {new_offset:.6e}")
    
    # Verify
    scaled_min = (data_min - new_offset) * new_scale
    scaled_max = (data_max - new_offset) * new_scale
    
    print(f"\n✅ Verification:")
    print(f"   data_min ({data_min:.6e}) → {scaled_min:.4f}")
    print(f"   data_max ({data_max:.6e}) → {scaled_max:.4f}")
    
    # Create complete scaler config
    # Copy other scalers from original if available
    if os.path.exists(original_scaler_path):
        with open(original_scaler_path, 'r') as f:
            all_scalers = json.load(f)
    else:
        all_scalers = {}
    
    # Update EF scaler
    all_scalers['EF'] = {
        'scale': float(new_scale),
        'offset': float(new_offset),
        'description': f'Maps EF from [0, {data_max:.6e}] to [{target_min}, {target_max}] for better gradient flow'
    }
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(all_scalers, f, indent=2)
    
    print(f"\n💾 Improved scaler saved to: {output_path}")
    print(f"\n📝 Next steps:")
    print(f"   1. Update config to use: 'scaler': '{output_path}'")
    print(f"   2. Re-train model with new scaler")
    print(f"   3. Expect Y_train in range [{target_min}, {target_max}]")
    print("=" * 80)
    
    return all_scalers

if __name__ == '__main__':
    if len(sys.argv) > 1:
        original = sys.argv[1]
    else:
        original = 'scalers_partitions.json'
    
    if len(sys.argv) > 2:
        output = sys.argv[2]
    else:
        output = 'scalers_partitions_improved.json'
    
    create_improved_scaler(original, output, target_min=0.2, target_max=0.8)
