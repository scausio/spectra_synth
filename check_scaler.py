#!/usr/bin/env python3
"""
Diagnostic script to check if scaler is working in inference
"""
import sys
import os

print("="*80)
print("SCALER DIAGNOSTIC SCRIPT")
print("="*80)
print()

# Check if model directory exists
model_dir = "output_hybrid_balanced"
if not os.path.exists(model_dir):
    print(f"❌ Model directory not found: {model_dir}")
    print("   Please provide the correct path to your model directory")
    sys.exit(1)

print(f"✓ Model directory exists: {model_dir}")
print()

# Check for scaler file
scaler_path = os.path.join(model_dir, "scaler.pkl")
if os.path.exists(scaler_path):
    print(f"✓ Scaler file exists: {scaler_path}")
    
    # Try to load it
    try:
        import joblib
        scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded successfully")
        print(f"  Type: {type(scaler)}")
        
        # Check if it has the right attributes
        if hasattr(scaler, 'data_min_'):
            print(f"  data_min_ shape: {scaler.data_min_.shape}")
            print(f"  data_min_ range: [{scaler.data_min_.min():.6f}, {scaler.data_min_.max():.6f}]")
        if hasattr(scaler, 'data_max_'):
            print(f"  data_max_ shape: {scaler.data_max_.shape}")
            print(f"  data_max_ range: [{scaler.data_max_.min():.6f}, {scaler.data_max_.max():.6f}]")
        if hasattr(scaler, 'feature_range'):
            print(f"  feature_range: {scaler.feature_range}")
            
    except Exception as e:
        print(f"❌ Failed to load scaler: {e}")
else:
    print(f"❌ Scaler file NOT found: {scaler_path}")
    print()
    print("This is the problem! The scaler wasn't saved during training.")
    print()
    print("Solutions:")
    print("1. Retrain the model with train.py (it saves the scaler)")
    print("2. Or check if the scaler is in a different location")

print()
print("="*80)

# Check what files ARE in the model directory
print(f"Files in {model_dir}:")
if os.path.exists(model_dir):
    for f in sorted(os.listdir(model_dir)):
        fpath = os.path.join(model_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  {f:40s} {size:>12,} bytes")
print("="*80)
