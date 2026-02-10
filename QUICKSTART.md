# 🚀 Quick Start Guide

Get started with the improved wave spectra reconstruction in 5 minutes!

## Step 1: Review Available Models

Run this to see all available model configurations:
```bash
python run_experiments.py --list
```

You'll see 7 pre-configured experiments:
- ⭐ **hybrid_balanced** - Best all-around choice (RECOMMENDED)
- **unet_deep** - Excellent for 2D spatial reconstruction
- **resnet_deep** - For complex patterns
- **transformer_large** - For parameter interactions
- **baseline_ffnn** - Simple baseline
- **lightweight_fast** - Fast inference
- **resnet_wide** - Maximum capacity

## Step 2: Run Your First Experiment

```bash
# Start with the recommended model
python run_experiments.py --experiment hybrid_balanced
```

This will:
- ✅ Load your data from the configured paths
- ✅ Train the Hybrid CNN-Attention model
- ✅ Save checkpoints every 10 epochs
- ✅ Apply early stopping (patience=15)
- ✅ Generate training curve plots
- ✅ Save the best model automatically

**Training time**: ~30-40 minutes on GPU, ~6-8 hours on CPU (for 100 epochs)

## Step 3: Monitor Training

Watch the console for progress:
```
Epoch 1/100
  Train Loss: 0.012345
  Val Loss: 0.011234, MSE: 0.000789, MAE: 0.0234, R²: 0.8756
  Peak Error: 0.0543, Integral Error: 0.0321
  Learning Rate: 1.00e-03
```

Training curves are saved every 10 epochs in:
- `output_hybrid_balanced/training_curves_10.png`
- `output_hybrid_balanced/training_curves_20.png`
- etc.

## Step 4: Evaluate Your Model

After training completes, evaluate the model:

```python
from evaluate_models import evaluate_single_model

data_config = {
    'spc_path': 'path/to/wave_spectra_Ita_red2.nc',
    'stats_path': 'path/to/wave_stats_Ita_red2.nc',
    'mp': True,
    'wind': False,
    'add_coords': False,
    'decimate_input': 100
}

# This creates comprehensive visualizations
evaluate_single_model(
    'output_hybrid_balanced/best_model_full.pt',
    data_config,
    'evaluation_results'
)
```

This generates:
- `evaluation_results/metrics.json` - All metrics
- `evaluation_results/sample_predictions.png` - 6 example predictions
- `evaluation_results/error_distribution.png` - Error analysis
- `evaluation_results/spectral_statistics.png` - Frequency/direction analysis

## Step 5: Use the Model for Inference

```python
import torch
import numpy as np
import joblib

# Load model and scalers
model = torch.load('output_hybrid_balanced/best_model_full.pt')
model.eval()

scalers = joblib.load('output_hybrid_balanced/scaler.pkl')
scaler_X = scalers['X']
scaler_Y = scalers['Y']

# Prepare input: [Hs_ww, Tm_ww, Dir_ww, Hs_sw1, Tm_sw1, Dir_sw1, 
#                 Hs_sw2, Tm_sw2, Dir_sw2, Depth]
input_data = np.array([[
    2.5,    # Significant wave height - wind waves (m)
    6.0,    # Mean period - wind waves (s)
    180.0,  # Direction - wind waves (degrees)
    1.2,    # Significant wave height - swell 1 (m)
    9.0,    # Mean period - swell 1 (s)
    270.0,  # Direction - swell 1 (degrees)
    0.5,    # Significant wave height - swell 2 (m)
    12.0,   # Mean period - swell 2 (s)
    90.0,   # Direction - swell 2 (degrees)
    50.0    # Depth (m)
]])

# Scale and predict
input_scaled = scaler_X.transform(input_data)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

with torch.no_grad():
    output = model(input_tensor)
    spectra_2d = output.cpu().numpy().reshape(32, 24)  # (frequency, direction)

print(f"Predicted spectra shape: {spectra_2d.shape}")
print(f"Max energy: {spectra_2d.max():.6f}")
print(f"Total energy: {spectra_2d.sum():.6f}")

# Optionally inverse transform if Y was scaled
# spectra_original = scaler_Y.inverse_transform(spectra_2d.reshape(1, -1)).reshape(32, 24)
```

## Step 6: Compare Multiple Models (Optional)

Try different models and compare:

```bash
# Run multiple experiments
python run_experiments.py --experiment hybrid_balanced
python run_experiments.py --experiment unet_deep
python run_experiments.py --experiment resnet_deep
```

Then compare them:

```python
from evaluate_models import compare_models

model_paths = {
    'Hybrid': 'output_hybrid_balanced/best_model_full.pt',
    'UNet': 'output_unet_deep/best_model_full.pt',
    'ResNet': 'output_resnet_deep/best_model_full.pt',
}

# Creates comparison table and plots
comparison_df = compare_models(model_paths, data_config, 'comparison/')
print(comparison_df)
```

Output:
```
           r2      rmse      mae     msle  peak_error  integral_error
Hybrid   0.923  0.01234  0.0189  0.00045      0.0421          0.0198
UNet     0.918  0.01289  0.0201  0.00051      0.0456          0.0223
ResNet   0.931  0.01156  0.0172  0.00038      0.0389          0.0176
```

## 📊 What to Look For

### Good Model Performance:
- ✅ R² > 0.85
- ✅ Peak Error < 0.10 (10%)
- ✅ Integral Error < 0.05 (5%)
- ✅ Direction Error < 20 degrees
- ✅ Training and validation loss converging

### Signs of Problems:
- ❌ R² < 0.70 → Try different model or tune hyperparameters
- ❌ Val loss >> Train loss → Overfitting (reduce model size or add dropout)
- ❌ Both losses high → Underfitting (increase model capacity)
- ❌ Loss not decreasing → Reduce learning rate

## 🔧 Customize Your Experiment

Edit `config_experiments.py` to create your own experiment:

```python
'my_experiment': {
    'model_name': 'hybrid',
    'model_params': {
        'hidden_dim': 1024,      # Increase capacity
        'num_res_blocks': 6,     # More layers
    },
    'learning_rate': 5e-4,       # Lower learning rate
    'batch_size': 16,            # Smaller batches
    'epochs': 150,               # Train longer
    'description': 'My custom configuration'
}
```

Then run it:
```bash
python run_experiments.py --experiment my_experiment
```

## 🎯 Recommended Workflow

For best results, follow this workflow:

1. **Baseline** (30 min):
   - Run `hybrid_balanced` or `baseline_ffnn`
   - Check if data loading works
   - Verify reasonable performance (R² > 0.80)

2. **Exploration** (2-3 hours):
   - Try 2-3 different models: `hybrid`, `unet`, `resnet`
   - Compare results
   - Identify best architecture

3. **Fine-tuning** (1-2 hours):
   - Adjust hyperparameters of best model
   - Try different learning rates
   - Experiment with model capacity

4. **Evaluation** (30 min):
   - Comprehensive evaluation with visualizations
   - Check all metrics
   - Validate on holdout set

5. **Deployment**:
   - Use best model for inference
   - Consider `lightweight` if speed is critical

## 📝 Important Files

After training, your output directory contains:

```
output_hybrid_balanced/
├── best_model_full.pt          # ⭐ Complete trained model (use this!)
├── best_model.pt               # Checkpoint with optimizer state
├── scaler.pkl                  # ⭐ Scalers for inference
├── training_history.json       # All metrics per epoch
├── final_training_curves.png   # Training visualization
├── train_loss.npy             # Training loss array
├── val_loss.npy               # Validation loss array
└── ckpt_epoch_*.pt            # Regular checkpoints
```

## 🆘 Need Help?

**Training Issues:**
- See `quick_reference.py` → TROUBLESHOOTING section
- Check `ARCHITECTURES.md` → HYPERPARAMETER SENSITIVITY

**Model Selection:**
- See `ARCHITECTURES.md` → DECISION TREE
- See `SUMMARY.md` → What Each Model Is Best For

**Code Examples:**
- See `quick_reference.py` → QUICK START EXAMPLES
- See `README_IMPROVED.md` → Complete documentation

**Architecture Details:**
- See `ARCHITECTURES.md` → Full architecture diagrams

## 🎉 You're Ready!

You now have everything you need to:
- ✅ Train multiple state-of-the-art models
- ✅ Evaluate and compare them
- ✅ Use the best model for inference
- ✅ Tune hyperparameters
- ✅ Monitor training progress
- ✅ Generate comprehensive visualizations

**Start now:**
```bash
python run_experiments.py --experiment hybrid_balanced
```

Good luck with your wave spectra reconstruction! 🌊
