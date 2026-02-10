#!/bin/bash

echo "======================================================================"
echo "Installing Improved Wave Spectra Reconstruction Models"
echo "======================================================================"
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "ERROR: This script must be run from your spectra_synth git repository"
    echo "Please cd to your repository directory first"
    exit 1
fi

# Check if files already exist
if [ -f "models_improved.py" ]; then
    echo "WARNING: Some files already exist. Do you want to overwrite them? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Installation cancelled"
        exit 0
    fi
fi

echo "Installing 10 new files..."
echo ""

# Extract the package
if [ -f "improved_models_package.tar.gz" ]; then
    tar -xzf improved_models_package.tar.gz
    echo "✓ Files extracted from tar.gz"
elif [ -f "improved_models_package.zip" ]; then
    unzip -q improved_models_package.zip
    echo "✓ Files extracted from zip"
else
    echo "ERROR: Package file not found. Please ensure improved_models_package.tar.gz or .zip is in this directory"
    exit 1
fi

# List installed files
echo ""
echo "Installed files:"
echo "  1. models_improved.py (21KB) - 6 model architectures"
echo "  2. train_improved.py (16KB) - Enhanced training script"
echo "  3. evaluate_models.py (15KB) - Evaluation tools"
echo "  4. config_experiments.py (5KB) - Experiment configs"
echo "  5. run_experiments.py (3KB) - CLI runner"
echo "  6. quick_reference.py (10KB) - Code examples"
echo "  7. README_IMPROVED.md (9KB) - Documentation"
echo "  8. SUMMARY.md (10KB) - Overview"
echo "  9. ARCHITECTURES.md (8KB) - Architecture details"
echo " 10. QUICKSTART.md (8KB) - Quick start guide"
echo ""

# Git operations
echo "======================================================================"
echo "Git Operations"
echo "======================================================================"
echo ""
echo "Would you like to commit these changes now? (y/n)"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    git add models_improved.py train.py config_experiments.py run_experiments.py evaluate_models.py quick_reference.py README_IMPROVED.md SUMMARY.md ARCHITECTURES.md QUICKSTART.md
    
    echo ""
    echo "Files staged for commit. Commit message:"
    echo ""
    cat << 'COMMIT_MSG'
feat: Add improved model architectures and training framework

- Add 6 new model architectures (AttentionFFNN, Transformer, UNet, ResNet, Hybrid, Lightweight)
- Implement enhanced training script with better logging and evaluation
- Add comprehensive metrics: R², MAE, RMSE, peak error, integral error, direction error
- Include learning rate scheduling and early stopping
- Add configuration system for easy experimentation
- Create evaluation and comparison tools with rich visualizations
- Add quick reference guide and documentation
- Improve code modularity with base classes and factory patterns
- Add gradient clipping and advanced optimization (AdamW)
- Include checkpoint management and training history tracking
COMMIT_MSG
    echo ""
    
    git commit -m "feat: Add improved model architectures and training framework

- Add 6 new model architectures (AttentionFFNN, Transformer, UNet, ResNet, Hybrid, Lightweight)
- Implement enhanced training script with better logging and evaluation
- Add comprehensive metrics: R², MAE, RMSE, peak error, integral error, direction error
- Include learning rate scheduling and early stopping
- Add configuration system for easy experimentation
- Create evaluation and comparison tools with rich visualizations
- Add quick reference guide and documentation
- Improve code modularity with base classes and factory patterns
- Add gradient clipping and advanced optimization (AdamW)
- Include checkpoint management and training history tracking"
    
    echo ""
    echo "✓ Changes committed!"
    echo ""
    echo "Would you like to push to GitHub now? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        git push origin main
        echo ""
        echo "✓ Changes pushed to GitHub!"
    else
        echo ""
        echo "Changes committed locally. Push later with: git push origin main"
    fi
else
    echo ""
    echo "Files installed but not committed. To commit manually:"
    echo "  git add models_improved.py train_improved.py config_experiments.py run_experiments.py evaluate_models.py quick_reference.py README_IMPROVED.md SUMMARY.md ARCHITECTURES.md QUICKSTART.md"
    echo "  git commit -m 'feat: Add improved model architectures'"
    echo "  git push origin main"
fi

echo ""
echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Read QUICKSTART.md to get started in 5 minutes"
echo "  2. Run your first experiment:"
echo "     python run_experiments.py --experiment hybrid_balanced"
echo "  3. Check SUMMARY.md for complete overview"
echo ""
echo "Happy modeling! 🌊"
