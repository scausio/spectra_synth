#!/usr/bin/env python3
"""
Automatic patcher for train.py

This script automatically patches your train.py to use the new loss functions
while maintaining backward compatibility.

Usage:
    python patch_train.py

What it does:
    1. Backs up train.py to train.py.backup_<timestamp>
    2. Adds import statement for get_loss_function_enhanced
    3. Replaces loss function selection code with single line
    4. Keeps everything else unchanged
"""

import os
import sys
import re
from datetime import datetime


def patch_train_py():
    """Patch train.py to use enhanced loss functions"""
    
    train_py_path = 'train.py'
    
    # Check if file exists
    if not os.path.exists(train_py_path):
        print(f"❌ Error: {train_py_path} not found")
        print(f"   Current directory: {os.getcwd()}")
        sys.exit(1)
    
    # Read current content
    with open(train_py_path, 'r') as f:
        content = f.read()
    
    # Create backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f'train.py.backup_{timestamp}'
    with open(backup_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Backup created: {backup_path}")
    
    # ========== PATCH 1: Add import ==========
    import_statement = "from train_sparse_utils import get_loss_function_enhanced"
    
    if import_statement in content:
        print("✅ Import already exists")
    else:
        # Find the last import from loss_functions
        import_pattern = r'from loss_functions import.*'
        match = re.search(import_pattern, content)
        
        if match:
            # Add new import after existing loss_functions import
            pos = match.end()
            content = content[:pos] + '\n' + import_statement + content[pos:]
            print("✅ Added import statement")
        else:
            print("⚠️  Warning: Could not find loss_functions import")
            print("   Please manually add this line after your imports:")
            print(f"   {import_statement}")
    
    # ========== PATCH 2: Replace loss function selection ==========
    
    # Pattern to match the old loss selection code
    old_pattern = r"""
    # Select loss function
    if config\['loss_function'\] == 'mse':
        criterion = nn\.MSELoss\(\)
    elif config\['loss_function'\] == 'msle':
        criterion = MSLELoss\(\)
    elif config\['loss_function'\] == 'msle_constraint':
        criterion = MSLELossContraint\(
            alpha=config\.get\('alpha', 0\.0001\),
            beta=config\.get\('beta', 0\.0001\),
            gamma=config\.get\('gamma', 0\)
        \)
    else:
        criterion = MSLELoss\(\)
    """.strip()
    
    # More flexible pattern that captures various formatting
    flexible_pattern = r"""(?s)# Select loss function.*?if config\['loss_function'\].*?else:.*?criterion = .*?\n"""
    
    replacement = """# Select loss function (enhanced with sparse data support)
    criterion = get_loss_function_enhanced(config)
    logging.info(f"Loss function: {config['loss_function']}")
    if 'loss_params' in config:
        logging.info(f"Loss parameters: {config['loss_params']}")
"""
    
    # Try to find and replace
    if re.search(flexible_pattern, content):
        content_new = re.sub(flexible_pattern, replacement, content)
        if content_new != content:
            content = content_new
            print("✅ Replaced loss function selection code")
        else:
            print("⚠️  Warning: Pattern matched but replacement failed")
    else:
        print("⚠️  Warning: Could not find loss function selection code")
        print("   Please manually replace the loss selection section with:")
        print("   criterion = get_loss_function_enhanced(config)")
    
    # Write patched content
    with open(train_py_path, 'w') as f:
        f.write(content)
    
    print(f"\n✅ train.py has been patched!")
    print(f"\n📝 Summary:")
    print(f"   - Backup: {backup_path}")
    print(f"   - Added import: from train_sparse_utils import get_loss_function_enhanced")
    print(f"   - Replaced loss selection with: criterion = get_loss_function_enhanced(config)")
    print(f"\n🧪 Test it:")
    print(f"   python -c \"from train import main; print('✅ Import works')\"")


def manual_instructions():
    """Print manual instructions if automatic patching fails"""
    print("\n" + "="*80)
    print("MANUAL PATCHING INSTRUCTIONS")
    print("="*80)
    print("\n1. Open train.py in your editor")
    print("\n2. Add this import after the loss_functions import:")
    print("   from train_sparse_utils import get_loss_function_enhanced")
    print("\n3. Find the loss function selection code (around line 200-220):")
    print("   if config['loss_function'] == 'mse':")
    print("       criterion = nn.MSELoss()")
    print("   elif config['loss_function'] == 'msle':")
    print("       ...")
    print("\n4. Replace it with this single line:")
    print("   criterion = get_loss_function_enhanced(config)")
    print("\n5. Optionally add logging:")
    print("   logging.info(f\"Loss function: {config['loss_function']}\")")
    print("   if 'loss_params' in config:")
    print("       logging.info(f\"Loss parameters: {config['loss_params']}\")")
    print("\n" + "="*80)


if __name__ == "__main__":
    print("="*80)
    print("TRAIN.PY PATCHER - Adds sparse data loss functions")
    print("="*80)
    print()
    
    try:
        patch_train_py()
    except Exception as e:
        print(f"\n❌ Automatic patching failed: {e}")
        print("\nFalling back to manual instructions:")
        manual_instructions()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Test the integration:")
    print("   python train_sparse_utils.py")
    print("\n2. List new experiments:")
    print("   python config_experiments.py --sparse")
    print("\n3. Run a quick test:")
    print("   python run_experiments.py --experiment unet_sparse_optimized")
    print("\n" + "="*80)
