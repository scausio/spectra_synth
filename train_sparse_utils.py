"""
Minimal integration patch for train.py

This adds the new loss functions while maintaining full backward compatibility
with your existing config structure and code.

USAGE:
------
1. Copy this file to your project
2. In train.py, replace the loss function selection section with:
   
   from train_sparse_utils import get_loss_function_enhanced
   criterion = get_loss_function_enhanced(config)

That's it! Your existing configs will work unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from YOUR existing loss_functions.py
from loss_functions import (
    MSLELoss, 
    MSLELossContraint, 
    MSLELossContraintRescaled,
    CombinedIntensityPositionLoss
)


# ============================================================================
# NEW LOSS FUNCTIONS FOR SPARSE DATA
# ============================================================================

class MSLEConstraintWeighted(nn.Module):
    """
    MSLE with physical constraints and weighting for sparse data
    
    Recommended for sparse spectral data where most values are near zero
    and a few peaks contain all the information.
    """
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.01, 
                 nonzero_weight=10.0, zero_weight=1.0, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.nonzero_weight = nonzero_weight
        self.zero_weight = zero_weight
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        # 1. MSLE (handles small values)
        log_pred = torch.log(y_pred + self.epsilon)
        log_true = torch.log(y_true + self.epsilon)
        msle_loss = F.mse_loss(log_pred, log_true)
        
        # 2. Peak preservation
        pred_max = y_pred.view(y_pred.size(0), -1).max(dim=1)[0]
        true_max = y_true.view(y_true.size(0), -1).max(dim=1)[0]
        peak_loss = F.mse_loss(pred_max, true_max)
        
        # 3. Energy conservation
        pred_sum = y_pred.view(y_pred.size(0), -1).sum(dim=1)
        true_sum = y_true.view(y_true.size(0), -1).sum(dim=1)
        energy_loss = F.mse_loss(pred_sum, true_sum)
        
        # 4. Weighted MSE (focus on non-zero values)
        nonzero_mask = (y_true > self.epsilon).float()
        weights = nonzero_mask * self.nonzero_weight + (1 - nonzero_mask) * self.zero_weight
        weighted_mse = (weights * (y_pred - y_true) ** 2).mean()
        
        # Combine losses
        total_loss = msle_loss + self.alpha * peak_loss + self.beta * energy_loss + self.gamma * weighted_mse
        
        return total_loss


class LogScaleLoss(nn.Module):
    """Pure log-scale MSE loss for very sparse data"""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        log_pred = torch.log(y_pred + self.epsilon)
        log_true = torch.log(y_true + self.epsilon)
        return F.mse_loss(log_pred, log_true)


class WeightedMSELoss(nn.Module):
    """MSE with high weight on non-zero spectral values"""
    def __init__(self, nonzero_weight=10.0, zero_weight=1.0, epsilon=1e-8):
        super().__init__()
        self.nonzero_weight = nonzero_weight
        self.zero_weight = zero_weight
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        nonzero_mask = (y_true > self.epsilon).float()
        weights = nonzero_mask * self.nonzero_weight + (1 - nonzero_mask) * self.zero_weight
        weighted_mse = (weights * (y_pred - y_true) ** 2).mean()
        return weighted_mse


class SpectralReconstructionLoss(nn.Module):
    """
    Comprehensive loss for 2D spectral reconstruction
    
    Combines pixel-wise accuracy, peak preservation, energy conservation,
    spatial structure, and sparsity preservation.
    """
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.01, delta=0.001, 
                 use_log_scale=True, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.use_log_scale = use_log_scale
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        # 1. Pixel-wise loss
        if self.use_log_scale:
            log_pred = torch.log(y_pred + self.epsilon)
            log_true = torch.log(y_true + self.epsilon)
            pixel_loss = F.mse_loss(log_pred, log_true)
        else:
            pixel_loss = F.mse_loss(y_pred, y_true)
        
        # 2. Peak preservation
        pred_max = y_pred.view(y_pred.size(0), -1).max(dim=1)[0]
        true_max = y_true.view(y_true.size(0), -1).max(dim=1)[0]
        peak_loss = F.mse_loss(pred_max, true_max)
        
        # 3. Energy conservation
        pred_sum = y_pred.view(y_pred.size(0), -1).sum(dim=1)
        true_sum = y_true.view(y_true.size(0), -1).sum(dim=1)
        energy_loss = F.mse_loss(pred_sum, true_sum)
        
        # 4. Spatial structure (gradient matching)
        pred_grad_h = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        true_grad_h = torch.abs(y_true[:, :, 1:, :] - y_true[:, :, :-1, :])
        pred_grad_w = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        true_grad_w = torch.abs(y_true[:, :, :, 1:] - y_true[:, :, :, :-1])
        gradient_loss = F.mse_loss(pred_grad_h, true_grad_h) + F.mse_loss(pred_grad_w, true_grad_w)
        
        # 5. Sparsity preservation
        sparsity_loss = torch.abs(y_pred).mean()
        
        # Combine
        total_loss = (pixel_loss + 
                     self.alpha * peak_loss + 
                     self.beta * energy_loss + 
                     self.gamma * gradient_loss + 
                     self.delta * sparsity_loss)
        
        return total_loss


# ============================================================================
# ENHANCED LOSS FUNCTION SELECTOR (BACKWARD COMPATIBLE)
# ============================================================================

def get_loss_function_enhanced(config):
    """
    Enhanced loss function selector that maintains backward compatibility
    with your existing config structure.
    
    Args:
        config: Dictionary with 'loss_function' key and optional 'loss_params'
    
    Returns:
        PyTorch loss function instance
    
    Usage in train.py:
        from train_sparse_utils import get_loss_function_enhanced
        criterion = get_loss_function_enhanced(config)
    """
    loss_name = config.get('loss_function', 'combined')
    
    # ========== NEW LOSS FUNCTIONS (use loss_params) ==========
    
    if loss_name == 'msle_constraint_weighted':
        loss_params = config.get('loss_params', {})
        return MSLEConstraintWeighted(
            alpha=loss_params.get('alpha', 0.15),
            beta=loss_params.get('beta', 0.15),
            gamma=loss_params.get('gamma', 0.02),
            nonzero_weight=loss_params.get('nonzero_weight', 20.0),
            zero_weight=loss_params.get('zero_weight', 1.0),
            epsilon=loss_params.get('epsilon', 1e-8)
        )
    
    elif loss_name == 'log_scale':
        loss_params = config.get('loss_params', {})
        return LogScaleLoss(epsilon=loss_params.get('epsilon', 1e-8))
    
    elif loss_name == 'weighted_mse':
        loss_params = config.get('loss_params', {})
        return WeightedMSELoss(
            nonzero_weight=loss_params.get('nonzero_weight', 10.0),
            zero_weight=loss_params.get('zero_weight', 1.0),
            epsilon=loss_params.get('epsilon', 1e-8)
        )
    
    elif loss_name == 'spectral_reconstruction':
        loss_params = config.get('loss_params', {})
        return SpectralReconstructionLoss(
            alpha=loss_params.get('alpha', 0.1),
            beta=loss_params.get('beta', 0.1),
            gamma=loss_params.get('gamma', 0.01),
            delta=loss_params.get('delta', 0.001),
            use_log_scale=loss_params.get('use_log_scale', True),
            epsilon=loss_params.get('epsilon', 1e-8)
        )
    
    # ========== EXISTING LOSS FUNCTIONS (use old parameters) ==========
    
    elif loss_name == 'mse':
        return nn.MSELoss()
    
    elif loss_name == 'msle':
        return MSLELoss()
    
    elif loss_name == 'msle_constraint':
        return MSLELossContraint(
            alpha=config.get('alpha', 0.0001),
            beta=config.get('beta', 0.0001),
            gamma=config.get('gamma', 0)
        )
    
    elif loss_name == 'msle_constraint_rescaled':
        return MSLELossContraintRescaled(
            alpha=config.get('alpha', 0.0001),
            beta=config.get('beta', 0.0001),
            gamma=config.get('gamma', 0)
        )
    
    elif loss_name == 'combined':
        # Use your existing CombinedIntensityPositionLoss
        return CombinedIntensityPositionLoss(
            alpha_intensity=config.get('alpha_intensity', 1.0),
            alpha_position=config.get('alpha_position', 0.3),
            alpha_moment=config.get('alpha_moment', 0.1),
            alpha_peak=config.get('alpha_peak', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Available: mse, msle, msle_constraint, msle_constraint_rescaled, combined, "
                        f"msle_constraint_weighted, log_scale, weighted_mse, spectral_reconstruction")


# ============================================================================
# CONVENIENCE FUNCTION FOR TESTING
# ============================================================================

def test_loss_functions():
    """Test all loss functions with dummy data"""
    import torch
    
    batch_size = 4
    y_true = torch.rand(batch_size, 1, 32, 24) * 0.02  # Sparse: [0, 0.02]
    y_pred = torch.rand(batch_size, 1, 32, 24) * 0.02
    
    print("Testing loss functions on sparse spectral data:")
    print(f"y_true range: [{y_true.min():.6f}, {y_true.max():.6f}]")
    print(f"y_pred range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
    print()
    
    # Test config for each loss
    configs = [
        {'loss_function': 'mse'},
        {'loss_function': 'msle'},
        {'loss_function': 'combined', 'alpha_intensity': 1.0, 'alpha_position': 0.3},
        {'loss_function': 'msle_constraint_weighted', 'loss_params': {'alpha': 0.15, 'beta': 0.15}},
        {'loss_function': 'log_scale', 'loss_params': {}},
        {'loss_function': 'weighted_mse', 'loss_params': {'nonzero_weight': 20.0}},
    ]
    
    for config in configs:
        try:
            loss_fn = get_loss_function_enhanced(config)
            loss_value = loss_fn(y_pred, y_true)
            print(f"{config['loss_function']:30s}: {loss_value.item():.6f}")
        except Exception as e:
            print(f"{config['loss_function']:30s}: ERROR - {e}")


if __name__ == "__main__":
    test_loss_functions()
