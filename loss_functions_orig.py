"""
Loss Functions for Wave Spectra Reconstruction

Contains various loss functions including:
- MSE, RMSE
- MSLE (Mean Squared Logarithmic Error)
- Constrained losses with physical penalties
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    """
    Root Mean Squared Error Loss
    
    RMSE = sqrt(mean((y_pred - y_true)²))
    
    Note: For optimization, MSE and RMSE are equivalent since:
    - sqrt is monotonic
    - gradients have same direction
    
    But RMSE gives more interpretable loss values in the same units as data.
    """
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


class MSLELoss(nn.Module):
    """
    Mean Squared Logarithmic Error
    
    MSLE = mean((log(y_pred + 1) - log(y_true + 1))²)
    
    Advantages:
    - Handles multi-scale data (values spanning orders of magnitude)
    - Treats relative errors more uniformly
    - More robust to outliers
    - Prevents model from ignoring small values
    
    Best for: Wave spectra with large dynamic range
    """
    def __init__(self):
        super(MSLELoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        # Ensure non-negative values
        y_pred = torch.clamp(y_pred, min=1e-7)
        y_true = torch.clamp(y_true, min=1e-7)
        
        # Compute log(1 + x) which is numerically stable
        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)
        
        return F.mse_loss(log_pred, log_true)


class MSLELossContraint(nn.Module):
    """
    MSLE Loss with Physical Constraints
    
    Loss = MSLE + alpha * (integral_error)² + beta * (peak_error)² + gamma * (custom_penalty)
    
    Components:
    - MSLE: Base reconstruction loss
    - Integral constraint: Energy conservation (sum of spectrum ≈ Hs²)
    - Peak constraint: Maximum value preservation
    - Custom penalty: Optional additional constraint
    
    Args:
        alpha: Weight for integral/energy conservation (default: 0.0001)
        beta: Weight for peak preservation (default: 0.0001)
        gamma: Weight for custom penalty (default: 0)
    
    Best for: Physically-constrained wave spectra reconstruction
    """
    def __init__(self, alpha=0.0001, beta=0.0001, gamma=0):
        super(MSLELossContraint, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.msle = MSLELoss()
    
    def forward(self, y_pred, y_true):
        # Base MSLE loss
        base_loss = self.msle(y_pred, y_true)
        
        # Reshape for easier computation: (batch, channels, height, width) -> (batch, -1)
        batch_size = y_pred.size(0)
        y_pred_flat = y_pred.view(batch_size, -1)
        y_true_flat = y_true.view(batch_size, -1)
        
        # Integral/Energy conservation constraint
        # Penalize if total energy differs
        pred_integral = y_pred_flat.sum(dim=1)  # Sum over all bins
        true_integral = y_true_flat.sum(dim=1)
        integral_error = F.mse_loss(pred_integral, true_integral)
        
        # Peak preservation constraint
        # Penalize if maximum value differs
        pred_peak = y_pred_flat.max(dim=1)[0]  # Max over all bins
        true_peak = y_true_flat.max(dim=1)[0]
        peak_error = F.mse_loss(pred_peak, true_peak)
        
        # Optional: Additional custom penalty (e.g., shape constraint)
        # For now, this is zero unless gamma > 0
        custom_penalty = 0
        if self.gamma > 0:
            # Example: Penalize non-smoothness (you can customize this)
            # pred_gradient = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]).mean()
            custom_penalty = 0  # Placeholder
        
        # Total loss
        total_loss = (
            base_loss 
            + self.alpha * integral_error 
            + self.beta * peak_error 
            + self.gamma * custom_penalty
        )
        
        return total_loss


class MSLELossContraintRescaled(nn.Module):
    """
    MSLE Loss with Rescaled Physical Constraints
    
    Similar to MSLELossContraint but normalizes constraints by data scale.
    This makes alpha and beta hyperparameters more stable across different datasets.
    
    Loss = MSLE + alpha * (integral_error / mean_integral)² + beta * (peak_error / mean_peak)²
    """
    def __init__(self, alpha=0.0001, beta=0.0001, gamma=0):
        super(MSLELossContraintRescaled, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.msle = MSLELoss()
    
    def forward(self, y_pred, y_true):
        # Base MSLE loss
        base_loss = self.msle(y_pred, y_true)
        
        # Reshape
        batch_size = y_pred.size(0)
        y_pred_flat = y_pred.view(batch_size, -1)
        y_true_flat = y_true.view(batch_size, -1)
        
        # Integral constraint (rescaled)
        pred_integral = y_pred_flat.sum(dim=1)
        true_integral = y_true_flat.sum(dim=1)
        mean_integral = true_integral.mean() + 1e-8  # Avoid division by zero
        integral_error = F.mse_loss(pred_integral / mean_integral, true_integral / mean_integral)
        
        # Peak constraint (rescaled)
        pred_peak = y_pred_flat.max(dim=1)[0]
        true_peak = y_true_flat.max(dim=1)[0]
        mean_peak = true_peak.mean() + 1e-8
        peak_error = F.mse_loss(pred_peak / mean_peak, true_peak / mean_peak)
        
        # Total loss
        total_loss = (
            base_loss 
            + self.alpha * integral_error 
            + self.beta * peak_error
        )
        
        return total_loss


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss with frequency/direction-dependent weights
    
    Useful for emphasizing certain regions of the spectrum
    (e.g., peak region, low-frequency region)
    
    Args:
        weight_map: Tensor of weights with same shape as spectrum (optional)
    """
    def __init__(self, weight_map=None):
        super(WeightedMSELoss, self).__init__()
        self.weight_map = weight_map
    
    def forward(self, y_pred, y_true):
        squared_error = (y_pred - y_true) ** 2
        
        if self.weight_map is not None:
            # Apply spatial weights
            weighted_error = squared_error * self.weight_map
            return weighted_error.mean()
        else:
            return squared_error.mean()


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1 Loss)
    
    Less sensitive to outliers than MSE
    Behaves like MSE for small errors, L1 for large errors
    
    Args:
        delta: Threshold for switching between L2 and L1 (default: 1.0)
    """
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        return F.smooth_l1_loss(y_pred, y_true, beta=self.delta)


# Convenience function to get loss by name
def get_loss_function(loss_name, **kwargs):
    """
    Factory function to create loss functions by name
    
    Args:
        loss_name: Name of loss function
        **kwargs: Additional parameters for the loss function
    
    Returns:
        Loss function instance
    
    Example:
        criterion = get_loss_function('msle_constraint', alpha=0.001, beta=0.001)
    """
    losses = {
        'mse': nn.MSELoss,
        'rmse': RMSELoss,
        'mae': nn.L1Loss,
        'msle': MSLELoss,
        'msle_constraint': MSLELossContraint,
        'msle_constraint_rescaled': MSLELossContraintRescaled,
        'huber': HuberLoss,
        'weighted_mse': WeightedMSELoss,
    }
    
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(losses.keys())}")
    
    return losses[loss_name](**kwargs)

