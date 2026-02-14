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


class CombinedIntensityPositionLoss(nn.Module):
    """
    RECOMMENDED FOR WAVE SPECTRA: Combined loss for BOTH intensity and spatial position
    
    Perfect for signals with multiple peaks where you need to match:
    1. Intensity/magnitude of peaks (using MSLE)
    2. Position of peaks in 2D space (using center-of-mass)
    3. Overall shape (using second moment)
    4. Peak preservation (ensuring important features maintained)
    
    This is SPECIFICALLY DESIGNED for your wave spectra reconstruction problem!
    
    Components:
    - Intensity matching: MSLE for multi-scale values
    - Position matching: Center-of-mass distance
    - Moment matching: Ensures shape similarity
    - Peak preservation: Ensures important features are maintained
    
    Usage:
        criterion = CombinedIntensityPositionLoss(
            alpha_intensity=1.0,   # Weight for intensity (MSLE)
            alpha_position=0.5,    # Weight for position (center-of-mass)
            alpha_moment=0.3,      # Weight for spread/shape
            alpha_peak=0.2         # Weight for peak preservation
        )
    
    Tuning guide:
    - Increase alpha_position if peaks are in wrong locations
    - Increase alpha_moment if shape/spread is incorrect
    - Increase alpha_peak if peak heights are wrong
    - Keep alpha_intensity at 1.0 as baseline
    """
    
    def __init__(self, 
                 alpha_intensity=1.0,
                 alpha_position=0.5,
                 alpha_moment=0.3,
                 alpha_peak=0.2,
                 reduction='mean'):
        super(CombinedIntensityPositionLoss, self).__init__()
        
        self.alpha_intensity = alpha_intensity
        self.alpha_position = alpha_position
        self.alpha_moment = alpha_moment
        self.alpha_peak = alpha_peak
        self.reduction = reduction
    
    def compute_center_of_mass(self, x):
        """
        Compute 2D center of mass (centroid) of the signal
        
        Args:
            x: [batch, 1, H, W]
        
        Returns:
            centers: [batch, 2] - (row, col) positions
        """
        batch_size, _, H, W = x.shape
        
        # Create coordinate grids
        row_coords = torch.arange(H, dtype=x.dtype, device=x.device).view(1, 1, H, 1)
        col_coords = torch.arange(W, dtype=x.dtype, device=x.device).view(1, 1, 1, W)
        
        # Normalize spectra to get probability-like distribution
        x_sum = x.sum(dim=(2, 3), keepdim=True) + 1e-8
        x_norm = x / x_sum
        
        # Compute weighted average position
        center_row = (x_norm * row_coords).sum(dim=(2, 3))
        center_col = (x_norm * col_coords).sum(dim=(2, 3))
        
        centers = torch.stack([center_row, center_col], dim=-1)
        
        return centers.squeeze(1)  # [batch, 2]
    
    def compute_second_moment(self, x):
        """
        Compute second moment (variance/spread) of the signal
        
        This captures how spread out the signal is from its center
        """
        batch_size, _, H, W = x.shape
        
        # Get center of mass
        centers = self.compute_center_of_mass(x)
        
        # Create coordinate grids
        row_coords = torch.arange(H, dtype=x.dtype, device=x.device).view(1, 1, H, 1)
        col_coords = torch.arange(W, dtype=x.dtype, device=x.device).view(1, 1, 1, W)
        
        # Normalize
        x_sum = x.sum(dim=(2, 3), keepdim=True) + 1e-8
        x_norm = x / x_sum
        
        # Compute variance
        center_row = centers[:, 0].view(-1, 1, 1, 1)
        center_col = centers[:, 1].view(-1, 1, 1, 1)
        
        var_row = (x_norm * (row_coords - center_row).pow(2)).sum(dim=(2, 3))
        var_col = (x_norm * (col_coords - center_col).pow(2)).sum(dim=(2, 3))
        
        # Combined spread measure
        spread = torch.sqrt(var_row.pow(2) + var_col.pow(2))
        
        return spread.squeeze(1)  # [batch]
    
    def forward(self, y_pred, y_true):
        """
        Compute combined loss
        
        Args:
            y_pred: [batch, 1, H, W] - Predicted spectra
            y_true: [batch, 1, H, W] - Ground truth spectra
        
        Returns:
            total_loss (scalar)
        """
        batch_size = y_pred.size(0)
        
        # 1. INTENSITY LOSS - Using MSLE for multi-scale values
        y_pred_safe = torch.clamp(y_pred, min=1e-7)
        y_true_safe = torch.clamp(y_true, min=1e-7)
        
        log_pred = torch.log1p(y_pred_safe)
        log_true = torch.log1p(y_true_safe)
        
        intensity_loss = F.mse_loss(log_pred, log_true, reduction=self.reduction)
        
        # 2. POSITION LOSS - Center of mass distance
        center_pred = self.compute_center_of_mass(y_pred)
        center_true = self.compute_center_of_mass(y_true)
        
        position_loss = F.mse_loss(center_pred, center_true, reduction=self.reduction)
        
        # 3. MOMENT LOSS - Spread/variance matching
        spread_pred = self.compute_second_moment(y_pred)
        spread_true = self.compute_second_moment(y_true)
        
        moment_loss = F.mse_loss(spread_pred, spread_true, reduction=self.reduction)
        
        # 4. PEAK LOSS - Maximum value preservation
        peak_pred = y_pred.view(batch_size, -1).max(dim=1)[0]
        peak_true = y_true.view(batch_size, -1).max(dim=1)[0]
        
        peak_loss = F.mse_loss(peak_pred, peak_true, reduction=self.reduction)
        
        # Combine losses
        total_loss = (
            self.alpha_intensity * intensity_loss +
            self.alpha_position * position_loss +
            self.alpha_moment * moment_loss +
            self.alpha_peak * peak_loss
        )
        
        return total_loss


class MultiPeakLoss(nn.Module):
    """
    Loss function aware of multiple peaks in the spectra
    
    Useful when your 2D spectra contains multiple distinct signals
    (e.g., multiple wave systems from different directions)
    
    Uses local maxima detection to identify and match peaks
    
    Usage:
        criterion = MultiPeakLoss(
            num_peaks=3,           # Expected number of peaks
            base_loss='msle',      # Base reconstruction loss
            peak_weight=0.5,       # Weight for peak value matching
            position_weight=0.3    # Weight for peak position matching
        )
    """
    
    def __init__(self, 
                 num_peaks=5,
                 base_loss='msle',
                 peak_weight=0.5,
                 position_weight=0.3):
        super(MultiPeakLoss, self).__init__()
        
        self.num_peaks = num_peaks
        self.peak_weight = peak_weight
        self.position_weight = position_weight
        
        # Base reconstruction loss
        if base_loss == 'msle':
            self.base_loss = MSLELoss()
        elif base_loss == 'mse':
            self.base_loss = nn.MSELoss()
        else:
            self.base_loss = base_loss
    
    def detect_peaks(self, x, num_peaks=5):
        """
        Detect top N peaks in 2D spectra
        
        Args:
            x: [batch, 1, H, W]
            num_peaks: Number of peaks to detect
        
        Returns:
            peak_values: [batch, num_peaks]
            peak_positions: [batch, num_peaks, 2]
        """
        batch_size, _, H, W = x.shape
        
        # Flatten spatial dimensions
        x_flat = x.view(batch_size, -1)
        
        # Get top k values and indices
        top_values, top_indices = torch.topk(x_flat, num_peaks, dim=1)
        
        # Convert flat indices to 2D positions
        row_positions = top_indices // W
        col_positions = top_indices % W
        
        peak_positions = torch.stack([row_positions, col_positions], dim=-1)
        
        return top_values, peak_positions.float()
    
    def forward(self, y_pred, y_true):
        """Compute multi-peak aware loss"""
        # Base reconstruction loss
        base_loss = self.base_loss(y_pred, y_true)
        
        # Detect peaks
        peaks_pred_val, peaks_pred_pos = self.detect_peaks(y_pred, self.num_peaks)
        peaks_true_val, peaks_true_pos = self.detect_peaks(y_true, self.num_peaks)
        
        # Peak value matching
        peak_value_loss = F.mse_loss(peaks_pred_val, peaks_true_val)
        
        # Peak position matching
        peak_position_loss = F.mse_loss(peaks_pred_pos, peaks_true_pos)
        
        # Combine
        total_loss = (
            base_loss +
            self.peak_weight * peak_value_loss +
            self.position_weight * peak_position_loss
        )
        
        return total_loss


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
        'combined_intensity_position': CombinedIntensityPositionLoss,
        'multi_peak': MultiPeakLoss,
    }
    
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(losses.keys())}")
    
    return losses[loss_name](**kwargs)


class MSLEConstraintWeighted(nn.Module):
    """
    MSLE with weighting for sparse data - RECOMMENDED for your problem

    Args:
        alpha: Peak preservation weight (default: 0.15)
        beta: Energy conservation weight (default: 0.15)
        gamma: Weighted MSE weight (default: 0.02)
        nonzero_weight: Weight for non-zero values (default: 20.0)
        zero_weight: Weight for zero values (default: 1.0)
    """

    def __init__(self, alpha=0.15, beta=0.15, gamma=0.02,
                 nonzero_weight=20.0, zero_weight=1.0, epsilon=1e-8):
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

        # 4. Weighted MSE (20× focus on non-zero values)
        nonzero_mask = (y_true > self.epsilon).float()
        weights = nonzero_mask * self.nonzero_weight + (1 - nonzero_mask) * self.zero_weight
        weighted_mse = (weights * (y_pred - y_true) ** 2).mean()

        # Combine
        total_loss = msle_loss + self.alpha * peak_loss + self.beta * energy_loss + self.gamma * weighted_mse

        return total_loss


class LogScaleLoss(nn.Module):
    """Pure log-scale MSE - simpler alternative for sparse data"""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        log_pred = torch.log(y_pred + self.epsilon)
        log_true = torch.log(y_true + self.epsilon)
        return F.mse_loss(log_pred, log_true)