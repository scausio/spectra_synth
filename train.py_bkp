"""
Improved Training Script for 2D Wave Spectra Reconstruction

This script provides enhanced training capabilities including:
- Better logging and monitoring
- Learning rate scheduling
- Early stopping
- Gradient clipping
- Model comparison utilities
"""
from sklearn.model_selection import train_test_split
from loss_functions import MSLELoss, MSLELossContraint, MSLELossContraintRescaled
from models_improved import get_model
import os
from reader import Reader,build_file_pairs,CreateDataset
from shaper import Ds_Conv
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
import logging
from utils import getDevice, init, setWorkingDirs
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import json
from datetime import datetime
from glob import glob

# Set seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Global config variable
config = None


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, min_delta=1e-6, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def batch_process(X_batch, y_batch, device, optimizer, model, criterion, is_training=True):
    """Process a single batch with optional gradient computation"""
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    
    if is_training:
        optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(X_batch)
    
    # Compute loss
    loss = criterion(y_pred, y_batch)
    
    if is_training:
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
    
    return loss.item()


def evaluate_model(model, val_loader, device, criterion):
    """
    Evaluate the model on a validation DataLoader.
    
    Args:
        model       : PyTorch model
        val_loader  : DataLoader yielding (X, Y) batches
        device      : torch.device
        criterion   : loss function (e.g., nn.MSELoss())
        
    Returns:
        metrics : dict containing loss, mse, mae, r2, peak_error, integral_error
    """
    model.eval()

    total_loss = 0.0
    total_mse  = 0.0
    total_mae  = 0.0

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for X, Y in val_loader:
            X = X.to(device)
            Y = Y.to(device)

            y_pred = model(X)  # shape: (B, 1, 24, 32)

            # ----- Pointwise losses -----
            total_loss += criterion(y_pred, Y).item()
            total_mse  += nn.MSELoss()(y_pred, Y).item()
            total_mae  += nn.L1Loss()(y_pred, Y).item()

            # store for global metrics
            y_true_all.append(Y)
            y_pred_all.append(y_pred)

    # ----- Concatenate all batches -----
    Y_true = torch.cat(y_true_all, dim=0)  # (N_total, 1, 24, 32)
    Y_pred = torch.cat(y_pred_all, dim=0)  # (N_total, 1, 24, 32)

    # ----- R2 Score -----
    y_true_flat = Y_true.cpu().numpy().ravel()
    y_pred_flat = Y_pred.cpu().numpy().ravel()
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ----- Peak preservation (mean max over batch) -----
    y_true_max = Y_true.view(Y_true.size(0), -1).max(dim=1)[0].mean().item()
    y_pred_max = Y_pred.view(Y_pred.size(0), -1).max(dim=1)[0].mean().item()
    peak_error = abs(y_true_max - y_pred_max) / y_true_max if y_true_max > 0 else 0.0

    # ----- Integral preservation (mean sum over batch) -----
    y_true_sum = Y_true.view(Y_true.size(0), -1).sum(dim=1).mean().item()
    y_pred_sum = Y_pred.view(Y_pred.size(0), -1).sum(dim=1).mean().item()
    integral_error = abs(y_true_sum - y_pred_sum) / y_true_sum if y_true_sum > 0 else 0.0

    # ----- Average pointwise metrics -----
    n_batches = len(val_loader)
    metrics = {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "mae": total_mae / n_batches,
        "r2": r2,
        "peak_error": peak_error,
        "integral_error": integral_error
    }

    return metrics



def plot_training_curves(train_history, val_history, save_path):
    """Plot comprehensive training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    metrics = ['loss', 'mse', 'mae', 'r2', 'peak_error', 'integral_error']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        if metric in train_history:
            ax.plot(train_history[metric], label='Train', alpha=0.7)
        if metric in val_history:
            ax.plot(val_history[metric], label='Validation', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_checkpoint(model, optimizer, epoch, metrics, paths, is_best=False):
    """Save model checkpoint with metadata"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(paths['checkpoint'], f'ckpt_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately
    if is_best:
        best_path = os.path.join(paths['checkpoint'], 'best_model.pt')
        torch.save(checkpoint, best_path)
        # Also save just the model for easy loading
        torch.save(model, os.path.join(paths['checkpoint'], 'best_model_full.pt'))


def train_epoch(model, train_loader, device, optimizer, criterion, epoch):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    num_batches = len(train_loader)
    
    with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
        for X_batch, y_batch in pbar:
            #plt.imshow(y_batch[0][0])
            #plt.show()
            #exit()
            loss = batch_process(X_batch, y_batch, device, optimizer, model, criterion, is_training=True)
            epoch_loss += loss
            pbar.set_postfix({'loss': f'{loss:.6f}'})
    
    avg_loss = epoch_loss / num_batches
    return avg_loss

def main(config_dict=None):
    """Main training function"""
    global config
    if config_dict is not None:
        config = config_dict
    
    start_time, logging = init()
    device = getDevice()
    paths = setWorkingDirs(config['outdir'])
    
    print(f"Using {device} device")
    logging.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # ========== Data Loading ==========
    logging.info("Loading dataset...")

    pairs = build_file_pairs(
        config["stats_path"],
        config["spc_path"],
        fname='*2025010*zarr'
       
    )
    train_pairs, tmp_pairs = train_test_split(
        pairs, test_size=0.3, random_state=42
    )
    val_pairs, test_pairs = train_test_split(
        tmp_pairs, test_size=0.5, random_state=42
    )

    ds_train = CreateDataset(
        train_pairs,
        Reader,
        config
    )
    ds_val = CreateDataset(
        val_pairs,
        Reader,
        config
    )
    # # Training dataset

    train_loader = DataLoader(
        ds_train,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4  # Increase if you have multiple CPUs
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4  # Increase if you have multiple CPUs
    )
    print(f"Dataset: {len(ds_train)} samples | "
      f"Batch size: {train_loader.batch_size} | "
      f"Batches/epoch: {len(train_loader)}")
    #print (ds_train.__getitem__(10))



    # ========== Model Setup ==========
    logging.info(f"Initializing model: {config['model_name']}")

    # Select loss function
    if config['loss_function'] == 'mse':
        criterion = nn.MSELoss()
    elif config['loss_function'] == 'msle':
        criterion = MSLELoss()
    elif config['loss_function'] == 'msle_constraint':
        criterion = MSLELossContraint(
            alpha=config.get('alpha', 0.0001),
            beta=config.get('beta', 0.0001),
            gamma=config.get('gamma', 0)
        )
    else:
        criterion = MSLELoss()
    #Initialize model
    print (config['model_name'])

    # Define spectra dimensions
    theta_bins = 24#ds_train.reader.theta_bins
    k_bins = 32#ds_train.reader.kappa_bins
    model = get_model(
        config['model_name'],
        9, # number of predictors
        (k_bins, theta_bins),
        **config.get('model_params', {})
    )
    #

    logging.info(f"Model: {model.__class__.__name__}")
    logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logging.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ========== Optimizer and Scheduler ==========
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    # Load checkpoint if resuming
    if config['init_epoch'] != 0:
        checkpoint_path = os.path.join(paths['checkpoint'], f'ckpt_epoch_{config["init_epoch"]}.pt')
        if os.path.exists(checkpoint_path):
            logging.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(
            latest_checkpoint,
            map_location=device,
            weights_only=False
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            logging.info(f"✓ Resumed from epoch {config['init_epoch']}")
        else:
            logging.warning(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
            config['init_epoch'] = 0

    # Auto-resume from last checkpoint if available
    elif config.get('auto_resume', False):
        # Find latest checkpoint
        checkpoint_files = glob(os.path.join(paths['checkpoint'], 'ckpt_epoch_*.pt'))
        if checkpoint_files:
            # Get the latest checkpoint
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            epoch_num = int(latest_checkpoint.split('_')[-1].split('.')[0])

            logging.info(f"Found existing checkpoint: {latest_checkpoint}")
            logging.info(f"Auto-resuming from epoch {epoch_num}")
            checkpoint = torch.load(
            latest_checkpoint,
            map_location=device,
            weights_only=False
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            config['init_epoch'] = epoch_num
            logging.info(f"✓ Auto-resumed from epoch {epoch_num}")

    model.to(device)
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience'),
        min_delta=1e-6
    )

    # ========== Training Loop ==========
    logging.info("Starting training...")

    best_vloss = float('inf')
    train_history = {'loss': [], 'mse': [], 'mae': [], 'r2': [], 'peak_error': [], 'integral_error': []}
    val_history = {'loss': [], 'mse': [], 'mae': [], 'r2': [], 'peak_error': [], 'integral_error': []}

    for epoch in range(config['init_epoch'], config['init_epoch'] + config['epochs']):
        # Train
        avg_tloss = train_epoch(model, train_loader, device, optimizer, criterion, epoch)
        train_history['loss'].append(avg_tloss)

        # Evaluate
        val_metrics = evaluate_model(model, val_loader, device, criterion)

        # Store metrics
        for key, value in val_metrics.items():
            val_history[key].append(value)

        # Log metrics
        logging.info(f'Epoch {epoch + 1}/{config["init_epoch"] + config["epochs"]}')
        logging.info(f'  Train Loss: {avg_tloss:.6f}')
        logging.info(f'  Val Loss: {val_metrics["loss"]:.6f}, MSE: {val_metrics["mse"]:.6f}, '
                     f'MAE: {val_metrics["mae"]:.6f}, R²: {val_metrics["r2"]:.4f}')
        logging.info(f'  Peak Error: {val_metrics["peak_error"]:.4f}, '
                     f'Integral Error: {val_metrics["integral_error"]:.4f}')

        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'  Learning Rate: {current_lr:.2e}')

        # Save best model
        is_best = val_metrics['loss'] < best_vloss
        if is_best:
            logging.info("  ✓ New best model!")
            best_vloss = val_metrics['loss']
            save_checkpoint(model, optimizer, epoch, val_metrics, paths, is_best=True)

        # Regular checkpoint
        if epoch % config['checkpoint_interval'] == 0 and epoch != 0:
            logging.info(f"  Saving checkpoint at epoch {epoch}")
            save_checkpoint(model, optimizer, epoch, val_metrics, paths, is_best=False)
            plot_training_curves(train_history, val_history,
                                 os.path.join(paths['checkpoint'], f"training_curves_{epoch}.png"))

        # Early stopping check
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break

 
    # ========== Final Saving ==========
    logging.info("Training completed!")
    
    # Save training history
    history = {
        'train': train_history,
        'val': val_history,
        'config': config,
        'best_epoch': val_history['loss'].index(min(val_history['loss'])),
        'best_val_loss': min(val_history['loss'])
    }
    
    with open(os.path.join(paths['checkpoint'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    np.save(os.path.join(paths['checkpoint'], "train_loss.npy"), np.array(train_history['loss']))
    np.save(os.path.join(paths['checkpoint'], "val_loss.npy"), np.array(val_history['loss']))
    
    # Final plots
    plot_training_curves(train_history, val_history, 
                        os.path.join(paths['checkpoint'], "final_training_curves.png"))
    
    # Training time
    end_time = datetime.now()
    duration = end_time - start_time
    logging.info(f"Total training time: {duration}")
    
    logging.info(f"Best validation loss: {min(val_history['loss']):.6f} "
                f"at epoch {val_history['loss'].index(min(val_history['loss']))}")

"""
if __name__ == "__main__":
    # ========== Configuration ==========
    # You can easily modify these parameters or load from a config file
    
    config = {
        # Training parameters
        'outdir': 'output_improved',
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'checkpoint_interval': 10,
        'early_stopping_patience': 15,
        'init_epoch': 0,
        'auto_resume': True,  # Automatically resume from latest checkpoint if available
        
        # Data parameters
        'mp': True,  # Multi-partition (wind waves + 2 swells)
        'wind': False,  # Include wind components
        'add_coords': False,  # Include lat/lon coordinates
        'decimate_input': 100,  # Decimation factor for input data
        
        # Model selection
        'model_name': 'hybrid',  # Options: 'attention_ffnn', 'transformer', 'unet', 'resnet', 'hybrid', 'lightweight'
        
        # Model-specific parameters (optional, will use defaults if not specified)
        'model_params': {
            'hidden_dim': 512,
            'num_res_blocks': 4,
            # Add other model-specific params as needed
        },
        
        # Loss function
        'loss_function': 'msle',  # Options: 'mse', 'msle', 'msle_constraint'
        'alpha': 0.0001,  # For constrained loss
        'beta': 0.0001,   # For constrained loss
        'gamma': 0,       # For constrained loss
        
        # Data paths
        'local_machine': False,
    }
    
    # Set data paths based on machine
    if config['local_machine']:
            if config['mp']:
                base = '/Users/scausio/CMCC Dropbox/Salvatore Causio/PycharmData/ML/MED/MP'
            else:
                base = '/Users/scausio/CMCC Dropbox/Salvatore Causio/PycharmData/ML/MED'
            config['stats_path'] = os.path.join(base, 'wave_stats_short.nc')
            config['spc_path'] = os.path.join(base, 'wave_spectra_short.nc')
    else:
        base = '/work/cmcc/ww3_cst-dev/work/ML/data/SSMEDdef_dataset_MP3'
        config['stats_path'] = os.path.join(base, 'wave_stats.nc')
        config['spc_path'] = os.path.join(base, 'wave_spectra.nc')

    # Run training
    main(config)
"""
