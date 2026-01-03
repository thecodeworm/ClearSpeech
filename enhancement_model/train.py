"""
FIXED Training Script for Speech Enhancement
Addresses the issue where model makes audio worse

Key fixes:
1. L1 loss instead of pure MSE (better for audio)
2. Lower learning rate for stability
3. Gradient clipping to prevent explosions
4. Better normalization checks
5. Progressive training strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import time
import json

from model import UNetAudioEnhancer
from dataset import get_dataloaders


class ImprovedLoss(nn.Module):
    """
    Improved loss function combining MSE and L1
    L1 is better for preserving audio quality
    """
    def __init__(self, l1_weight=0.7, mse_weight=0.3):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        total = self.l1_weight * l1 + self.mse_weight * mse
        return total, {'l1': l1.item(), 'mse': mse.item()}


class Config:
    """Fixed training configuration"""
    # Paths
    METADATA_PATH = "../data/metadata/metadata.json"
    CHECKPOINT_DIR = Path("./checkpoints")
    LOG_DIR = Path("./logs")
    
    # Training hyperparameters - FIXED VALUES
    BATCH_SIZE = 8  # Smaller batch for stability
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4  # Lower learning rate
    NUM_WORKERS = 4
    
    # Loss weights - Favor L1 over MSE
    L1_WEIGHT = 0.7
    MSE_WEIGHT = 0.3
    
    # Model
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    
    # Device
    if torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    elif torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    
    # Checkpointing
    SAVE_EVERY = 5
    
    # Training stability
    GRADIENT_CLIP = 1.0  # Clip gradients
    WARMUP_EPOCHS = 5  # Warmup learning rate
    
    @classmethod
    def create_directories(cls):
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_lr_scheduler(optimizer, config):
    """
    Learning rate scheduler with warmup
    Starts low and gradually increases
    """
    def lr_lambda(epoch):
        if epoch < config.WARMUP_EPOCHS:
            # Warmup: linearly increase from 0.1x to 1x
            return 0.1 + 0.9 * (epoch / config.WARMUP_EPOCHS)
        else:
            # After warmup: cosine decay
            progress = (epoch - config.WARMUP_EPOCHS) / (config.NUM_EPOCHS - config.WARMUP_EPOCHS)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, config):
    """Train for one epoch with improved stability"""
    model.train()
    total_loss = 0
    total_l1 = 0
    total_mse = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, (noisy, clean) in enumerate(pbar):
        # Move to device
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        # Check for NaN in input
        if torch.isnan(noisy).any() or torch.isnan(clean).any():
            print(f"Warning: NaN in input at batch {batch_idx}")
            continue
        
        # Forward pass
        optimizer.zero_grad()
        output = model(noisy)
        
        # Check for NaN in output
        if torch.isnan(output).any():
            print(f"Warning: NaN in model output at batch {batch_idx}")
            continue
        
        # Compute loss
        loss, loss_components = criterion(output, clean)
        
        # Check if loss is valid
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss at batch {batch_idx}: {loss.item()}")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_l1 += loss_components['l1']
        total_mse += loss_components['mse']
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'l1': f'{loss_components["l1"]:.4f}',
            'mse': f'{loss_components["mse"]:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_l1 = total_l1 / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    
    return avg_loss, avg_l1, avg_mse


def validate(model, dataloader, criterion, device, epoch):
    """Validation with stability checks"""
    model.eval()
    total_loss = 0
    total_l1 = 0
    total_mse = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
        for noisy, clean in pbar:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Forward pass
            output = model(noisy)
            
            # Check for NaN
            if torch.isnan(output).any():
                print(f"Warning: NaN in validation output")
                continue
            
            # Compute loss
            loss, loss_components = criterion(output, clean)
            
            total_loss += loss.item()
            total_l1 += loss_components['l1']
            total_mse += loss_components['mse']
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'l1': f'{loss_components["l1"]:.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_l1 = total_l1 / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    
    return avg_loss, avg_l1, avg_mse


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'l1_weight': config.L1_WEIGHT,
            'mse_weight': config.MSE_WEIGHT
        }
    }
    
    filepath = config.CHECKPOINT_DIR / filename
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def train(config):
    """Main training loop with all fixes"""
    
    print("="*70)
    print("FIXED TRAINING - AUDIO ENHANCEMENT MODEL")
    print("="*70)
    print("\nKey improvements:")
    print("  • L1 loss (70%) + MSE loss (30%)")
    print("  • Lower learning rate with warmup")
    print("  • Gradient clipping")
    print("  • NaN detection and handling")
    print("="*70)
    
    # Setup
    config.create_directories()
    device = config.DEVICE
    print(f"\nDevice: {device}")
    
    # Create dataloaders
    print("\n📁 Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        config.METADATA_PATH,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # Create model
    print("\nCreating model...")
    model = UNetAudioEnhancer(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS
    ).to(device)
    
    print(f"   Parameters: {model.count_parameters():,}")
    
    # Loss function - L1 + MSE
    criterion = ImprovedLoss(
        l1_weight=config.L1_WEIGHT,
        mse_weight=config.MSE_WEIGHT
    )
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup
    scheduler = get_lr_scheduler(optimizer, config)
    
    # TensorBoard
    writer = SummaryWriter(config.LOG_DIR)
    
    # Training loop
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    print(f"Warmup: first {config.WARMUP_EPOCHS} epochs")
    print("="*70)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss, train_l1, train_mse = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # Validate
        val_loss, val_l1, val_mse = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train_total', train_loss, epoch)
        writer.add_scalar('Loss/train_l1', train_l1, epoch)
        writer.add_scalar('Loss/train_mse', train_mse, epoch)
        writer.add_scalar('Loss/val_total', val_loss, epoch)
        writer.add_scalar('Loss/val_l1', val_l1, epoch)
        writer.add_scalar('Loss/val_mse', val_mse, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"   Train Loss: {train_loss:.4f} (L1: {train_l1:.4f}, MSE: {train_mse:.4f})")
        print(f"   Val Loss:   {val_loss:.4f} (L1: {val_l1:.4f}, MSE: {val_mse:.4f})")
        print(f"   Time: {epoch_time:.2f}s")
        print(f"   LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                config, 'best_model_fixed.pt'
            )
            print(f"   🎯 New best model! (val_loss: {val_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                config, f'checkpoint_fixed_epoch_{epoch+1}.pt'
            )
        
        print("-"*70)
    
    # Save final model
    save_checkpoint(
        model, optimizer, config.NUM_EPOCHS-1, train_loss, val_loss,
        config, 'final_model_fixed.pt'
    )
    
    writer.close()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Checkpoints saved to: {config.CHECKPOINT_DIR}")
    print(f"TensorBoard logs: {config.LOG_DIR}")
    print(f"   View with: tensorboard --logdir={config.LOG_DIR}")
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print("="*70)


if __name__ == "__main__":
    config = Config()
    train(config)