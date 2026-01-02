"""
Training script for U-Net Audio Enhancement Model
Trains on mel-spectrogram pairs (noisy -> clean)
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


class Config:
    """Training configuration"""
    # Paths
    METADATA_PATH = "../data/metadata/metadata.json"
    CHECKPOINT_DIR = Path("./checkpoints")
    LOG_DIR = Path("./logs")
    
    # Training hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    
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
    SAVE_EVERY = 5  # Save checkpoint every N epochs
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for noisy, clean in pbar:
        # Move to device
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate the model
    
    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
        for noisy, clean in pbar:
            # Move to device
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Forward pass
            output = model(noisy)
            loss = criterion(output, clean)
            
            # Track loss
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


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
            'in_channels': config.IN_CHANNELS,
            'out_channels': config.OUT_CHANNELS
        }
    }
    
    filepath = config.CHECKPOINT_DIR / filename
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']


def train(config):
    """Main training loop"""
    
    print("="*70)
    print("TRAINING U-NET AUDIO ENHANCEMENT MODEL")
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
    
    # Loss and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # TensorBoard
    writer = SummaryWriter(config.LOG_DIR)
    
    # Training loop
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    print("="*70)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   Time: {epoch_time:.2f}s")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                config, 'best_model.pt'
            )
            print(f"    New best model!")
        
        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                config, f'checkpoint_epoch_{epoch+1}.pt'
            )
        
        print("-"*70)
    
    # Save final model
    save_checkpoint(
        model, optimizer, config.NUM_EPOCHS-1, train_loss, val_loss,
        config, 'final_model.pt'
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
