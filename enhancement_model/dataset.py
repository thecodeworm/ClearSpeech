"""
PyTorch Dataset for loading mel-spectrogram pairs
Loads noisy and clean spectrograms for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path


class SpectrogramDataset(Dataset):
    """
    Dataset for loading noisy/clean spectrogram pairs
    
    Args:
        metadata_path: Path to metadata.json file
        split: 'train' or 'val'
        transform: Optional transform to apply to spectrograms
    """
    def __init__(self, metadata_path, split='train', transform=None):
        self.metadata_path = Path(metadata_path)
        self.split = split
        self.transform = transform
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        # Get samples for this split
        self.samples = data[split]
        self.config = data['config']
        
        print(f"📁 Loaded {len(self.samples)} {split} samples")
        print(f"   Sample rate: {self.config['sample_rate']} Hz")
        print(f"   Duration: {self.config['duration']} sec")
        print(f"   Mel bins: {self.config['n_mels']}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample
    
        Returns:
            noisy_spec: Noisy spectrogram tensor (1, H, W)
            clean_spec: Clean spectrogram tensor (1, H, W)
        """
        sample = self.samples[idx]
    
        # Load spectrograms
        noisy_spec = np.load(sample['noisy_spec'])
        clean_spec = np.load(sample['clean_spec'])
    
        # Assuming mels are in [-80, 0] dB
        noisy_spec = (noisy_spec + 80.0) / 80.0   # [0,1]
        clean_spec = (clean_spec + 80.0) / 80.0   # [0,1]
    
        noisy_spec = noisy_spec * 2.0 - 1.0       # [-1,1]
        clean_spec = clean_spec * 2.0 - 1.0       # [-1,1]
    
        # Convert to tensors and add channel dimension
        noisy_spec = torch.FloatTensor(noisy_spec).unsqueeze(0)  # (1, H, W)
        clean_spec = torch.FloatTensor(clean_spec).unsqueeze(0)  # (1, H, W)
    
        # Apply transforms if any
        if self.transform:
            noisy_spec = self.transform(noisy_spec)
            clean_spec = self.transform(clean_spec)
    
        return noisy_spec, clean_spec



def get_dataloaders(metadata_path, batch_size=16, num_workers=4):
    """
    Create train and validation dataloaders
    
    Args:
        metadata_path: Path to metadata.json
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    # Create datasets
    train_dataset = SpectrogramDataset(metadata_path, split='train')
    val_dataset = SpectrogramDataset(metadata_path, split='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataloaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def test_dataset():
    """Test the dataset loading"""
    print("="*70)
    print("Testing Spectrogram Dataset")
    print("="*70)
    
    # Path to metadata (adjust if needed)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    metadata_path = PROJECT_ROOT / "data" / "metadata" / "metadata.json"

    
    # Create dataset
    dataset = SpectrogramDataset(metadata_path, split='train')
    
    # Load a sample
    noisy, clean = dataset[0]
    
    print(f"\nSample shapes:")
    print(f"   Noisy: {noisy.shape}")
    print(f"   Clean: {clean.shape}")
    
    print(f"\nValue ranges:")
    print(f"   Noisy: [{noisy.min():.2f}, {noisy.max():.2f}]")
    print(f"   Clean: [{clean.min():.2f}, {clean.max():.2f}]")
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(metadata_path, batch_size=8)
    
    # Test a batch
    noisy_batch, clean_batch = next(iter(train_loader))
    print(f"\n🎯 Batch shapes:")
    print(f"   Noisy batch: {noisy_batch.shape}")
    print(f"   Clean batch: {clean_batch.shape}")
    
    print("\nDataset test passed!")
    print("="*70)


if __name__ == "__main__":
    test_dataset()
