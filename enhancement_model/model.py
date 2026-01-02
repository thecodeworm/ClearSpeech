"""
U-Net Autoencoder for Speech Enhancement
Processes mel-spectrograms to remove noise

Architecture: Encoder-Decoder with skip connections
Input: Noisy mel-spectrogram (128 x T)
Output: Clean mel-spectrogram (128 x T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    Used as basic building block in U-Net
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block: MaxPool -> DoubleConv
    Reduces spatial dimensions, increases channels
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block: Upsample -> Concat with skip connection -> DoubleConv
    Increases spatial dimensions, decreases channels
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map from decoder path
            x2: Feature map from encoder path (skip connection)
        """
        x1 = self.up(x1)
        
        # Handle size mismatch due to padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetAudioEnhancer(nn.Module):
    """
    U-Net model for audio enhancement
    
    Architecture:
        Encoder: 4 downsampling stages (64 -> 128 -> 256 -> 512)
        Bottleneck: 1024 channels
        Decoder: 4 upsampling stages (512 -> 256 -> 128 -> 64)
        Output: 1 channel (clean spectrogram)
    
    Args:
        in_channels: Number of input channels (1 for single spectrogram)
        out_channels: Number of output channels (1 for single spectrogram)
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, 64)
        
        # Encoder (downsampling path)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder (upsampling path)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # Output convolution
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through U-Net
        
        Args:
            x: Input tensor (batch_size, 1, height, width)
               For mel-spectrograms: (B, 1, 128, T)
        
        Returns:
            Output tensor (batch_size, 1, height, width)
        """
        # Encoder path (save features for skip connections)
        x1 = self.inc(x)      # 64 channels
        x2 = self.down1(x1)   # 128 channels
        x3 = self.down2(x2)   # 256 channels
        x4 = self.down3(x3)   # 512 channels
        x5 = self.down4(x4)   # 1024 channels (bottleneck)
        
        # Decoder path (with skip connections)
        x = self.up1(x5, x4)  # 512 channels
        x = self.up2(x, x3)   # 256 channels
        x = self.up3(x, x2)   # 128 channels
        x = self.up4(x, x1)   # 64 channels
        
        # Output
        x = self.outc(x)      # 1 channel
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """
    Test the model with dummy input
    Verifies input/output dimensions
    """
    print("="*70)
    print("Testing U-Net Audio Enhancer Model")
    print("="*70)
    
    # Create model
    model = UNetAudioEnhancer(in_channels=1, out_channels=1)
    
    # Print model info
    print(f"\nModel Parameters: {model.count_parameters():,}")
    print(f"   (~{model.count_parameters() / 1e6:.2f}M parameters)")
    
    # Test with dummy input
    # Mel-spectrogram size: (batch, channels, mels, time)
    # Time frames = (3 seconds * 16000 Hz) / 256 hop_length = 187.5 ≈ 188
    batch_size = 4
    mel_bins = 128
    time_frames = 188
    
    dummy_input = torch.randn(batch_size, 1, mel_bins, time_frames)
    print(f"\n🔍 Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    
    # Verify shapes match
    assert output.shape == dummy_input.shape, "Output shape mismatch!"
    print("\nModel test passed!")
    print("="*70)


if __name__ == "__main__":
    test_model()
