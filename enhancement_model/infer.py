"""
Inference script for audio enhancement
Loads trained model and enhances noisy audio
"""

import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse
import sys
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT.parent))

from enhancement_model.model import UNetAudioEnhancer


class ImprovedAudioEnhancer:
    """
    Audio enhancement with improved reconstruction
    """
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)
        
        # Audio settings
        self.sample_rate = 16000
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 128
        self.fmax = 8000
        
        # Load U-Net
        self.model = UNetAudioEnhancer(in_channels=1, out_channels=1)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 0) + 1}")
        print(f"   Val loss: {checkpoint.get('val_loss', 'N/A')}")
    
    def audio_to_spectrogram(self, audio):
        """Convert audio to mel-spectrogram in dB scale"""
        # Ensure audio is normalized
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()
        
        # Create mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=self.fmax,
            power=2.0  # Use power spectrogram
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Clip to reasonable range
        mel_spec_db = np.clip(mel_spec_db, -80.0, 0.0)
        
        return mel_spec_db
    
    def spectrogram_to_audio_improved(self, mel_spec_db, original_audio=None):
        """
        Improved spectrogram to audio conversion
        
        Args:
            mel_spec_db: Mel-spectrogram in dB scale
            original_audio: Original audio for phase estimation (optional)
        
        Returns:
            audio: Reconstructed audio
        """
        # Ensure valid dB range
        mel_spec_db = np.clip(mel_spec_db, -80.0, 0.0)
        mel_spec_db = np.nan_to_num(mel_spec_db, nan=-80.0, posinf=0.0, neginf=-80.0)
        
        # Convert to power
        mel_spec_power = librosa.db_to_power(mel_spec_db)
        mel_spec_power = np.maximum(mel_spec_power, 1e-10)
        
        # Method 1: High-quality Griffin-Lim with more iterations
        print("  Reconstructing audio (this may take a moment)...")
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec_power,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_iter=100,  # More iterations = better quality
            length=len(original_audio) if original_audio is not None else None
        )
        
        # Clean up audio
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize to prevent clipping
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.95  # Leave headroom
        
        return audio
    
    def enhance(self, audio):
        """
        Enhance audio using U-Net
        
        Args:
            audio: Input audio waveform
        
        Returns:
            enhanced_audio: Cleaned audio
        """
        original_length = len(audio)
        
        # Step 1: Convert to mel-spectrogram
        print("  Converting to mel-spectrogram...")
        noisy_spec_db = self.audio_to_spectrogram(audio)
        
        # Step 2: Normalize for CNN (matching training normalization)
        noisy_spec_norm = (noisy_spec_db + 80.0) / 80.0  # [0, 1]
        noisy_spec_norm = noisy_spec_norm * 2.0 - 1.0    # [-1, 1]
        
        # Step 3: Prepare tensor
        noisy_tensor = torch.FloatTensor(noisy_spec_norm).unsqueeze(0).unsqueeze(0)
        noisy_tensor = noisy_tensor.to(self.device)
        
        # Step 4: Run CNN
        print("  Running U-Net enhancement...")
        with torch.no_grad():
            clean_tensor = self.model(noisy_tensor)
            # Immediately handle NaN/Inf
            clean_tensor = torch.nan_to_num(clean_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            # Clamp to valid range
            clean_tensor = torch.clamp(clean_tensor, -1.0, 1.0)
        
        # Step 5: Denormalize
        clean_spec_norm = clean_tensor.squeeze().cpu().numpy()
        clean_spec_norm = (clean_spec_norm + 1.0) / 2.0     # [-1,1] → [0,1]
        clean_spec_db = clean_spec_norm * 80.0 - 80.0       # [0,1] → [-80,0] dB
        
        # Ensure valid range
        clean_spec_db = np.clip(clean_spec_db, -80.0, 0.0)
        clean_spec_db = np.nan_to_num(clean_spec_db, nan=-80.0)
        
        # Step 6: Convert back to audio with improved method
        enhanced_audio = self.spectrogram_to_audio_improved(
            clean_spec_db,
            original_audio=audio
        )
        
        # Ensure same length as input
        if len(enhanced_audio) > original_length:
            enhanced_audio = enhanced_audio[:original_length]
        elif len(enhanced_audio) < original_length:
            enhanced_audio = np.pad(enhanced_audio, (0, original_length - len(enhanced_audio)))
        
        return enhanced_audio
    
    def enhance_file(self, input_path, output_path, save_comparison=False):
        """
        Enhance a file and optionally save comparison
        
        Args:
            input_path: Path to noisy audio
            output_path: Path to save enhanced audio
            save_comparison: Save side-by-side comparison
        """
        print(f"\n{'='*60}")
        print(f"Processing: {input_path}")
        print(f"{'='*60}")
        
        # Load audio
        audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True)
        print(f"  Duration: {len(audio)/sr:.2f}s")
        print(f"  Samples: {len(audio)}")
        
        # Enhance
        enhanced_audio = self.enhance(audio)
        
        # Save enhanced
        sf.write(output_path, enhanced_audio, self.sample_rate)
        print(f"\n✅ Enhanced audio saved to: {output_path}")
        
        # Optionally save comparison
        if save_comparison:
            comparison_path = Path(output_path).parent / f"comparison_{Path(output_path).name}"
            
            # Create stereo file: left=original, right=enhanced
            stereo = np.stack([audio, enhanced_audio], axis=1)
            sf.write(comparison_path, stereo, self.sample_rate)
            print(f"📊 Comparison saved to: {comparison_path}")
            print("   (Left channel: original, Right channel: enhanced)")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Enhanced audio processing')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to noisy audio file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save enhanced audio')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use')
    parser.add_argument('--comparison', action='store_true',
                       help='Save side-by-side comparison')
    
    args = parser.parse_args()
    
    # Create enhancer
    enhancer = ImprovedAudioEnhancer(args.checkpoint, device=args.device)
    
    # Enhance file
    enhancer.enhance_file(args.input, args.output, save_comparison=args.comparison)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage:")
        print("  python improved_infer.py \\")
        print("    --checkpoint enhancement_model/checkpoints/best_model.pt \\")
        print("    --input data/audio_raw/noisy_0000.wav \\")
        print("    --output enhanced_output.wav \\")
        print("    --device cpu \\")
        print("    --comparison")
