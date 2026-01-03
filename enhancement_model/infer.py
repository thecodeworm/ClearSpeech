"""
Enhanced Inference with Perceptual Optimization
Improves PESQ scores through better audio reconstruction
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


class PerceptualAudioEnhancer:
    """
    Audio enhancement with perceptual optimization for better PESQ scores
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
        """Convert audio to mel-spectrogram with better settings"""
        # Ensure proper normalization
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.95
        
        # Use power=2.0 for better reconstruction
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=self.fmax,
            power=2.0,  # Power spectrogram
            window='hann',  # Hann window for smoother transitions
            center=True,
            pad_mode='reflect'
        )
        
        # Convert to dB with stable reference
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)
        mel_spec_db = np.clip(mel_spec_db, -80.0, 0.0)
        
        return mel_spec_db
    
    def spectrogram_to_audio_phase_vocoder(self, mel_spec_db, original_audio):
        """
        IMPROVED: Use original audio phase information
        This significantly improves perceptual quality
        """
        # Ensure valid range
        mel_spec_db = np.clip(mel_spec_db, -80.0, 0.0)
        mel_spec_db = np.nan_to_num(mel_spec_db, nan=-80.0, posinf=0.0, neginf=-80.0)
        
        # Convert to power
        mel_spec_power = librosa.db_to_power(mel_spec_db)
        mel_spec_power = np.maximum(mel_spec_power, 1e-10)
        
        # Get STFT of original audio for phase information
        stft_original = librosa.stft(
            original_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann',
            center=True
        )
        
        # Get phase from original
        phase_original = np.angle(stft_original)
        
        # Convert mel to linear spectrogram
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmax=self.fmax
        )
        
        # Pseudo-inverse to convert mel back to linear
        mel_basis_inv = np.linalg.pinv(mel_basis)
        linear_spec = mel_basis_inv @ mel_spec_power
        
        # Ensure same shape as phase
        if linear_spec.shape[1] > phase_original.shape[1]:
            linear_spec = linear_spec[:, :phase_original.shape[1]]
        elif linear_spec.shape[1] < phase_original.shape[1]:
            phase_original = phase_original[:, :linear_spec.shape[1]]
        
        # Combine enhanced magnitude with original phase
        enhanced_stft = np.sqrt(linear_spec) * np.exp(1j * phase_original)
        
        # Inverse STFT
        audio = librosa.istft(
            enhanced_stft,
            hop_length=self.hop_length,
            window='hann',
            center=True,
            length=len(original_audio)
        )
        
        # Clean up
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize with headroom
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.95
        
        return audio
    
    def spectrogram_to_audio_iterative(self, mel_spec_db, n_iter=200):
        """
        Alternative: High-quality Griffin-Lim with more iterations
        """
        mel_spec_db = np.clip(mel_spec_db, -80.0, 0.0)
        mel_spec_db = np.nan_to_num(mel_spec_db, nan=-80.0)
        
        mel_spec_power = librosa.db_to_power(mel_spec_db)
        mel_spec_power = np.maximum(mel_spec_power, 1e-10)
        
        # Use more iterations for better quality
        print(f"  Reconstructing with {n_iter} iterations...")
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec_power,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_iter=n_iter,  # More iterations = better quality
            window='hann'
        )
        
        audio = np.nan_to_num(audio, nan=0.0)
        
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.95
        
        return audio
    
    def post_process_audio(self, audio):
        """
        Apply post-processing for better perceptual quality
        """
        # 1. De-emphasis filter (reverse pre-emphasis)
        # This restores natural frequency balance
        de_emphasis = 0.97
        audio_deemph = np.copy(audio)
        for i in range(1, len(audio)):
            audio_deemph[i] = audio[i] + de_emphasis * audio[i-1]
        
        # 2. Soft clipping to avoid harsh distortion
        audio_deemph = np.tanh(audio_deemph * 1.2) / 1.2
        
        # 3. Normalize
        if np.abs(audio_deemph).max() > 0:
            audio_deemph = audio_deemph / np.abs(audio_deemph).max() * 0.95
        
        return audio_deemph
    
    def enhance(self, audio, use_phase=True, post_process=True):
        """
        Enhance audio with perceptual optimization
        
        Args:
            audio: Input audio waveform
            use_phase: Use original phase information (better PESQ)
            post_process: Apply post-processing filters
        
        Returns:
            enhanced_audio: Cleaned audio
        """
        original_length = len(audio)
        original_audio = np.copy(audio)
        
        # Step 1: Convert to mel-spectrogram
        print("  Converting to mel-spectrogram...")
        noisy_spec_db = self.audio_to_spectrogram(audio)
        
        # Step 2: Normalize for CNN
        noisy_spec_norm = (noisy_spec_db + 80.0) / 80.0  # [0, 1]
        noisy_spec_norm = noisy_spec_norm * 2.0 - 1.0    # [-1, 1]
        
        # Step 3: Prepare tensor
        noisy_tensor = torch.FloatTensor(noisy_spec_norm).unsqueeze(0).unsqueeze(0)
        noisy_tensor = noisy_tensor.to(self.device)
        
        # Step 4: Run CNN
        print("  Running U-Net enhancement...")
        with torch.no_grad():
            clean_tensor = self.model(noisy_tensor)
            clean_tensor = torch.nan_to_num(clean_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            clean_tensor = torch.clamp(clean_tensor, -1.0, 1.0)
        
        # Step 5: Denormalize
        clean_spec_norm = clean_tensor.squeeze().cpu().numpy()
        clean_spec_norm = (clean_spec_norm + 1.0) / 2.0     # [-1,1] → [0,1]
        clean_spec_db = clean_spec_norm * 80.0 - 80.0       # [0,1] → [-80,0] dB
        
        clean_spec_db = np.clip(clean_spec_db, -80.0, 0.0)
        clean_spec_db = np.nan_to_num(clean_spec_db, nan=-80.0)
        
        # Step 6: Convert back to audio
        if use_phase:
            print("  Using phase vocoder reconstruction...")
            enhanced_audio = self.spectrogram_to_audio_phase_vocoder(
                clean_spec_db,
                original_audio
            )
        else:
            print("  Using iterative reconstruction...")
            enhanced_audio = self.spectrogram_to_audio_iterative(
                clean_spec_db,
                n_iter=200
            )
        
        # Step 7: Post-processing
        if post_process:
            print("  Applying post-processing...")
            enhanced_audio = self.post_process_audio(enhanced_audio)
        
        # Ensure same length
        if len(enhanced_audio) > original_length:
            enhanced_audio = enhanced_audio[:original_length]
        elif len(enhanced_audio) < original_length:
            enhanced_audio = np.pad(enhanced_audio, (0, original_length - len(enhanced_audio)))
        
        return enhanced_audio
    
    def enhance_file(self, input_path, output_path, use_phase=True, post_process=True):
        """
        Enhance a file with perceptual optimization
        """
        print(f"\n{'='*60}")
        print(f"Processing: {input_path}")
        print(f"{'='*60}")
        
        # Load audio
        audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True)
        print(f"  Duration: {len(audio)/sr:.2f}s")
        
        # Enhance
        enhanced_audio = self.enhance(audio, use_phase=use_phase, post_process=post_process)
        
        # Save
        sf.write(output_path, enhanced_audio, self.sample_rate)
        print(f"\n✅ Enhanced audio saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Perceptual audio enhancement')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to noisy audio file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save enhanced audio')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use')
    parser.add_argument('--no-phase', action='store_true',
                       help='Disable phase vocoder (use Griffin-Lim)')
    parser.add_argument('--no-postprocess', action='store_true',
                       help='Disable post-processing')
    
    args = parser.parse_args()
    
    enhancer = PerceptualAudioEnhancer(args.checkpoint, device=args.device)
    
    enhancer.enhance_file(
        args.input,
        args.output,
        use_phase=not args.no_phase,
        post_process=not args.no_postprocess
    )
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage:")
        print("  python perceptual_infer.py \\")
        print("    --checkpoint enhancement_model/checkpoints/best_model.pt \\")
        print("    --input noisy_audio.wav \\")
        print("    --output enhanced_audio.wav \\")
        print("    --device cpu")