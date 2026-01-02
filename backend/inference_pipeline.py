"""
Audio Enhancement and Transcription Pipeline
Handles real-time processing of user-uploaded audio
"""

import sys
from pathlib import Path
import io

# Project root = ClearSpeech
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhancement_model.model import UNetAudioEnhancer

import torch
import numpy as np
import librosa
import soundfile as sf
import whisper
from typing import Union, Dict, Tuple
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning)


def get_default_device() -> str:
    """Auto-detect best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class AudioProcessor:
    """
    Handles audio preprocessing (in-memory)
    Reuses logic from preprocessing.py but for single files
    """
    
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmax = 8000
    
    def load_audio(self, audio_file: Union[str, Path, bytes, io.BytesIO]) -> np.ndarray:
        """
        Load audio from file or bytes
        
        Args:
            audio_file: File path, file object, or bytes
        
        Returns:
            audio: Numpy array of audio samples
        """
        try:
            if isinstance(audio_file, (str, Path)):
                audio, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
            elif isinstance(audio_file, bytes):
                audio, _ = librosa.load(io.BytesIO(audio_file), sr=self.sample_rate, mono=True)
            else:
                audio, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
            return audio
        except Exception as e:
            raise ValueError(f"Failed to load audio: {e}")
    
    def normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio with RMS-based approach for consistency
        
        Args:
            audio: Input audio
            target_db: Target RMS level in dB
        
        Returns:
            Normalized audio
        """
        # RMS-based normalization (better than peak normalization)
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 10 ** (target_db / 20)
            audio = audio * (target_rms / rms)
        
        # Clip to prevent distortion
        audio = np.clip(audio, -1.0, 1.0)
        return audio
    
    def audio_to_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to mel-spectrogram
        
        Args:
            audio: Audio waveform
        
        Returns:
            mel_spec_db: Mel-spectrogram in dB scale
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=self.fmax
        )
        
        # Convert to dB with proper reference
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Ensure valid range
        mel_spec_db = np.clip(mel_spec_db, -80.0, 0.0)
        return mel_spec_db
    
    def spectrogram_to_audio(self, mel_spec_db: np.ndarray, n_iter: int = 60) -> np.ndarray:
        """
        Convert mel-spectrogram back to audio using Griffin-Lim
        
        Args:
            mel_spec_db: Mel-spectrogram in dB
            n_iter: Griffin-Lim iterations (more = better quality)
        
        Returns:
            audio: Reconstructed waveform
        """
        # Ensure valid dB range
        mel_spec_db = np.clip(mel_spec_db, -80.0, 0.0)
        mel_spec_db = np.nan_to_num(mel_spec_db, nan=-80.0, posinf=0.0, neginf=-80.0)
        
        # Convert from dB to power
        mel_spec = librosa.db_to_power(mel_spec_db)
        
        # Ensure non-negative power values
        mel_spec = np.maximum(mel_spec, 1e-10)
        
        # Convert to audio using Griffin-Lim with more iterations
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_iter=n_iter
        )
        
        # Handle any NaN or Inf in audio
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return audio


class EnhancementPipeline:
    """
    Complete audio enhancement and transcription pipeline
    """
    
    def __init__(
        self, 
        cnn_checkpoint_path: str, 
        whisper_model_name: str = "base", 
        device: str = None,
        use_fp16: bool = False
    ):
        """
        Initialize the pipeline with models
        
        Args:
            cnn_checkpoint_path: Path to trained CNN model
            whisper_model_name: Whisper model size (tiny, base, small, medium, large)
            device: 'cuda', 'mps', or 'cpu'
            use_fp16: Use half precision for Whisper (faster on GPU)
        """
        if device is None:
            device = get_default_device()
        self.device = torch.device(device)
        self.use_fp16 = use_fp16 and (device == "cuda")
        
        print(f"🖥️  Using device: {self.device}")
        
        self.audio_processor = AudioProcessor()
        
        # Load CNN enhancement model
        print(f"📥 Loading U-Net enhancement model...")
        self.cnn_model = UNetAudioEnhancer(in_channels=1, out_channels=1)
        
        try:
            checkpoint = torch.load(cnn_checkpoint_path, map_location=self.device)
            self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.cnn_model.to(self.device)
            self.cnn_model.eval()
            
            epoch = checkpoint.get('epoch', 'unknown')
            val_loss = checkpoint.get('val_loss', 'unknown')
            print(f"✅ U-Net loaded (epoch {epoch}, val_loss: {val_loss})")
        except Exception as e:
            raise RuntimeError(f"Failed to load CNN model: {e}")
        
        # Load Whisper model
        print(f"📥 Loading Whisper model ({whisper_model_name})...")
        try:
            self.whisper_model = whisper.load_model(whisper_model_name, device=str(self.device))
            print("✅ Whisper model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance audio using U-Net model
        
        Args:
            audio: Raw audio waveform
        
        Returns:
            enhanced_audio: Cleaned audio waveform
        """
        # Convert to spectrogram (dB scale: [-80, 0])
        noisy_spec = self.audio_processor.audio_to_spectrogram(audio)
        
        # Normalize to [-1, 1] (matching training normalization)
        noisy_spec_norm = (noisy_spec + 80.0) / 80.0    # [0, 1]
        noisy_spec_norm = noisy_spec_norm * 2.0 - 1.0   # [-1, 1]
        
        # Add batch and channel dimensions: (1, 1, H, W)
        noisy_spec_tensor = torch.FloatTensor(noisy_spec_norm).unsqueeze(0).unsqueeze(0)
        noisy_spec_tensor = noisy_spec_tensor.to(self.device)
        
        # Run U-Net inference
        with torch.no_grad():
            clean_spec_tensor = self.cnn_model(noisy_spec_tensor)
            # Handle NaN/Inf immediately after model output
            clean_spec_tensor = torch.nan_to_num(clean_spec_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            clean_spec_tensor = torch.clamp(clean_spec_tensor, -1.0, 1.0)
        
        # Convert back to numpy
        clean_spec_norm = clean_spec_tensor.squeeze().cpu().numpy()
        
        # Denormalize: [-1, 1] → [0, 1] → [-80, 0] dB
        clean_spec_norm = (clean_spec_norm + 1.0) / 2.0       # [-1,1] → [0,1]
        clean_spec_db = clean_spec_norm * 80.0 - 80.0         # [0,1] → [-80,0]
        
        # Ensure valid dB range
        clean_spec_db = np.nan_to_num(clean_spec_db, nan=-80.0, posinf=0.0, neginf=-80.0)
        clean_spec_db = np.clip(clean_spec_db, -80.0, 0.0)
        
        # Convert spectrogram to audio (more iterations for better quality)
        enhanced_audio = self.audio_processor.spectrogram_to_audio(clean_spec_db, n_iter=60)
        
        # Normalize and clip
        enhanced_audio = self.audio_processor.normalize_audio(enhanced_audio)
        enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
        
        return enhanced_audio
    
    def transcribe_audio(self, audio: np.ndarray, language: str = 'en') -> Dict:
        """
        Transcribe audio using Whisper
        
        Args:
            audio: Audio waveform (numpy array)
            language: Language code (e.g., 'en', 'es', 'fr')
        
        Returns:
            result: Dictionary with transcription and metadata
        """
        # Whisper expects float32 audio normalized to [-1, 1]
        audio = audio.astype(np.float32)
        
        # Pad or trim to 30 seconds max for efficiency
        max_length = 30 * self.audio_processor.sample_rate
        if len(audio) > max_length:
            print(f"⚠️  Audio longer than 30s, processing in chunks...")
        
        result = self.whisper_model.transcribe(
            audio,
            language=language if language else None,
            fp16=self.use_fp16,
            verbose=False
        )
        return result
    
    def process(
        self, 
        audio_file: Union[str, Path, bytes, io.BytesIO],
        language: str = 'en',
        skip_enhancement: bool = False
    ) -> Dict:
        """
        Complete processing pipeline
        
        Args:
            audio_file: Input audio (file path, bytes, or file object)
            language: Target language for transcription
            skip_enhancement: Skip enhancement step (use original audio)
        
        Returns:
            result: Dictionary containing:
                - transcript: Text transcription
                - enhanced_audio: Cleaned audio (numpy array)
                - duration: Audio duration in seconds
                - language: Detected language
                - segments: Timestamped segments
        """
        # Load and preprocess
        print("🎵 Loading audio...")
        audio = self.audio_processor.load_audio(audio_file)
        audio = self.audio_processor.normalize_audio(audio)
        
        duration = len(audio) / self.audio_processor.sample_rate
        print(f"   Duration: {duration:.2f}s")
        
        # Enhance with U-Net
        if not skip_enhancement:
            print("🧹 Enhancing audio with U-Net...")
            enhanced_audio = self.enhance_audio(audio)
        else:
            print("⏭️  Skipping enhancement...")
            enhanced_audio = audio
        
        # Transcribe with Whisper
        print("📝 Transcribing with Whisper...")
        transcription_result = self.transcribe_audio(enhanced_audio, language=language)
        
        # Compile results
        result = {
            'transcript': transcription_result['text'].strip(),
            'enhanced_audio': enhanced_audio,
            'sample_rate': self.audio_processor.sample_rate,
            'duration': duration,
            'language': transcription_result.get('language', language),
            'segments': transcription_result.get('segments', [])
        }
        
        print("✅ Processing complete!")
        return result


def test_pipeline():
    """Test the pipeline with a sample audio file"""
    print("="*70)
    print("🧪 TESTING AUDIO ENHANCEMENT PIPELINE")
    print("="*70)
    
    # Paths
    cnn_checkpoint = PROJECT_ROOT / "enhancement_model/checkpoints/best_model.pt"
    test_audio = PROJECT_ROOT / "data/audio_raw/noisy_0000.wav"
    output_audio = PROJECT_ROOT / "enhanced_test_output.wav"
    
    if not test_audio.exists():
        print(f"❌ Test audio not found: {test_audio}")
        return
    
    # Initialize pipeline
    pipeline = EnhancementPipeline(
        cnn_checkpoint_path=str(cnn_checkpoint),
        whisper_model_name="base",
        device=get_default_device()
    )
    
    # Process audio
    result = pipeline.process(test_audio)
    
    # Print results
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"Transcript: {result['transcript']}")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Language: {result['language']}")
    print(f"Segments: {len(result.get('segments', []))}")
    
    # Save enhanced audio
    sf.write(output_audio, result['enhanced_audio'], result['sample_rate'])
    print(f"\n💾 Enhanced audio saved to: {output_audio}")
    print("="*70)


if __name__ == "__main__":
    test_pipeline()
