import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for preprocessing"""
    
    # Dataset settings
    DATASET_NAME = "JacobLinCool/VoiceBank-DEMAND-16k"
    NUM_SAMPLES = 1500  # Using subset for cost efficiency
    TRAIN_SPLIT = 0.85  # 85% train, 15% validation
    
    # Audio processing parameters
    SAMPLE_RATE = 16000  # Whisper expects 16kHz
    DURATION = 3.0  # seconds - trim/pad to fixed length
    N_FFT = 1024  # FFT window size
    HOP_LENGTH = 256  # Number of samples between frames
    N_MELS = 128  # Number of mel bands
    
    # Output paths
    OUTPUT_DIR = Path("data/processed")
    RAW_AUDIO_DIR = OUTPUT_DIR / "audio_raw"
    CLEAN_AUDIO_DIR = OUTPUT_DIR / "audio_clean"
    NOISY_SPEC_DIR = OUTPUT_DIR / "spectrograms/noisy"
    CLEAN_SPEC_DIR = OUTPUT_DIR / "spectrograms/clean"
    METADATA_FILE = OUTPUT_DIR / "metadata.json"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary output directories"""
        for directory in [cls.RAW_AUDIO_DIR, cls.CLEAN_AUDIO_DIR, 
                         cls.NOISY_SPEC_DIR, cls.CLEAN_SPEC_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# AUDIO PROCESSING FUNCTIONS
# ============================================================================

def load_and_resample(audio_path, target_sr=16000):
    """
    Load audio file and resample to target sample rate
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (16kHz for Whisper)
    
    Returns:
        audio: Numpy array of audio samples
        sr: Sample rate
    """
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio, sr


def normalize_audio(audio):
    """
    Normalize audio to [-1, 1] range
    Prevents clipping and standardizes volume
    
    Args:
        audio: Raw audio array
    
    Returns:
        Normalized audio array
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def trim_or_pad(audio, target_length, sr=16000):
    """
    Trim or pad audio to fixed length
    Ensures all samples have same duration for batching
    
    Args:
        audio: Audio array
        target_length: Target duration in seconds
        sr: Sample rate
    
    Returns:
        Audio array of exact target length
    """
    target_samples = int(target_length * sr)
    
    if len(audio) > target_samples:
        # Trim from center to preserve important content
        start = (len(audio) - target_samples) // 2
        audio = audio[start:start + target_samples]
    elif len(audio) < target_samples:
        # Pad with zeros
        padding = target_samples - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    
    return audio


def audio_to_mel_spectrogram(audio, sr=16000, n_fft=1024, 
                             hop_length=256, n_mels=128):
    """
    Convert audio waveform to mel-spectrogram
    This is what the CNN will process
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length between frames
        n_mels: Number of mel frequency bands
    
    Returns:
        mel_spec: Mel-spectrogram in dB scale (2D array)
    """
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=8000  # Maximum frequency (half of 16kHz)
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def mel_spectrogram_to_audio(mel_spec_db, sr=16000, n_fft=1024, 
                             hop_length=256, n_iter=32):
    """
    Convert mel-spectrogram back to audio (for CNN output)
    Uses Griffin-Lim algorithm
    
    Args:
        mel_spec_db: Mel-spectrogram in dB
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_iter: Number of Griffin-Lim iterations
    
    Returns:
        audio: Reconstructed audio waveform
    """
    # Convert from dB back to power
    mel_spec = librosa.db_to_power(mel_spec_db)
    
    # Inverse mel-spectrogram to audio
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spec,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=n_iter
    )
    
    return audio


# ============================================================================
# DATASET PROCESSING
# ============================================================================

def load_voicebank_dataset(num_samples=1500):
    """
    Load VoiceBank-DEMAND-16k dataset from HuggingFace
    This version is already resampled to 16kHz
    
    Args:
        num_samples: Number of samples to load
    
    Returns:
        dataset: HuggingFace dataset with train split
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")
    
    print(f"Loading VoiceBank-DEMAND-16k dataset (first {num_samples} samples)...")
    
    # Load dataset - this version is pre-processed at 16kHz
    dataset = load_dataset(
        "JacobLinCool/VoiceBank-DEMAND-16k",
        split=f"train[:{num_samples}]"
    )
    
    print(f"Loaded {len(dataset)} samples")
    print(f"Dataset features: {dataset.features}")
    return dataset


def process_single_sample(sample, idx, config):
    """
    Process a single audio pair (noisy + clean)
    
    Args:
        sample: Dataset sample with 'noisy' and 'clean' audio
        idx: Sample index
        config: Configuration object
    
    Returns:
        metadata: Dictionary with processing info
    """
    try:
        # Extract audio arrays from dataset
        # JacobLinCool dataset structure: sample has 'noisy' and 'clean' keys
        # Each contains 'array' (waveform) and 'sampling_rate'
        noisy_audio = np.array(sample['noisy']['array'])
        clean_audio = np.array(sample['clean']['array'])
        
        # Get sample rate (should already be 16kHz)
        sr = sample['noisy']['sampling_rate']
        
        # Verify sample rate (dataset should already be 16kHz)
        if sr != config.SAMPLE_RATE:
            print(f"Sample {idx}: Expected {config.SAMPLE_RATE}Hz, got {sr}Hz. Resampling...")
            noisy_audio = librosa.resample(
                noisy_audio, orig_sr=sr, target_sr=config.SAMPLE_RATE
            )
            clean_audio = librosa.resample(
                clean_audio, orig_sr=sr, target_sr=config.SAMPLE_RATE
            )
        
        # Normalize
        noisy_audio = normalize_audio(noisy_audio)
        clean_audio = normalize_audio(clean_audio)
        
        # Trim/pad to fixed length
        noisy_audio = trim_or_pad(noisy_audio, config.DURATION, config.SAMPLE_RATE)
        clean_audio = trim_or_pad(clean_audio, config.DURATION, config.SAMPLE_RATE)
        
        # Save raw audio files (for Whisper later)
        noisy_path = config.RAW_AUDIO_DIR / f"noisy_{idx:04d}.wav"
        clean_path = config.CLEAN_AUDIO_DIR / f"clean_{idx:04d}.wav"
        
        sf.write(noisy_path, noisy_audio, config.SAMPLE_RATE)
        sf.write(clean_path, clean_audio, config.SAMPLE_RATE)
        
        # Convert to mel-spectrograms (for CNN)
        noisy_spec = audio_to_mel_spectrogram(
            noisy_audio, 
            sr=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS
        )
        
        clean_spec = audio_to_mel_spectrogram(
            clean_audio,
            sr=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS
        )
        
        # Save spectrograms as numpy arrays
        noisy_spec_path = config.NOISY_SPEC_DIR / f"noisy_{idx:04d}.npy"
        clean_spec_path = config.CLEAN_SPEC_DIR / f"clean_{idx:04d}.npy"
        
        np.save(noisy_spec_path, noisy_spec)
        np.save(clean_spec_path, clean_spec)
        
        # Return metadata
        return {
            'idx': idx,
            'noisy_audio': str(noisy_path),
            'clean_audio': str(clean_path),
            'noisy_spec': str(noisy_spec_path),
            'clean_spec': str(clean_spec_path),
            'duration': config.DURATION,
            'sample_rate': config.SAMPLE_RATE
        }
        
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return None


def process_dataset(dataset, config):
    """
    Process entire dataset
    
    Args:
        dataset: HuggingFace dataset
        config: Configuration object
    
    Returns:
        metadata_list: List of metadata dicts
    """
    print(f"\nProcessing {len(dataset)} samples...")
    
    metadata_list = []
    
    for idx, sample in enumerate(tqdm(dataset, desc="Processing")):
        metadata = process_single_sample(sample, idx, config)
        if metadata:
            metadata_list.append(metadata)
    
    return metadata_list


def split_and_save_metadata(metadata_list, config):
    """
    Split into train/val and save metadata
    
    Args:
        metadata_list: List of sample metadata
        config: Configuration object
    """
    # Shuffle and split
    np.random.shuffle(metadata_list)
    split_idx = int(len(metadata_list) * config.TRAIN_SPLIT)
    
    train_data = metadata_list[:split_idx]
    val_data = metadata_list[split_idx:]
    
    # Save to JSON
    metadata = {
        'config': {
            'sample_rate': config.SAMPLE_RATE,
            'duration': config.DURATION,
            'n_mels': config.N_MELS,
            'n_fft': config.N_FFT,
            'hop_length': config.HOP_LENGTH
        },
        'train': train_data,
        'val': val_data,
        'stats': {
            'total_samples': len(metadata_list),
            'train_samples': len(train_data),
            'val_samples': len(val_data)
        }
    }
    
    with open(config.METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved metadata to {config.METADATA_FILE}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main preprocessing pipeline"""
    
    print("="*70)
    print("SPEECH ENHANCEMENT - DATA PREPROCESSING")
    print("="*70)
    
    # Initialize config
    config = Config()
    config.create_directories()
    
    # Load dataset
    dataset = load_voicebank_dataset(num_samples=config.NUM_SAMPLES)
    
    # Process all samples
    metadata_list = process_dataset(dataset, config)
    
    # Split and save
    split_and_save_metadata(metadata_list, config)
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Audio files: {len(metadata_list)} pairs saved")
    print(f"Spectrograms: {len(metadata_list)} pairs saved")
    print("="*70)


if __name__ == "__main__":
    main()

# preprocessing script end