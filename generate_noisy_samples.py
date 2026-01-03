"""
Synthetic Noise Generator for Testing Audio Enhancement
Adds various types of realistic noise to clean audio samples
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse
from scipy import signal
import random


class NoiseGenerator:
    """
    Generate various types of synthetic noise for audio testing
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def add_white_noise(self, audio, snr_db=10):
        """
        Add white (Gaussian) noise
        
        Args:
            audio: Clean audio signal
            snr_db: Signal-to-noise ratio in dB (lower = more noise)
        
        Returns:
            Noisy audio
        """
        # Calculate signal power
        signal_power = np.mean(audio ** 2)
        
        # Calculate noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate white noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        
        # Add noise to signal
        noisy_audio = audio + noise
        
        return noisy_audio
    
    def add_pink_noise(self, audio, snr_db=10):
        """
        Add pink noise (1/f noise, more natural sounding)
        
        Args:
            audio: Clean audio signal
            snr_db: Signal-to-noise ratio in dB
        
        Returns:
            Noisy audio
        """
        # Generate white noise
        white_noise = np.random.randn(len(audio))
        
        # Apply pink filter (approximate 1/f spectrum)
        # Use a simple IIR filter
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        pink_noise = signal.lfilter(b, a, white_noise)
        
        # Normalize
        pink_noise = pink_noise / np.std(pink_noise)
        
        # Scale to desired SNR
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        pink_noise = pink_noise * np.sqrt(noise_power)
        
        noisy_audio = audio + pink_noise
        
        return noisy_audio
    
    def add_brown_noise(self, audio, snr_db=10):
        """
        Add brown noise (deeper, rumbling noise)
        
        Args:
            audio: Clean audio signal
            snr_db: Signal-to-noise ratio in dB
        
        Returns:
            Noisy audio
        """
        # Generate white noise
        white_noise = np.random.randn(len(audio))
        
        # Integrate to get brown noise
        brown_noise = np.cumsum(white_noise)
        
        # Normalize
        brown_noise = brown_noise / np.std(brown_noise)
        
        # Scale to desired SNR
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        brown_noise = brown_noise * np.sqrt(noise_power)
        
        noisy_audio = audio + brown_noise
        
        return noisy_audio
    
    def add_babble_noise(self, audio, num_speakers=5, snr_db=10):
        """
        Simulate background conversation/babble noise
        
        Args:
            audio: Clean audio signal
            num_speakers: Number of simulated speakers
            snr_db: Signal-to-noise ratio in dB
        
        Returns:
            Noisy audio
        """
        babble = np.zeros(len(audio))
        
        # Simulate multiple speakers with different frequencies
        for i in range(num_speakers):
            # Random fundamental frequency (80-300 Hz for human voice)
            f0 = np.random.uniform(80, 300)
            
            # Generate voice-like signal with harmonics
            t = np.arange(len(audio)) / self.sample_rate
            voice = np.zeros(len(audio))
            
            # Add harmonics
            for harmonic in range(1, 6):
                amplitude = 1.0 / harmonic
                voice += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)
            
            # Add random amplitude modulation (speaking rhythm)
            mod_freq = np.random.uniform(2, 8)  # 2-8 Hz modulation
            modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
            voice *= modulation
            
            # Add to babble
            babble += voice
        
        # Add some randomness
        babble += np.random.randn(len(audio)) * 0.1
        
        # Normalize and scale
        babble = babble / np.std(babble)
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        babble = babble * np.sqrt(noise_power)
        
        noisy_audio = audio + babble
        
        return noisy_audio
    
    def add_cafe_noise(self, audio, snr_db=10):
        """
        Simulate cafe/restaurant ambient noise
        
        Args:
            audio: Clean audio signal
            snr_db: Signal-to-noise ratio in dB
        
        Returns:
            Noisy audio
        """
        # Combine different noise types
        # 40% babble, 30% white, 20% pink, 10% clinking sounds
        
        babble = self.add_babble_noise(np.zeros(len(audio)), num_speakers=8, snr_db=float('inf'))
        white = np.random.randn(len(audio))
        pink = self.add_pink_noise(np.zeros(len(audio)), snr_db=float('inf')) - np.zeros(len(audio))
        
        # Simulate clinking/impact sounds
        clinks = np.zeros(len(audio))
        num_clinks = int(len(audio) / self.sample_rate * 2)  # 2 per second
        for _ in range(num_clinks):
            pos = np.random.randint(0, len(audio) - 1000)
            # Create short impact sound
            clink_duration = int(self.sample_rate * 0.1)  # 100ms
            t = np.arange(clink_duration) / self.sample_rate
            clink = np.exp(-t * 30) * np.sin(2 * np.pi * 2000 * t)
            if pos + len(clink) < len(audio):
                clinks[pos:pos+len(clink)] += clink
        
        # Mix all components
        cafe_noise = 0.4 * babble + 0.3 * white + 0.2 * pink + 0.1 * clinks
        
        # Normalize and scale
        cafe_noise = cafe_noise / np.std(cafe_noise)
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        cafe_noise = cafe_noise * np.sqrt(noise_power)
        
        noisy_audio = audio + cafe_noise
        
        return noisy_audio
    
    def add_street_noise(self, audio, snr_db=10):
        """
        Simulate street/traffic noise
        
        Args:
            audio: Clean audio signal
            snr_db: Signal-to-noise ratio in dB
        
        Returns:
            Noisy audio
        """
        # Low frequency rumble (traffic)
        t = np.arange(len(audio)) / self.sample_rate
        
        # Multiple low frequency components
        rumble = np.zeros(len(audio))
        for freq in [30, 50, 70, 90, 120]:
            amplitude = np.random.uniform(0.5, 1.0)
            rumble += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add brown noise for continuous traffic
        brown = self.add_brown_noise(np.zeros(len(audio)), snr_db=float('inf')) - np.zeros(len(audio))
        
        # Occasional horn/siren sounds
        horns = np.zeros(len(audio))
        num_horns = int(len(audio) / self.sample_rate * 0.5)  # 0.5 per second
        for _ in range(num_horns):
            pos = np.random.randint(0, len(audio) - self.sample_rate)
            horn_duration = int(self.sample_rate * 0.3)
            t_horn = np.arange(horn_duration) / self.sample_rate
            horn_freq = np.random.choice([400, 500, 600, 800])
            horn = np.sin(2 * np.pi * horn_freq * t_horn) * np.exp(-t_horn * 2)
            if pos + len(horn) < len(audio):
                horns[pos:pos+len(horn)] += horn
        
        # Mix components
        street_noise = 0.5 * rumble + 0.4 * brown + 0.1 * horns
        
        # Normalize and scale
        street_noise = street_noise / np.std(street_noise)
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        street_noise = street_noise * np.sqrt(noise_power)
        
        noisy_audio = audio + street_noise
        
        return noisy_audio
    
    def add_fan_noise(self, audio, snr_db=10):
        """
        Simulate fan/HVAC noise
        
        Args:
            audio: Clean audio signal
            snr_db: Signal-to-noise ratio in dB
        
        Returns:
            Noisy audio
        """
        # Combination of tonal components and broadband noise
        t = np.arange(len(audio)) / self.sample_rate
        
        # Tonal components (fan blade frequency and harmonics)
        fundamental = 120  # Hz
        fan_noise = np.zeros(len(audio))
        
        for harmonic in range(1, 8):
            amplitude = 1.0 / (harmonic ** 1.5)
            fan_noise += amplitude * np.sin(2 * np.pi * fundamental * harmonic * t)
        
        # Add broadband component
        broadband = np.random.randn(len(audio)) * 0.3
        
        # Combine
        fan_noise = fan_noise + broadband
        
        # Normalize and scale
        fan_noise = fan_noise / np.std(fan_noise)
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        fan_noise = fan_noise * np.sqrt(noise_power)
        
        noisy_audio = audio + fan_noise
        
        return noisy_audio


def generate_noisy_samples(
    input_path,
    output_dir,
    noise_types=None,
    snr_levels=None,
    sample_rate=16000
):
    """
    Generate noisy samples from a clean audio file
    
    Args:
        input_path: Path to clean audio file
        output_dir: Directory to save noisy samples
        noise_types: List of noise types to apply (default: all)
        snr_levels: List of SNR levels in dB (default: [5, 10, 15, 20])
        sample_rate: Audio sample rate
    """
    # Default noise types
    if noise_types is None:
        noise_types = ['white', 'pink', 'brown', 'babble', 'cafe', 'street', 'fan']
    
    # Default SNR levels
    if snr_levels is None:
        snr_levels = [5, 10, 15, 20]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load clean audio
    print(f"Loading clean audio: {input_path}")
    clean_audio, sr = librosa.load(input_path, sr=sample_rate, mono=True)
    
    # Save clean audio as reference
    clean_output = output_dir / f"clean_{Path(input_path).stem}.wav"
    sf.write(clean_output, clean_audio, sample_rate)
    print(f"Saved clean reference: {clean_output}")
    
    # Initialize noise generator
    noise_gen = NoiseGenerator(sample_rate)
    
    # Generate noisy samples
    total_samples = len(noise_types) * len(snr_levels)
    sample_count = 0
    
    print(f"\nGenerating {total_samples} noisy samples...")
    print("="*70)
    
    for noise_type in noise_types:
        for snr_db in snr_levels:
            sample_count += 1
            
            # Apply noise
            if noise_type == 'white':
                noisy = noise_gen.add_white_noise(clean_audio, snr_db)
            elif noise_type == 'pink':
                noisy = noise_gen.add_pink_noise(clean_audio, snr_db)
            elif noise_type == 'brown':
                noisy = noise_gen.add_brown_noise(clean_audio, snr_db)
            elif noise_type == 'babble':
                noisy = noise_gen.add_babble_noise(clean_audio, snr_db=snr_db)
            elif noise_type == 'cafe':
                noisy = noise_gen.add_cafe_noise(clean_audio, snr_db)
            elif noise_type == 'street':
                noisy = noise_gen.add_street_noise(clean_audio, snr_db)
            elif noise_type == 'fan':
                noisy = noise_gen.add_fan_noise(clean_audio, snr_db)
            else:
                print(f"Unknown noise type: {noise_type}")
                continue
            
            # Normalize to prevent clipping
            noisy = np.clip(noisy, -1.0, 1.0)
            
            # Save noisy sample
            filename = f"noisy_{Path(input_path).stem}_{noise_type}_snr{snr_db}db.wav"
            output_path = output_dir / filename
            sf.write(output_path, noisy, sample_rate)
            
            print(f"[{sample_count}/{total_samples}] {noise_type:8s} @ SNR {snr_db:2d}dB → {filename}")
    
    print("="*70)
    print(f"Generated {sample_count} noisy samples in: {output_dir}")
    print("\nNoise types generated:")
    for noise_type in noise_types:
        print(f"  • {noise_type}")
    print(f"\nSNR levels: {snr_levels} dB")


def test_single_noise(input_path, output_path, noise_type='white', snr_db=10):
    """
    Quick test: generate a single noisy sample
    
    Args:
        input_path: Path to clean audio
        output_path: Path to save noisy audio
        noise_type: Type of noise to add
        snr_db: Signal-to-noise ratio in dB
    """
    print(f"Generating single test sample...")
    print(f"  Input: {input_path}")
    print(f"  Noise: {noise_type} @ SNR {snr_db}dB")
    
    # Load audio
    audio, sr = librosa.load(input_path, sr=16000, mono=True)
    
    # Generate noise
    noise_gen = NoiseGenerator(sr)
    
    if noise_type == 'white':
        noisy = noise_gen.add_white_noise(audio, snr_db)
    elif noise_type == 'pink':
        noisy = noise_gen.add_pink_noise(audio, snr_db)
    elif noise_type == 'brown':
        noisy = noise_gen.add_brown_noise(audio, snr_db)
    elif noise_type == 'babble':
        noisy = noise_gen.add_babble_noise(audio, snr_db=snr_db)
    elif noise_type == 'cafe':
        noisy = noise_gen.add_cafe_noise(audio, snr_db)
    elif noise_type == 'street':
        noisy = noise_gen.add_street_noise(audio, snr_db)
    elif noise_type == 'fan':
        noisy = noise_gen.add_fan_noise(audio, snr_db)
    else:
        print(f"Unknown noise type: {noise_type}")
        return
    
    # Normalize
    noisy = np.clip(noisy, -1.0, 1.0)
    
    # Save
    sf.write(output_path, noisy, sr)
    print(f"Saved: {output_path}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic noisy audio for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all noise types at multiple SNR levels
  python generate_noisy_samples.py --input my_voice.wav --output test_samples/
  
  # Generate specific noise types
  python generate_noisy_samples.py --input my_voice.wav --output test_samples/ \\
    --noise-types white pink babble
  
  # Generate at specific SNR levels
  python generate_noisy_samples.py --input my_voice.wav --output test_samples/ \\
    --snr-levels 5 10 15
  
  # Quick test: single sample
  python generate_noisy_samples.py --input my_voice.wav --output test.wav \\
    --test --noise-type cafe --snr 10

Available noise types:
  white   - White (Gaussian) noise
  pink    - Pink (1/f) noise
  brown   - Brown (low frequency) noise
  babble  - Background conversation
  cafe    - Cafe/restaurant ambience
  street  - Traffic/street noise
  fan     - Fan/HVAC noise
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to clean audio file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory or file path')
    parser.add_argument('--noise-types', nargs='+', 
                       choices=['white', 'pink', 'brown', 'babble', 'cafe', 'street', 'fan'],
                       help='Types of noise to generate (default: all)')
    parser.add_argument('--snr-levels', nargs='+', type=int,
                       help='SNR levels in dB (default: 5 10 15 20)')
    parser.add_argument('--test', action='store_true',
                       help='Quick test mode: generate single sample')
    parser.add_argument('--noise-type', type=str, default='white',
                       choices=['white', 'pink', 'brown', 'babble', 'cafe', 'street', 'fan'],
                       help='Noise type for test mode')
    parser.add_argument('--snr', type=int, default=10,
                       help='SNR level for test mode (default: 10)')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    print("="*70)
    print("🔊 SYNTHETIC NOISE GENERATOR")
    print("="*70)
    
    if args.test:
        # Quick test mode
        test_single_noise(args.input, args.output, args.noise_type, args.snr)
    else:
        # Full generation mode
        generate_noisy_samples(
            args.input,
            args.output,
            noise_types=args.noise_types,
            snr_levels=args.snr_levels
        )
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()