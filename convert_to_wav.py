"""
Universal Audio Converter to WAV
Converts any audio/video file to WAV format
Supports: MP3, MP4, M4A, FLAC, OGG, AAC, WMA, MKV, AVI, MOV, WEBM, etc.
"""

import subprocess
import sys
from pathlib import Path
import argparse
import os


class AudioConverter:
    """
    Convert any audio/video file to WAV format
    Uses ffmpeg for maximum compatibility
    """
    
    def __init__(self):
        self.supported_formats = [
            # Audio formats
            '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus', 
            '.aiff', '.ape', '.ac3', '.dts', '.alac', '.amr',
            # Video formats (extract audio)
            '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', 
            '.mpg', '.mpeg', '.3gp', '.m4v', '.vob'
        ]
    
    def check_ffmpeg(self):
        """Check if ffmpeg is installed"""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def convert_to_wav(
        self,
        input_file,
        output_file=None,
        sample_rate=16000,
        channels=1,
        bit_depth=16
    ):
        """
        Convert audio/video file to WAV
        
        Args:
            input_file: Path to input file
            output_file: Path to output WAV file (optional)
            sample_rate: Sample rate in Hz (default: 16000)
            channels: Number of channels - 1=mono, 2=stereo (default: 1)
            bit_depth: Bit depth - 16 or 24 (default: 16)
        
        Returns:
            Path to output file
        """
        input_path = Path(input_file)
        
        # Check input file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Generate output filename if not provided
        if output_file is None:
            output_file = input_path.with_suffix('.wav')
        
        output_path = Path(output_file)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check file extension
        ext = input_path.suffix.lower()
        if ext not in self.supported_formats and ext != '.wav':
            print(f"⚠️  Warning: Uncommon format '{ext}' - will try to convert anyway")
        
        # Build ffmpeg command
        command = [
            'ffmpeg',
            '-i', str(input_path),          # Input file
            '-vn',                           # No video
            '-acodec', 'pcm_s16le',         # PCM 16-bit little-endian
            '-ar', str(sample_rate),        # Sample rate
            '-ac', str(channels),           # Number of channels
            '-y',                            # Overwrite output file
            str(output_path)                 # Output file
        ]
        
        # Adjust for bit depth
        if bit_depth == 24:
            command[4] = 'pcm_s24le'  # 24-bit
        
        print(f"Converting: {input_path.name}")
        print(f"  → Sample rate: {sample_rate} Hz")
        print(f"  → Channels: {channels} ({'mono' if channels == 1 else 'stereo'})")
        print(f"  → Bit depth: {bit_depth}-bit")
        
        try:
            # Run ffmpeg
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            print(f"✅ Converted: {output_path}")
            return str(output_path)
            
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode('utf-8') if e.stderr else str(e)
            raise RuntimeError(f"Conversion failed: {error_message}")
    
    def batch_convert(
        self,
        input_dir,
        output_dir=None,
        recursive=False,
        sample_rate=16000,
        channels=1
    ):
        """
        Convert all audio files in a directory
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Output directory (default: same as input)
            recursive: Search subdirectories (default: False)
            sample_rate: Sample rate in Hz
            channels: Number of channels
        
        Returns:
            List of converted files
        """
        input_path = Path(input_dir)
        
        if not input_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {input_dir}")
        
        # Set output directory
        if output_dir is None:
            output_path = input_path / "wav_output"
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        if recursive:
            files = []
            for ext in self.supported_formats:
                files.extend(input_path.rglob(f"*{ext}"))
        else:
            files = []
            for ext in self.supported_formats:
                files.extend(input_path.glob(f"*{ext}"))
        
        if not files:
            print(f"⚠️  No audio files found in {input_dir}")
            return []
        
        print(f"Found {len(files)} audio file(s)")
        print("="*70)
        
        converted = []
        failed = []
        
        for i, file in enumerate(files, 1):
            try:
                # Generate output path maintaining subdirectory structure
                if recursive:
                    rel_path = file.relative_to(input_path)
                    out_file = output_path / rel_path.with_suffix('.wav')
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                else:
                    out_file = output_path / file.with_suffix('.wav').name
                
                print(f"[{i}/{len(files)}] ", end="")
                
                self.convert_to_wav(
                    file,
                    out_file,
                    sample_rate=sample_rate,
                    channels=channels
                )
                converted.append(str(out_file))
                
            except Exception as e:
                print(f"❌ Failed: {file.name} - {e}")
                failed.append(str(file))
        
        print("="*70)
        print(f"✅ Successfully converted: {len(converted)}")
        if failed:
            print(f"❌ Failed: {len(failed)}")
        
        return converted


def install_ffmpeg_instructions():
    """Print instructions to install ffmpeg"""
    print("\n" + "="*70)
    print("❌ ffmpeg is not installed!")
    print("="*70)
    print("\nffmpeg is required for audio conversion.")
    print("\nInstallation instructions:\n")
    
    print("macOS:")
    print("  brew install ffmpeg\n")
    
    print("Ubuntu/Debian:")
    print("  sudo apt update")
    print("  sudo apt install ffmpeg\n")
    
    print("Windows:")
    print("  1. Download from: https://ffmpeg.org/download.html")
    print("  2. Or use Chocolatey: choco install ffmpeg\n")
    
    print("Arch Linux:")
    print("  sudo pacman -S ffmpeg\n")
    
    print("="*70)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Convert any audio/video file to WAV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python convert_to_wav.py song.mp3
  
  # Convert with custom settings
  python convert_to_wav.py song.mp3 --output clean_song.wav --sample-rate 44100 --stereo
  
  # Convert all files in directory
  python convert_to_wav.py --input-dir ./audio_files/ --output-dir ./wav_files/
  
  # Convert all files recursively
  python convert_to_wav.py --input-dir ./audio_files/ --recursive
  
  # Convert for speech recognition (16kHz mono)
  python convert_to_wav.py interview.mp4 --sample-rate 16000 --mono

Supported formats:
  Audio: MP3, M4A, FLAC, OGG, AAC, WMA, OPUS, AIFF, APE, etc.
  Video: MP4, MKV, AVI, MOV, WMV, FLV, WEBM (extracts audio)
        """
    )
    
    # Input arguments
    parser.add_argument('input', nargs='?', type=str,
                       help='Input audio/video file')
    parser.add_argument('--input-dir', '-d', type=str,
                       help='Input directory for batch conversion')
    
    # Output arguments
    parser.add_argument('--output', '-o', type=str,
                       help='Output WAV file (for single file conversion)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (for batch conversion)')
    
    # Audio settings
    parser.add_argument('--sample-rate', '-r', type=int, default=16000,
                       help='Sample rate in Hz (default: 16000)')
    parser.add_argument('--mono', action='store_true',
                       help='Convert to mono (1 channel)')
    parser.add_argument('--stereo', action='store_true',
                       help='Convert to stereo (2 channels)')
    parser.add_argument('--bit-depth', type=int, choices=[16, 24], default=16,
                       help='Bit depth (default: 16)')
    
    # Batch options
    parser.add_argument('--recursive', '-R', action='store_true',
                       help='Process subdirectories recursively')
    
    args = parser.parse_args()
    
    # Determine channels
    if args.stereo:
        channels = 2
    else:
        channels = 1  # Default to mono
    
    # Initialize converter
    converter = AudioConverter()
    
    # Check ffmpeg
    if not converter.check_ffmpeg():
        install_ffmpeg_instructions()
        sys.exit(1)
    
    print("="*70)
    print("🎵 AUDIO CONVERTER TO WAV")
    print("="*70)
    
    try:
        # Batch conversion
        if args.input_dir:
            converter.batch_convert(
                args.input_dir,
                output_dir=args.output_dir,
                recursive=args.recursive,
                sample_rate=args.sample_rate,
                channels=channels
            )
        
        # Single file conversion
        elif args.input:
            converter.convert_to_wav(
                args.input,
                output_file=args.output,
                sample_rate=args.sample_rate,
                channels=channels,
                bit_depth=args.bit_depth
            )
        
        else:
            parser.print_help()
            sys.exit(1)
        
        print("\n" + "="*70)
        print("✅ Conversion complete!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


# For use as a module
def convert(input_file, output_file=None, sample_rate=16000, channels=1):
    """
    Simple function to convert audio file to WAV
    
    Args:
        input_file: Path to input audio/video file
        output_file: Path to output WAV file (optional)
        sample_rate: Sample rate in Hz (default: 16000)
        channels: Number of channels (default: 1 for mono)
    
    Returns:
        Path to output WAV file
    
    Example:
        >>> from convert_to_wav import convert
        >>> convert('song.mp3', 'song.wav', sample_rate=44100, channels=2)
        'song.wav'
    """
    converter = AudioConverter()
    
    if not converter.check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed. Please install ffmpeg first.")
    
    return converter.convert_to_wav(input_file, output_file, sample_rate, channels)


if __name__ == "__main__":
    main()