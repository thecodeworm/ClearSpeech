"""
Comprehensive Model Evaluation for ClearSpeech
Generates graphs, charts, and metrics to evaluate model performance
"""

import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
from scipy import signal as scipy_signal
import sys

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from enhancement_model.model import UNetAudioEnhancer
from enhancement_model.dataset import SpectrogramDataset


class ModelEvaluator:
    """
    Comprehensive evaluation of audio enhancement model
    """
    
    def __init__(self, checkpoint_path, metadata_path, output_dir="evaluation_results"):
        self.checkpoint_path = Path(checkpoint_path)
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio settings
        self.sample_rate = 16000
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 128
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNetAudioEnhancer(in_channels=1, out_channels=1)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.checkpoint_info = checkpoint
        
        print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 0) + 1}")
        print(f"   Val loss: {checkpoint.get('val_loss', 'N/A')}")
    
    def calculate_snr(self, clean_audio, noisy_audio):
        """Calculate Signal-to-Noise Ratio in dB"""
        noise = noisy_audio - clean_audio
        signal_power = np.mean(clean_audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def calculate_pesq(self, reference, degraded):
        """
        Calculate PESQ score (requires pesq library)
        """
        try:
            from pesq import pesq
            score = pesq(self.sample_rate, reference, degraded, 'wb')  # wideband
            return score
        except ImportError:
            print("⚠️  PESQ not available. Install with: pip install pesq")
            return None
    
    def calculate_stoi(self, clean, enhanced):
        """
        Calculate STOI (Short-Time Objective Intelligibility)
        """
        try:
            from pystoi import stoi
            score = stoi(clean, enhanced, self.sample_rate, extended=False)
            return score
        except ImportError:
            print("⚠️  STOI not available. Install with: pip install pystoi")
            return None
    
    def calculate_sisdr(self, reference, estimate):
        """Calculate Scale-Invariant SDR"""
        # Remove mean
        reference = reference - np.mean(reference)
        estimate = estimate - np.mean(estimate)
        
        # Calculate scaling factor
        alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-8)
        
        # Calculate SI-SDR
        scaled_reference = alpha * reference
        noise = estimate - scaled_reference
        
        sisdr = 10 * np.log10(
            (np.sum(scaled_reference ** 2) + 1e-8) / 
            (np.sum(noise ** 2) + 1e-8)
        )
        
        return sisdr
    
    def evaluate_on_dataset(self, split='val', num_samples=None):
        """
        Evaluate model on dataset
        
        Args:
            split: 'train' or 'val'
            num_samples: Number of samples to evaluate (None = all)
        
        Returns:
            Dictionary of metrics
        """
        print(f"\n{'='*70}")
        print(f"Evaluating on {split} set")
        print('='*70)
        
        # Load dataset
        dataset = SpectrogramDataset(self.metadata_path, split=split)
        
        if num_samples is None:
            num_samples = len(dataset)
        else:
            num_samples = min(num_samples, len(dataset))
        
        metrics = {
            'mse': [],
            'mae': [],
            'snr_improvement': [],
            'snr_before': [],
            'snr_after': [],
            'pesq': [],
            'stoi': [],
            'sisdr': [],
            'sisdr_improvement': []
        }
        
        print(f"Evaluating {num_samples} samples...")
        
        for i in tqdm(range(num_samples)):
            noisy_spec, clean_spec = dataset[i]
            
            # Add batch dimension
            noisy_spec = noisy_spec.unsqueeze(0).to(self.device)
            clean_spec = clean_spec.unsqueeze(0).to(self.device)
            
            # Model prediction
            with torch.no_grad():
                enhanced_spec = self.model(noisy_spec)
            
            # Calculate MSE and MAE on spectrograms
            mse = torch.mean((enhanced_spec - clean_spec) ** 2).item()
            mae = torch.mean(torch.abs(enhanced_spec - clean_spec)).item()
            
            metrics['mse'].append(mse)
            metrics['mae'].append(mae)
            
            # Convert to audio for perceptual metrics
            # (This is optional and takes more time)
            if i < 10:  # Only do this for first 10 samples to save time
                try:
                    # Denormalize spectrograms
                    noisy_db = (noisy_spec.squeeze().cpu().numpy() + 1.0) / 2.0 * 80.0 - 80.0
                    clean_db = (clean_spec.squeeze().cpu().numpy() + 1.0) / 2.0 * 80.0 - 80.0
                    enhanced_db = (enhanced_spec.squeeze().cpu().numpy() + 1.0) / 2.0 * 80.0 - 80.0
                    
                    # Convert to audio
                    noisy_audio = self._spec_to_audio(noisy_db)
                    clean_audio = self._spec_to_audio(clean_db)
                    enhanced_audio = self._spec_to_audio(enhanced_db)
                    
                    # Calculate SNR
                    snr_before = self.calculate_snr(clean_audio, noisy_audio)
                    snr_after = self.calculate_snr(clean_audio, enhanced_audio)
                    metrics['snr_before'].append(snr_before)
                    metrics['snr_after'].append(snr_after)
                    metrics['snr_improvement'].append(snr_after - snr_before)
                    
                    # Calculate perceptual metrics
                    pesq_score = self.calculate_pesq(clean_audio, enhanced_audio)
                    if pesq_score is not None:
                        metrics['pesq'].append(pesq_score)
                    
                    stoi_score = self.calculate_stoi(clean_audio, enhanced_audio)
                    if stoi_score is not None:
                        metrics['stoi'].append(stoi_score)
                    
                    # Calculate SI-SDR
                    sisdr_before = self.calculate_sisdr(clean_audio, noisy_audio)
                    sisdr_after = self.calculate_sisdr(clean_audio, enhanced_audio)
                    metrics['sisdr'].append(sisdr_after)
                    metrics['sisdr_improvement'].append(sisdr_after - sisdr_before)
                    
                except Exception as e:
                    print(f"Error calculating audio metrics for sample {i}: {e}")
        
        # Calculate statistics
        results = {}
        for key, values in metrics.items():
            if values:
                results[f'{key}_mean'] = np.mean(values)
                results[f'{key}_std'] = np.std(values)
                results[f'{key}_median'] = np.median(values)
        
        return results, metrics
    
    def _spec_to_audio(self, mel_spec_db):
        """Convert mel-spectrogram to audio using Griffin-Lim"""
        mel_spec_db = np.clip(mel_spec_db, -80.0, 0.0)
        mel_spec = librosa.db_to_power(mel_spec_db)
        mel_spec = np.maximum(mel_spec, 1e-10)
        
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_iter=60
        )
        
        return audio
    
    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        # Try to load training history from tensorboard logs
        log_dir = PROJECT_ROOT / "enhancement_model" / "logs"
        
        # For now, just plot the checkpoint info
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Mock data - replace with actual training history if available
        epochs = [self.checkpoint_info.get('epoch', 0) + 1]
        train_loss = [self.checkpoint_info.get('train_loss', 0)]
        val_loss = [self.checkpoint_info.get('val_loss', 0)]
        
        ax.plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2, markersize=8)
        ax.plot(epochs, val_loss, 's-', label='Val Loss', linewidth=2, markersize=8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: training_curves.png")
    
    def plot_metrics_comparison(self, metrics):
        """Plot comparison of different metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
        
        # MSE distribution
        if metrics['mse']:
            axes[0, 0].hist(metrics['mse'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('MSE Distribution')
            axes[0, 0].set_xlabel('MSE')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(np.mean(metrics['mse']), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(metrics["mse"]):.4f}')
            axes[0, 0].legend()
        
        # MAE distribution
        if metrics['mae']:
            axes[0, 1].hist(metrics['mae'], bins=30, color='coral', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('MAE Distribution')
            axes[0, 1].set_xlabel('MAE')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(np.mean(metrics['mae']), color='red', linestyle='--',
                              label=f'Mean: {np.mean(metrics["mae"]):.4f}')
            axes[0, 1].legend()
        
        # SNR Improvement
        if metrics['snr_improvement']:
            axes[0, 2].hist(metrics['snr_improvement'], bins=20, color='green', alpha=0.7, edgecolor='black')
            axes[0, 2].set_title('SNR Improvement')
            axes[0, 2].set_xlabel('SNR Improvement (dB)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].axvline(np.mean(metrics['snr_improvement']), color='red', linestyle='--',
                              label=f'Mean: {np.mean(metrics["snr_improvement"]):.2f} dB')
            axes[0, 2].legend()
        
        # SNR Before vs After
        if metrics['snr_before'] and metrics['snr_after']:
            x = np.arange(len(metrics['snr_before']))
            width = 0.35
            axes[1, 0].bar(x - width/2, metrics['snr_before'], width, label='Before', alpha=0.7)
            axes[1, 0].bar(x + width/2, metrics['snr_after'], width, label='After', alpha=0.7)
            axes[1, 0].set_title('SNR Before vs After')
            axes[1, 0].set_xlabel('Sample')
            axes[1, 0].set_ylabel('SNR (dB)')
            axes[1, 0].legend()
        
        # PESQ scores
        if metrics['pesq']:
            axes[1, 1].hist(metrics['pesq'], bins=15, color='purple', alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('PESQ Scores')
            axes[1, 1].set_xlabel('PESQ')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].axvline(np.mean(metrics['pesq']), color='red', linestyle='--',
                              label=f'Mean: {np.mean(metrics["pesq"]):.2f}')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'PESQ not available\nInstall with:\npip install pesq',
                           ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('PESQ Scores')
        
        # STOI scores
        if metrics['stoi']:
            axes[1, 2].hist(metrics['stoi'], bins=15, color='orange', alpha=0.7, edgecolor='black')
            axes[1, 2].set_title('STOI Scores')
            axes[1, 2].set_xlabel('STOI')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].axvline(np.mean(metrics['stoi']), color='red', linestyle='--',
                              label=f'Mean: {np.mean(metrics["stoi"]):.3f}')
            axes[1, 2].legend()
        else:
            axes[1, 2].text(0.5, 0.5, 'STOI not available\nInstall with:\npip install pystoi',
                           ha='center', va='center', fontsize=12)
            axes[1, 2].set_title('STOI Scores')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: metrics_comparison.png")
    
    def plot_spectrogram_comparison(self, num_samples=3):
        """Plot spectrograms: noisy, clean, enhanced"""
        dataset = SpectrogramDataset(self.metadata_path, split='val')
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
        
        for i in range(num_samples):
            noisy_spec, clean_spec = dataset[i]
            
            # Model prediction
            with torch.no_grad():
                noisy_input = noisy_spec.unsqueeze(0).to(self.device)
                enhanced_spec = self.model(noisy_input).squeeze().cpu()
            
            # Denormalize for visualization
            noisy_db = (noisy_spec.squeeze().numpy() + 1.0) / 2.0 * 80.0 - 80.0
            clean_db = (clean_spec.squeeze().numpy() + 1.0) / 2.0 * 80.0 - 80.0
            enhanced_db = (enhanced_spec.numpy() + 1.0) / 2.0 * 80.0 - 80.0
            
            # Plot
            ax_row = axes[i] if num_samples > 1 else axes
            
            im1 = ax_row[0].imshow(noisy_db, aspect='auto', origin='lower', cmap='viridis')
            ax_row[0].set_title(f'Sample {i+1}: Noisy')
            ax_row[0].set_ylabel('Mel Bin')
            plt.colorbar(im1, ax=ax_row[0])
            
            im2 = ax_row[1].imshow(clean_db, aspect='auto', origin='lower', cmap='viridis')
            ax_row[1].set_title(f'Sample {i+1}: Clean (Target)')
            plt.colorbar(im2, ax=ax_row[1])
            
            im3 = ax_row[2].imshow(enhanced_db, aspect='auto', origin='lower', cmap='viridis')
            ax_row[2].set_title(f'Sample {i+1}: Enhanced (Model Output)')
            plt.colorbar(im3, ax=ax_row[2])
            
            if i == num_samples - 1:
                ax_row[0].set_xlabel('Time Frame')
                ax_row[1].set_xlabel('Time Frame')
                ax_row[2].set_xlabel('Time Frame')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'spectrogram_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: spectrogram_comparison.png")
    
    def generate_report(self, results):
        """Generate evaluation report"""
        report_path = self.output_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CLEARSPEECH MODEL EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("Model Information:\n")
            f.write(f"  Checkpoint: {self.checkpoint_path}\n")
            f.write(f"  Epoch: {self.checkpoint_info.get('epoch', 0) + 1}\n")
            f.write(f"  Train Loss: {self.checkpoint_info.get('train_loss', 'N/A'):.6f}\n")
            f.write(f"  Val Loss: {self.checkpoint_info.get('val_loss', 'N/A'):.6f}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write("-"*70 + "\n")
            
            for key, value in sorted(results.items()):
                f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("Interpretation:\n")
            f.write("-"*70 + "\n")
            
            if 'mse_mean' in results:
                f.write(f"\nMSE: {results['mse_mean']:.6f}\n")
                f.write(f"  - Lower is better\n")
                f.write(f"  - Your model: {results['mse_mean']:.6f}\n")
                if results['mse_mean'] < 0.05:
                    f.write(f"  - ✅ Excellent!\n")
                elif results['mse_mean'] < 0.1:
                    f.write(f"  - ✅ Good\n")
                else:
                    f.write(f"  - ⚠️  Could be improved\n")
            
            if 'snr_improvement_mean' in results:
                f.write(f"\nSNR Improvement: {results['snr_improvement_mean']:.2f} dB\n")
                f.write(f"  - Positive values = improvement\n")
                if results['snr_improvement_mean'] > 5:
                    f.write(f"  - ✅ Significant improvement!\n")
                elif results['snr_improvement_mean'] > 0:
                    f.write(f"  - ✅ Moderate improvement\n")
                else:
                    f.write(f"  - ❌ No improvement\n")
            
            if 'pesq_mean' in results:
                f.write(f"\nPESQ: {results['pesq_mean']:.2f}\n")
                f.write(f"  - Range: 1.0 (bad) to 4.5 (excellent)\n")
                if results['pesq_mean'] > 3.5:
                    f.write(f"  - ✅ Excellent quality!\n")
                elif results['pesq_mean'] > 2.5:
                    f.write(f"  - ✅ Good quality\n")
                else:
                    f.write(f"  - ⚠️  Fair quality\n")
            
            if 'stoi_mean' in results:
                f.write(f"\nSTOI: {results['stoi_mean']:.3f}\n")
                f.write(f"  - Range: 0.0 to 1.0 (higher is better)\n")
                if results['stoi_mean'] > 0.9:
                    f.write(f"  - ✅ Excellent intelligibility!\n")
                elif results['stoi_mean'] > 0.7:
                    f.write(f"  - ✅ Good intelligibility\n")
                else:
                    f.write(f"  - ⚠️  Moderate intelligibility\n")
        
        print(f"✅ Saved: evaluation_report.txt")
        
        # Also print to console
        with open(report_path, 'r') as f:
            print("\n" + f.read())


def main():
    """Run complete evaluation"""
    print("="*70)
    print("CLEARSPEECH MODEL EVALUATION")
    print("="*70)
    
    # Paths
    checkpoint_path = "enhancement_model/checkpoints/best_model.pt"
    metadata_path = "data/metadata/metadata.json"
    output_dir = "evaluation_results"
    
    # Create evaluator
    evaluator = ModelEvaluator(checkpoint_path, metadata_path, output_dir)
    
    # Run evaluation
    results, metrics = evaluator.evaluate_on_dataset(split='val', num_samples=50)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    evaluator.plot_training_curves()
    evaluator.plot_metrics_comparison(metrics)
    evaluator.plot_spectrogram_comparison(num_samples=3)
    
    # Generate report
    print("\n📝 Generating report...")
    evaluator.generate_report(results)
    
    # Save metrics to JSON
    results_json = output_dir / Path("metrics.json")
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved: metrics.json")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - evaluation_report.txt")
    print(f"  - training_curves.png")
    print(f"  - metrics_comparison.png")
    print(f"  - spectrogram_comparison.png")
    print(f"  - metrics.json")

if __name__ == "__main__":
    main()