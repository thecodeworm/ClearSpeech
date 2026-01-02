#!/bin/bash

echo "🧪 Comprehensive CNN Test"
echo "=========================="

# 1. Generate test samples
echo "Step 1: Generating noisy samples..."
python generate_noisy_samples.py \
  --input clean_song.wav \
  --output cnn_test/ \
  --noise-types white babble cafe street \
  --snr-levels 5 10 15

# 2. Enhance all samples
echo "Step 2: Enhancing with CNN..."
for noisy_file in cnn_test/noisy_*.wav; do
    filename=$(basename "$noisy_file" .wav)
    echo "  Processing: $filename"
    
    python enhancement_model/infer.py \
      --checkpoint enhancement_model/checkpoints/best_model.pt \
      --input "$noisy_file" \
      --output "cnn_test/enhanced_$filename.wav" \
      --comparison
done

echo "✅ Done! Check cnn_test/ folder"
echo "Listen to comparison files (stereo: left=noisy, right=enhanced)"