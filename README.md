# 🎙️ ClearSpeech

**AI-Powered Speech Enhancement & Transcription System**

ClearSpeech uses a custom U-Net deep learning model to remove background noise from audio, then transcribes the enhanced audio using OpenAI's Whisper. Perfect for cleaning up voice recordings, meeting audio, podcasts, or any noisy speech.

**🌐 Live Website (will be updated)**: https://clearspeech.yourdomain.com 



## 🌟 Features

- 🧹 **AI-Powered Noise Reduction**: Custom U-Net model trained to remove background noise
- 📝 **Automatic Transcription**: Whisper integration for accurate speech-to-text
- ⚡ **Fast Processing**: Optimized pipeline with GPU support
- 🌐 **REST API**: Easy-to-use FastAPI backend
- 🎯 **High Quality**: Val loss of 0.031
- 🔧 **Flexible**: Enhancement-only, transcription-only, or both


## 📋 Table of Contents

- Installation
- Quick Start
- API Documentation
- Project Structure
- Contributing


## 🚀 Installation

### Prerequisites

-   Python 3.8+
    
-   pip
    
-   Optional CUDA GPU
### Step 1: Clone Repository
```
git clone https://github.com/yourusername/ClearSpeech.git
cd ClearSpeech
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
pip install -r enhancement_model/requirements.txt
```

### Step 2: Create Virtual Environment
```
# Create environment 
python -m venv venv 
# Activate (macOS/Linux) 
source venv/bin/activate 
# Activate (Windows) 
venv\Scripts\activate
```
### Step 3: Install Dependencies
```
# Install backend dependencies 
pip install -r backend/requirements.txt 

# Install enhancement model dependencies 
pip install -r enhancement_model/requirements.txt
```
### Step 4: Download Pretrained Model
```
# Install huggingface-hub
pip install huggingface-hub

# Download model
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='thecodeworm/clearspeech-unet',
    filename='best_model.pt',
    local_dir='enhancement_model/checkpoints/'
)
"
```
### Step 5: Generate Noisy Samples
 1. Make your own WAV sample
 2. Run the generate_noisy_samples.py file on the sample to make the audio noisier to test the model
 ```
 # Generate all noise types at multiple SNR levels
python generate_noisy_samples.py \
  --input my_clean_voice.wav \
  --output test_samples/
  ```


## ⚡ Quick Start

### Method 1: Using the API (Recommended)

**Start the server:**
```
python -m backend.app
```
Server starts at `http://localhost:8000`

**Process audio:**
```
# Full pipeline (enhance + transcribe)
curl -X POST "http://localhost:8000/process" \
  -F "file=@your_audio.wav" \
  | jq .

# Enhance only
curl -X POST "http://localhost:8000/enhance" \
  -F "file=@your_audio.wav" \
  -o enhanced_output.wav

# Transcribe only
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@your_audio.wav" \
  -F "enhance=true" \
  | jq .
  ```
**Method 2: Using Python**
```
from backend.inference_pipeline import EnhancementPipeline

# Initialize pipeline
pipeline = EnhancementPipeline(
    cnn_checkpoint_path="enhancement_model/checkpoints/best_model.pt",
    whisper_model_name="base",
    device="cpu"  # or "cuda" or "mps"
)

# Process audio
result = pipeline.process("path/to/noisy_audio.wav")

print(f"Transcript: {result['transcript']}")
print(f"Duration: {result['duration']:.2f}s")

# Save enhanced audio
import soundfile as sf
sf.write("enhanced.wav", result['enhanced_audio'], result['sample_rate'])
```
**Method 3: Command Line**
```
# Enhance audio file
python enhancement_model/infer.py \
  --checkpoint enhancement_model/checkpoints/best_model.pt \
  --input noisy_audio.wav \
  --output enhanced_audio.wav \
  --comparison  # Creates stereo comparison file
  ```
  ## 📚 API Documentation

### Interactive Docs

Once the server is running, visit:

-   **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
-   **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Endpoints

#### `POST /process`

Process audio with enhancement and transcription.

**Request:**
```curl -X POST "http://localhost:8000/process" \
  -F "file=@audio.wav" \
  -F "language=en" \
  -F "skip_enhancement=false"
  ```
  **Response:**
  ```
  {
  "success": true,
  "transcript": "Transcribed text here",
  "duration": 3.5,
  "language": "en",
  "enhanced_audio_url": "/download/enhanced_123.wav",
  "segments": [...],
  "processing_time": 2.3
}
```
#### `POST /enhance`

Enhance audio only (no transcription).

**Request:**
```
curl -X POST "http://localhost:8000/enhance" \
  -F "file=@audio.wav" \
  -o enhanced.wav
  ```
  **Response:** Enhanced audio file (WAV)

#### `POST /transcribe`

Transcribe audio with optional enhancement.

**Request:**
```
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav" \
  -F "language=en" \
  -F "enhance=true"
  ```
  **Response:**
  ```
  {
  "success": true,
  "transcript": "Transcribed text",
  "duration": 3.5,
  "language": "en",
  "segments": [...]
}
```
#### `GET /download/{filename}`
Download enhanced audio file.

#### `GET /health`
Health check endpoint.

## 📁 Project Structure
```
ClearSpeech/
├── backend/                    # FastAPI backend
│   ├── app.py                 # Main API server
│   ├── inference_pipeline.py  # Processing pipeline
│   └── requirements.txt
├── enhancement_model/          # U-Net model
│   ├── model.py               # U-Net architecture
│   ├── dataset.py             # PyTorch dataset
│   ├── train.py               # Training script
│   ├── infer.py               # Inference script
│   ├── checkpoints/           # Trained models
│   │   └── best_model.pt
│   └── requirements.txt
├── data/                       # Training/test data
│   ├── audio_clean/           # Clean audio
│   ├── audio_raw/             # Noisy audio
│   ├── metadata/
│   │   └── metadata.json      # Dataset metadata
│   └── spectrograms/          # Mel-spectrograms
│       ├── clean/
│       └── noisy/
├── frontend/                   # Web interface (optional)
│   ├── index.html
│   └── script.js
├── tests/                      # Test files
│   └── test_backend.py
├── README.md
└── requirements.txt
```
## 🤝 Contributing

We welcome contributions! Here's how:

1.  **Fork the repository**
2.  **Create a feature branch**: `git checkout -b feature/amazing-feature`
3.  **Commit changes**: `git commit -m 'Add amazing feature'`
4.  **Push to branch**: `git push origin feature/amazing-feature`
5.  **Open a Pull Request**

**Development Setup**
```
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests before committing
python -m pytest tests/

# Format code
black backend/ enhancement_model/
```
## 🙏 Acknowledgments

-   **U-Net Architecture**: Inspired by [Ronneberger et al.](https://arxiv.org/abs/1505.04597)
-   **Whisper**: [OpenAI Whisper](https://github.com/openai/whisper)
-   **Training Data**: [LibriSpeech](http://www.openslr.org/12/), [MS-SNSD](https://github.com/microsoft/MS-SNSD)

## 📧 Contact
**Project Maintainers**: Aditya Chanda, Josh Pal, Advik Kumar Singh

**Project Link**: [https://github.com/thecodeworm/ClearSpeech](https://github.com/thecodeworm/ClearSpeech)

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

----------

**Built with ❤️ using PyTorch and FastAPI**

