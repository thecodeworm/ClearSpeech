"""
FastAPI Backend with TTS Option
Provides REST API endpoints for audio processing + Text-to-Speech
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import soundfile as sf
import tempfile
import os
from pathlib import Path
import logging
from typing import Optional
import time
import json
import io

from .inference_pipeline import EnhancementPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()

# Initialize FastAPI app
app = FastAPI(
    title="Speech Enhancement API with TTS",
    description="API for audio enhancement, transcription, and text-to-speech",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None
temp_files = {}

# Configuration
class Config:
    CNN_CHECKPOINT = (BASE_DIR / "../enhancement_model/checkpoints/best_model.pt").resolve()
    WHISPER_MODEL = "base"
    DEVICE = "cpu"
    USE_FP16 = False
    MAX_FILE_SIZE = 50 * 1024 * 1024
    TEMP_DIR = Path(tempfile.gettempdir()) / "clearspeech"
    
    @classmethod
    def setup(cls):
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)


# Response models
class ProcessResponse(BaseModel):
    success: bool
    transcript: str
    duration: float
    language: str
    enhanced_audio_url: str
    tts_audio_url: Optional[str] = None  # NEW: TTS option
    segments: list = []
    processing_time: float


class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    voice: str = "default"


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    cnn_checkpoint: str
    whisper_model: str
    device: str
    tts_available: bool  # NEW


@app.on_event("startup")
async def startup_event():
    """Load models on server startup"""
    global pipeline
    logger.info("🚀 Starting server and loading models...")
    
    try:
        Config.setup()
        
        if not Config.CNN_CHECKPOINT.exists():
            raise FileNotFoundError(f"CNN checkpoint not found: {Config.CNN_CHECKPOINT}")
        
        pipeline = EnhancementPipeline(
            cnn_checkpoint_path=str(Config.CNN_CHECKPOINT),
            whisper_model_name=Config.WHISPER_MODEL,
            device=Config.DEVICE,
            use_fp16=Config.USE_FP16
        )
        logger.info("✅ Models loaded successfully!")
        logger.info(f"📍 Checkpoint: {Config.CNN_CHECKPOINT}")
        logger.info(f"📍 Whisper: {Config.WHISPER_MODEL}")
        logger.info(f"📍 Device: {Config.DEVICE}")
        
        # Check TTS availability
        try:
            import pyttsx3
            logger.info("✅ TTS (pyttsx3) available")
        except ImportError:
            logger.warning("⚠️  TTS not available - install with: pip install pyttsx3")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    logger.info("Shutting down server...")
    
    for filepath in temp_files.values():
        try:
            if Path(filepath).exists():
                os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to cleanup {filepath}: {e}")
    
    temp_files.clear()
    
    try:
        import shutil
        if Config.TEMP_DIR.exists():
            shutil.rmtree(Config.TEMP_DIR)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp directory: {e}")


# ============================================================================
# TTS FUNCTIONS
# ============================================================================

def generate_tts_pyttsx3(text: str, output_path: str, language: str = "en"):
    """
    Generate TTS using pyttsx3 (offline, cross-platform)
    """
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Configure voice
        voices = engine.getProperty('voices')
        
        # Try to set appropriate voice based on language
        if language == "en":
            # Use first English voice (usually default)
            if len(voices) > 0:
                engine.setProperty('voice', voices[0].id)
        
        # Configure speech rate and volume
        engine.setProperty('rate', 150)    # Speed (words per minute)
        engine.setProperty('volume', 0.9)  # Volume (0-1)
        
        # Save to file
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        
        return True
        
    except Exception as e:
        logger.error(f"pyttsx3 TTS failed: {e}")
        return False


def generate_tts_gtts(text: str, output_path: str, language: str = "en"):
    """
    Generate TTS using gTTS (requires internet, better quality)
    """
    try:
        from gtts import gTTS
        
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(output_path)
        
        return True
        
    except Exception as e:
        logger.error(f"gTTS failed: {e}")
        return False


def generate_tts(text: str, output_path: str, language: str = "en", method: str = "auto"):
    """
    Generate TTS using available method
    
    Args:
        text: Text to convert to speech
        output_path: Where to save audio file
        language: Language code (en, es, fr, etc.)
        method: 'pyttsx3', 'gtts', or 'auto'
    
    Returns:
        True if successful, False otherwise
    """
    if method == "auto":
        # Try gTTS first (better quality), fall back to pyttsx3
        if generate_tts_gtts(text, output_path, language):
            return True
        return generate_tts_pyttsx3(text, output_path, language)
    
    elif method == "gtts":
        return generate_tts_gtts(text, output_path, language)
    
    elif method == "pyttsx3":
        return generate_tts_pyttsx3(text, output_path, language)
    
    return False


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "ClearSpeech API - Speech Enhancement, Transcription & TTS",
        "version": "2.1.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "process": "/process",
            "enhance": "/enhance",
            "transcribe": "/transcribe",
            "tts": "/tts"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    tts_available = False
    try:
        import pyttsx3
        tts_available = True
    except ImportError:
        try:
            import gtts
            tts_available = True
        except ImportError:
            pass
    
    return {
        "status": "healthy" if pipeline is not None else "unhealthy",
        "models_loaded": pipeline is not None,
        "cnn_checkpoint": str(Config.CNN_CHECKPOINT),
        "whisper_model": Config.WHISPER_MODEL,
        "device": Config.DEVICE,
        "tts_available": tts_available
    }


@app.post("/process", response_model=ProcessResponse)
async def process_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default="en"),
    skip_enhancement: Optional[str] = Form(default="false"),
    generate_tts_param: Optional[str] = Form(default="false", alias="generate_tts")
):
    """
    Process audio file: enhance and transcribe
    
    Args:
        file: Audio file (wav, mp3, flac, etc.)
        language: Target language code (en, es, fr, etc.)
        skip_enhancement: Skip enhancement step
        generate_tts_param: Generate TTS audio from transcript
    
    Returns:
        JSON with transcript, enhanced audio URL, and optional TTS URL
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Convert string parameters to boolean
    skip_enhancement_bool = skip_enhancement.lower() in ['true', '1', 'yes']
    generate_tts_bool = generate_tts_param.lower() in ['true', '1', 'yes']
    
    start_time = time.time()
    
    try:
        # Read file
        contents = await file.read()
        
        # Check file size
        if len(contents) > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {Config.MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        logger.info(f"📥 Processing file: {file.filename} ({len(contents)/1024:.1f} KB)")
        
        # Process audio
        result = pipeline.process(
            contents,
            language=language,
            skip_enhancement=skip_enhancement_bool
        )
        
        # Save enhanced audio to temporary file
        temp_filename = f"enhanced_{int(time.time())}_{file.filename}"
        if not temp_filename.endswith('.wav'):
            temp_filename = temp_filename.rsplit('.', 1)[0] + '.wav'
        
        temp_path = Config.TEMP_DIR / temp_filename
        sf.write(
            temp_path,
            result['enhanced_audio'],
            result['sample_rate']
        )
        
        # Track temp file for cleanup
        temp_files[temp_filename] = str(temp_path)
        
        enhanced_audio_url = f"/download/{temp_filename}"
        
        # Generate TTS if requested
        tts_audio_url = None
        if generate_tts_bool and result['transcript']:
            tts_filename = f"tts_{int(time.time())}_{file.filename}"
            if not tts_filename.endswith('.wav'):
                tts_filename = tts_filename.rsplit('.', 1)[0] + '.wav'
            
            tts_path = Config.TEMP_DIR / tts_filename
            
            # Generate TTS (calling the function, not the boolean!)
            if generate_tts(result['transcript'], str(tts_path), language):
                temp_files[tts_filename] = str(tts_path)
                tts_audio_url = f"/download/{tts_filename}"
                logger.info(f"✅ Generated TTS audio")
            else:
                logger.warning(f"⚠️  TTS generation failed")
        
        processing_time = time.time() - start_time
        
        # Return results
        response = {
            "success": True,
            "transcript": result['transcript'],
            "duration": result['duration'],
            "language": result['language'],
            "enhanced_audio_url": enhanced_audio_url,
            "tts_audio_url": tts_audio_url,
            "segments": result.get('segments', []),
            "processing_time": round(processing_time, 2)
        }
        
        logger.info(f"✅ Processed in {processing_time:.2f}s: {file.filename}")
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech
    
    Args:
        text: Text to convert
        language: Language code
        voice: Voice identifier (optional)
    
    Returns:
        Audio file (WAV format)
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        logger.info(f"🔊 Generating TTS for text: {request.text[:50]}...")
        
        # Create temporary file
        temp_filename = f"tts_{int(time.time())}.wav"
        temp_path = Config.TEMP_DIR / temp_filename
        
        # Generate TTS
        if not generate_tts(request.text, str(temp_path), request.language):
            raise HTTPException(
                status_code=500,
                detail="TTS generation failed. Make sure pyttsx3 or gtts is installed."
            )
        
        logger.info(f"✅ Generated TTS: {temp_filename}")
        
        # Return file
        return FileResponse(
            temp_path,
            media_type="audio/wav",
            filename=temp_filename,
            headers={
                "Content-Disposition": f'attachment; filename="{temp_filename}"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating TTS: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download enhanced or TTS audio file
    
    Args:
        filename: Temporary file name
    
    Returns:
        Audio file (WAV format)
    """
    if filename not in temp_files:
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    file_path = Path(temp_files[filename])
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


# Keep other endpoints (enhance, transcribe, cleanup) from original app.py


if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("🎙️  ClearSpeech API Server with TTS")
    print("="*70)
    print(f"CNN Model: {Config.CNN_CHECKPOINT}")
    print(f"Whisper: {Config.WHISPER_MODEL}")
    print(f"Device: {Config.DEVICE}")
    print(f"TTS: Available")
    print("="*70)
    
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )