"""
FastAPI Backend for Speech Enhancement and Transcription
Provides REST API endpoints for audio processing
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
    title="Speech Enhancement API",
    description="API for audio enhancement and transcription using U-Net + Whisper",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None
temp_files = {}  # Track temporary files for cleanup

# Configuration
class Config:
    CNN_CHECKPOINT = (BASE_DIR / "../enhancement_model/checkpoints/best_model.pt").resolve()
    WHISPER_MODEL = "base"  # tiny, base, small, medium, large
    DEVICE = "cpu"  # Change to 'cuda' or 'mps' for GPU
    USE_FP16 = False  # Enable for GPU acceleration
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    TEMP_DIR = Path(tempfile.gettempdir()) / "clearspeech"
    
    @classmethod
    def setup(cls):
        """Create necessary directories"""
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)


# Response models
class ProcessResponse(BaseModel):
    success: bool
    transcript: str
    duration: float
    language: str
    enhanced_audio_url: str
    segments: list = []
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    cnn_checkpoint: str
    whisper_model: str
    device: str


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
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    logger.info("Shutting down server...")
    
    # Cleanup temporary files
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
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "ClearSpeech API - Speech Enhancement & Transcription",
        "version": "2.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "process": "/process",
            "enhance": "/enhance",
            "transcribe": "/transcribe"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if pipeline is not None else "unhealthy",
        "models_loaded": pipeline is not None,
        "cnn_checkpoint": str(Config.CNN_CHECKPOINT),
        "whisper_model": Config.WHISPER_MODEL,
        "device": Config.DEVICE
    }


@app.post("/process", response_model=ProcessResponse)
async def process_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default="en"),
    skip_enhancement: Optional[bool] = Form(default=False)
):
    """
    Process audio file: enhance and transcribe
    
    Args:
        file: Audio file (wav, mp3, flac, etc.)
        language: Target language code (en, es, fr, etc.)
        skip_enhancement: Skip enhancement step
    
    Returns:
        JSON with transcript, enhanced audio URL, and metadata
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
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
            skip_enhancement=skip_enhancement
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
        
        processing_time = time.time() - start_time
        
        # Return results
        response = {
            "success": True,
            "transcript": result['transcript'],
            "duration": result['duration'],
            "language": result['language'],
            "enhanced_audio_url": f"/download/{temp_filename}",
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


@app.post("/enhance")
async def enhance_only(file: UploadFile = File(...)):
    """
    Enhance audio only (no transcription)
    
    Args:
        file: Audio file
    
    Returns:
        Enhanced audio file (WAV format)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        logger.info(f"🧹 Enhancing file: {file.filename}")
        
        # Read file
        contents = await file.read()
        
        if len(contents) > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Load and enhance
        audio = pipeline.audio_processor.load_audio(contents)
        audio = pipeline.audio_processor.normalize_audio(audio)
        enhanced_audio = pipeline.enhance_audio(audio)
        
        # Create temporary file
        temp_filename = f"enhanced_{int(time.time())}_{file.filename}"
        if not temp_filename.endswith('.wav'):
            temp_filename = temp_filename.rsplit('.', 1)[0] + '.wav'
        
        temp_path = Config.TEMP_DIR / temp_filename
        sf.write(
            temp_path,
            enhanced_audio,
            pipeline.audio_processor.sample_rate
        )
        
        logger.info(f"✅ Enhanced: {file.filename}")
        
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
        logger.error(f"❌ Error enhancing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")


@app.post("/transcribe")
async def transcribe_only(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default="en"),
    enhance: Optional[bool] = Form(default=True)
):
    """
    Transcribe audio (with optional enhancement)
    
    Args:
        file: Audio file
        language: Target language code
        enhance: Apply enhancement before transcription
    
    Returns:
        JSON with transcript and metadata
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        logger.info(f"📝 Transcribing file: {file.filename}")
        
        contents = await file.read()
        
        if len(contents) > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        result = pipeline.process(
            contents,
            language=language,
            skip_enhancement=not enhance
        )
        
        response = {
            "success": True,
            "transcript": result['transcript'],
            "duration": result['duration'],
            "language": result['language'],
            "segments": result.get('segments', [])
        }
        
        logger.info(f"✅ Transcribed: {file.filename}")
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error transcribing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download enhanced audio file
    
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


@app.delete("/cleanup/{filename}")
async def cleanup_file(filename: str):
    """
    Manually cleanup a temporary file
    
    Args:
        filename: Temporary file name
    
    Returns:
        Success status
    """
    if filename not in temp_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        file_path = Path(temp_files[filename])
        if file_path.exists():
            os.remove(file_path)
        del temp_files[filename]
        logger.info(f"🗑️  Cleaned up: {filename}")
        return {"success": True, "message": "File cleaned up"}
    except Exception as e:
        logger.error(f"Failed to cleanup {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("🎙️  ClearSpeech API Server")
    print("="*70)
    print(f"CNN Model: {Config.CNN_CHECKPOINT}")
    print(f"Whisper: {Config.WHISPER_MODEL}")
    print(f"Device: {Config.DEVICE}")
    print("="*70)
    
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
