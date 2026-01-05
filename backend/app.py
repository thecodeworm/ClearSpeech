"""
FastAPI Backend for Hugging Face Spaces
Provides REST API endpoints for audio processing + Text-to-Speech
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import soundfile as sf
import tempfile
import os
from pathlib import Path
import logging
from typing import Optional
import time
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio
from huggingface_hub import hf_hub_download

# Direct import (no 'backend.' prefix for HF Spaces)
from inference_pipeline import EnhancementPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()

# Security: Allowed file types
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm'}
ALLOWED_MIMETYPES = {
    'audio/wav', 'audio/wave', 'audio/x-wav',
    'audio/mpeg', 'audio/mp3',
    'audio/mp4', 'audio/m4a', 'audio/x-m4a',
    'audio/ogg', 'audio/flac', 'audio/webm'
}

# Initialize FastAPI app
app = FastAPI(
    title="ClearSpeech API",
    description="Speech Enhancement, Transcription & Text-to-Speech",
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


# ============================================================================
# SECURITY: Rate Limiting & File Validation
# ============================================================================

class SimpleRateLimiter:
    """Simple in-memory rate limiter for demo protection"""
    def __init__(self, max_requests: int = 20, window_minutes: int = 60):
        self.max_requests = max_requests
        self.window = timedelta(minutes=window_minutes)
        self.requests = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def check_rate_limit(self, client_ip: str) -> bool:
        async with self.lock:
            now = datetime.now()
            self.requests[client_ip] = [
                ts for ts in self.requests[client_ip]
                if now - ts < self.window
            ]
            
            if len(self.requests[client_ip]) >= self.max_requests:
                return False
            
            self.requests[client_ip].append(now)
            return True
    
    async def cleanup(self):
        while True:
            await asyncio.sleep(3600)
            async with self.lock:
                now = datetime.now()
                for ip in list(self.requests.keys()):
                    self.requests[ip] = [ts for ts in self.requests[ip] if now - ts < self.window]
                    if not self.requests[ip]:
                        del self.requests[ip]


rate_limiter = SimpleRateLimiter(max_requests=20, window_minutes=60)


def get_client_ip(request: Request) -> str:
    """Get client IP from request"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    return request.client.host if request.client else "unknown"


def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded file is a safe audio file"""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    if file.content_type and file.content_type not in ALLOWED_MIMETYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}"
        )
    
    if '..' in file.filename or '/' in file.filename or '\\' in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")


# Configuration
class Config:
    # Hugging Face Hub Configuration
    HF_REPO_ID = os.getenv("HF_REPO_ID", "thecodeworm/clearspeech-unet")
    HF_CHECKPOINT_FILENAME = "best_model.pt"
    
    # Local paths
    CHECKPOINT_DIR = Path(tempfile.gettempdir()) / "clearspeech_models"
    CNN_CHECKPOINT = CHECKPOINT_DIR / HF_CHECKPOINT_FILENAME
    
    # Model configuration
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # Can use 'base' with 16GB RAM!
    DEVICE = os.getenv("DEVICE", "cpu")
    USE_FP16 = False
    
    # Limits
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))
    TEMP_DIR = Path(tempfile.gettempdir()) / "clearspeech"
    
    @classmethod
    def setup(cls):
        """Setup: Download checkpoint from Hugging Face Hub"""
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download from HF Hub if not exists
        if not cls.CNN_CHECKPOINT.exists():
            logger.info("="*70)
            logger.info("📥 Downloading model checkpoint from Hugging Face Hub")
            logger.info("="*70)
            logger.info(f"Repository: {cls.HF_REPO_ID}")
            logger.info(f"Filename: {cls.HF_CHECKPOINT_FILENAME}")
            
            try:
                downloaded_path = hf_hub_download(
                    repo_id=cls.HF_REPO_ID,
                    filename=cls.HF_CHECKPOINT_FILENAME,
                    cache_dir=str(cls.CHECKPOINT_DIR.parent),
                    local_dir=str(cls.CHECKPOINT_DIR),
                    local_dir_use_symlinks=False
                )
                
                cls.CNN_CHECKPOINT = Path(downloaded_path)
                logger.info(f"✅ Checkpoint downloaded successfully!")
                logger.info(f"   Saved to: {cls.CNN_CHECKPOINT}")
                logger.info("="*70)
                
            except Exception as e:
                logger.error("="*70)
                logger.error("❌ Failed to download checkpoint")
                logger.error("="*70)
                logger.error(f"Error: {e}")
                logger.error(f"Please verify HF_REPO_ID: {cls.HF_REPO_ID}")
                raise
        else:
            logger.info(f"✅ Using cached checkpoint: {cls.CNN_CHECKPOINT}")


# Response models
class ProcessResponse(BaseModel):
    success: bool
    transcript: str
    duration: float
    language: str
    enhanced_audio_url: str
    tts_audio_url: Optional[str] = None
    segments: list = []
    processing_time: float


class EnhanceResponse(BaseModel):
    success: bool
    enhanced_audio_url: str
    duration: float
    processing_time: float


class TranscribeResponse(BaseModel):
    success: bool
    transcript: str
    duration: float
    language: str
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
    tts_available: bool


@app.on_event("startup")
async def startup_event():
    """Load models on server startup"""
    global pipeline
    logger.info("🚀 Starting ClearSpeech API Server on Hugging Face Spaces...")
    
    try:
        Config.setup()
        
        if not Config.CNN_CHECKPOINT.exists():
            raise FileNotFoundError(f"Checkpoint not found: {Config.CNN_CHECKPOINT}")
        
        pipeline = EnhancementPipeline(
            cnn_checkpoint_path=str(Config.CNN_CHECKPOINT),
            whisper_model_name=Config.WHISPER_MODEL,
            device=Config.DEVICE,
            use_fp16=Config.USE_FP16
        )
        
        logger.info("✅ Models loaded successfully!")
        logger.info(f"📍 CNN Checkpoint: {Config.CNN_CHECKPOINT}")
        logger.info(f"📍 Whisper Model: {Config.WHISPER_MODEL}")
        logger.info(f"📍 Device: {Config.DEVICE}")
        
        # Check TTS
        try:
            import gtts
            logger.info("✅ TTS (gtts) available")
        except ImportError:
            logger.warning("⚠️  TTS not available")
        
        logger.info("="*70)
        logger.info("Server ready! Visit /docs for API documentation")
        logger.info("="*70)
        
        # Start rate limiter cleanup
        asyncio.create_task(rate_limiter.cleanup())
        
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


# ============================================================================
# TTS FUNCTIONS
# ============================================================================

def generate_tts_gtts(text: str, output_path: str, language: str = "en"):
    """Generate TTS using gTTS"""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(output_path)
        return True
    except Exception as e:
        logger.error(f"gTTS failed: {e}")
        return False


def generate_tts(text: str, output_path: str, language: str = "en"):
    """Generate TTS"""
    return generate_tts_gtts(text, output_path, language)


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
        "platform": "Hugging Face Spaces",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "process": "/process (POST)",
            "enhance": "/enhance (POST)",
            "transcribe": "/transcribe (POST)",
            "tts": "/tts (POST)",
            "download": "/download/{filename} (GET)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    tts_available = False
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
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(default="en"),
    skip_enhancement: Optional[str] = Form(default="false"),
    generate_tts_param: Optional[str] = Form(default="false", alias="generate_tts")
):
    """Complete pipeline: enhance + transcribe + optional TTS"""
    # Rate limiting
    client_ip = get_client_ip(request)
    if not await rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 20 requests per hour."
        )
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # File validation
    validate_audio_file(file)
    
    # Convert string parameters to boolean
    skip_enhancement_bool = skip_enhancement.lower() in ['true', '1', 'yes']
    generate_tts_bool = generate_tts_param.lower() in ['true', '1', 'yes']
    
    start_time = time.time()
    
    try:
        contents = await file.read()
        
        if len(contents) > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max: {Config.MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        logger.info(f"📥 Processing: {file.filename} ({len(contents)/1024:.1f} KB)")
        
        # Process audio
        result = pipeline.process(
            contents,
            language=language,
            skip_enhancement=skip_enhancement_bool
        )
        
        # Save enhanced audio
        temp_filename = f"enhanced_{int(time.time())}_{file.filename}"
        if not temp_filename.endswith('.wav'):
            temp_filename = temp_filename.rsplit('.', 1)[0] + '.wav'
        
        temp_path = Config.TEMP_DIR / temp_filename
        sf.write(temp_path, result['enhanced_audio'], result['sample_rate'])
        temp_files[temp_filename] = str(temp_path)
        
        enhanced_audio_url = f"/download/{temp_filename}"
        
        # Generate TTS if requested
        tts_audio_url = None
        if generate_tts_bool and result['transcript']:
            tts_filename = f"tts_{int(time.time())}_{file.filename}"
            if not tts_filename.endswith('.wav'):
                tts_filename = tts_filename.rsplit('.', 1)[0] + '.wav'
            
            tts_path = Config.TEMP_DIR / tts_filename
            
            if generate_tts(result['transcript'], str(tts_path), language):
                temp_files[tts_filename] = str(tts_path)
                tts_audio_url = f"/download/{tts_filename}"
                logger.info(f"✅ Generated TTS")
            else:
                logger.warning(f"⚠️  TTS generation failed")
        
        processing_time = time.time() - start_time
        
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
        
        logger.info(f"✅ Processed in {processing_time:.2f}s")
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/enhance", response_model=EnhanceResponse)
async def enhance_only(
    request: Request,
    file: UploadFile = File(...)
):
    """Enhancement only (no transcription)"""
    # Rate limiting
    client_ip = get_client_ip(request)
    if not await rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # File validation
    validate_audio_file(file)
    
    start_time = time.time()
    
    try:
        contents = await file.read()
        
        # Load and enhance
        audio = pipeline.audio_processor.load_audio(contents)
        enhanced_audio = pipeline.enhance_audio(audio)
        
        # Save
        temp_filename = f"enhanced_{int(time.time())}_{file.filename}"
        if not temp_filename.endswith('.wav'):
            temp_filename = temp_filename.rsplit('.', 1)[0] + '.wav'
        
        temp_path = Config.TEMP_DIR / temp_filename
        sf.write(temp_path, enhanced_audio, pipeline.audio_processor.sample_rate)
        temp_files[temp_filename] = str(temp_path)
        
        duration = len(enhanced_audio) / pipeline.audio_processor.sample_rate
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "enhanced_audio_url": f"/download/{temp_filename}",
            "duration": duration,
            "processing_time": round(processing_time, 2)
        }
    
    except Exception as e:
        logger.error(f"❌ Enhancement error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_only(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(default="en"),
    enhance_first: Optional[str] = Form(default="true")
):
    """Transcription with optional enhancement"""
    # Rate limiting
    client_ip = get_client_ip(request)
    if not await rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # File validation
    validate_audio_file(file)
    
    enhance_bool = enhance_first.lower() in ['true', '1', 'yes']
    start_time = time.time()
    
    try:
        contents = await file.read()
        
        # Load audio
        audio = pipeline.audio_processor.load_audio(contents)
        
        # Optionally enhance
        if enhance_bool:
            audio = pipeline.enhance_audio(audio)
        
        # Transcribe
        result = pipeline.transcribe_audio(audio, language)
        
        duration = len(audio) / pipeline.audio_processor.sample_rate
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "transcript": result['text'].strip(),
            "duration": duration,
            "language": result.get('language', language),
            "segments": result.get('segments', []),
            "processing_time": round(processing_time, 2)
        }
    
    except Exception as e:
        logger.error(f"❌ Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech"""
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        temp_filename = f"tts_{int(time.time())}.wav"
        temp_path = Config.TEMP_DIR / temp_filename
        
        if not generate_tts(request.text, str(temp_path), request.language):
            raise HTTPException(
                status_code=500,
                detail="TTS failed. Install gtts."
            )
        
        return FileResponse(
            temp_path,
            media_type="audio/wav",
            filename=temp_filename
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ TTS error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download processed audio file"""
    if filename not in temp_files:
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    file_path = Path(temp_files[filename])
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename
    )


@app.delete("/cleanup/{filename}")
async def cleanup_file(filename: str):
    """Manually cleanup a temporary file"""
    if filename not in temp_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        file_path = Path(temp_files[filename])
        if file_path.exists():
            os.remove(file_path)
        del temp_files[filename]
        return {"success": True, "message": "File deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # HF Spaces uses port 7860
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )