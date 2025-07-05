from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import os
import uuid
import asyncio
import aiohttp
import logging
import subprocess
import json
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MoviePy Video Generator ADVANCED FIXED",
    description="Sistema completo para videos motivacionales con imágenes, voz y efectos - BUGFIXED",
    version="3.1.0"  # FIXED VERSION
)

# Configuration
VIDEO_OUTPUT_DIR = os.getenv("VIDEO_OUTPUT_DIR", "/app/videos")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", "600"))  # 10 minutes
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp")

# Create directories
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Pydantic models for ADVANCED video generation
class AdvancedScene(BaseModel):
    image_url: str = Field(..., description="URL de la imagen para esta escena")
    duration: float = Field(..., description="Duración de la escena en segundos")
    zoom_effect: Optional[Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "static"]] = Field("static", description="Efecto de zoom/pan")
    zoom_intensity: Optional[float] = Field(1.3, description="Intensidad del zoom (1.0 = sin zoom, 1.5 = zoom moderado)")
    overlay_text: Optional[str] = Field(None, description="Texto superpuesto en la imagen")
    text_position: Optional[Literal["top", "center", "bottom"]] = Field("bottom", description="Posición del texto")
    text_style: Optional[str] = Field("bold_white_shadow", description="Estilo del texto")
    text_size: Optional[int] = Field(60, description="Tamaño del texto")
    transition_duration: Optional[float] = Field(0.5, description="Duración de transición entre escenas")

class AdvancedVideoRequest(BaseModel):
    title: str = Field(..., description="Título del video")
    scenes: List[AdvancedScene] = Field(..., description="Lista de escenas del video")
    background_audio_url: Optional[str] = Field(None, description="URL del audio de fondo/voz en off")
    resolution: Optional[str] = Field("1920x1080", description="Resolución del video")
    fps: Optional[int] = Field(30, description="Frames por segundo")
    audio_volume: Optional[float] = Field(0.8, description="Volumen del audio (0.0 a 1.0)")
    fade_in_duration: Optional[float] = Field(0.5, description="Duración del fade in inicial")
    fade_out_duration: Optional[float] = Field(0.5, description="Duración del fade out final")

class VideoResponse(BaseModel):
    video_id: str
    video_url: str
    duration: float
    status: str
    message: Optional[str] = None

# In-memory storage for video status
video_status = {}

async def download_file(url: str, temp_dir: str, filename: str) -> Optional[str]:
    """Download file from URL"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    file_path = os.path.join(temp_dir, filename)
                    with open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    return file_path
                else:
                    logger.error(f"Failed to download {url}: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return None

def create_zoom_filter(effect: str, intensity: float, duration: float, width: int, height: int) -> str:
    """Create ffmpeg zoom/pan filter based on effect type"""
    
    if effect == "zoom_in":
        return f"zoompan=z='min(zoom+0.0015,{intensity})':d={int(duration*30)}:s={width}x{height}"
    elif effect == "zoom_out":
        return f"zoompan=z='max(zoom-0.0015,1.0)':d={int(duration*30)}:s={width}x{height}"
    elif effect == "pan_left":
        return f"zoompan=z='min(zoom+0.001,{intensity})':x='iw-iw/zoom':d={int(duration*30)}:s={width}x{height}"
    elif effect == "pan_right":
        return f"zoompan=z='min(zoom+0.001,{intensity})':x='0':d={int(duration*30)}:s={width}x{height}"
    else:  # static
        return f"scale={width}:{height}"

def create_text_filter(text: str, position: str, size: int, style: str, width: int, height: int) -> str:
    """Create ffmpeg text overlay filter"""
    
    # Escape text for ffmpeg
    text_escaped = text.replace("'", "\\''").replace(":", "\\:")
    
    # Position calculations
    if position == "top":
        y_pos = "50"
    elif position == "center":
        y_pos = f"(h-text_h)/2"
    else:  # bottom
        y_pos = f"h-text_h-50"
    
    # Style configurations
    if style == "bold_white_shadow":
        return f"drawtext=text='{text_escaped}':fontsize={size}:fontcolor=white:borderw=3:bordercolor=black:x=(w-text_w)/2:y={y_pos}:fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    elif style == "elegant_gold":
        return f"drawtext=text='{text_escaped}':fontsize={size}:fontcolor=#FFD700:borderw=2:bordercolor=black:x=(w-text_w)/2:y={y_pos}:fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    else:  # default
        return f"drawtext=text='{text_escaped}':fontsize={size}:fontcolor=white:x=(w-text_w)/2:y={y_pos}:fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def run_ffmpeg_command(cmd: List[str]) -> bool:
    """Run ffmpeg command with detailed error handling"""
    try:
        logger.info(f"Running ffmpeg command: {' '.join(cmd[:10])}...")  # Log first 10 args
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=600  # 10 minute timeout for complex videos
        )
        logger.info("FFmpeg command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg command timed out")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running ffmpeg: {e}")
        return False

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "moviepy-video-generator-advanced-fixed",
        "version": "3.1.0",
        "renderer": "Advanced FFmpeg with Effects - BUGFIXED",
        "features": ["images", "zoom_effects", "text_overlay", "voice_over"]
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "MoviePy Video Generator ADVANCED FIXED",
        "version": "3.1.0",
        "renderer": "Advanced FFmpeg with zoom, text, audio mixing - BUGFIXED",
        "max_duration": f"{MAX_VIDEO_DURATION}s",
        "features": [
            "Real images with zoom/pan effects",
            "Text overlays with custom styles", 
            "Voice-over audio mixing",
            "Professional transitions",
            "Ken Burns effects"
        ],
        "endpoints": {
            "generate_basic": "/generate-video",
            "generate_advanced": "/generate-advanced-video",
            "status": "/status/{video_id}",
            "download": "/videos/{video_id}",
            "health": "/health"
        }
    }

async def create_advanced_video(request: AdvancedVideoRequest, video_id: str) -> str:
    """ADVANCED video creation with images, zoom effects, text overlays and audio - FIXED"""
    temp_dir = None
    
    try:
        # Parse resolution
        width, height = map(int, request.resolution.split('x'))
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"advanced_video_{video_id}_")
        logger.info(f"Creating ADVANCED video {video_id} with {len(request.scenes)} scenes")
        
        # Update status
        video_status[video_id] = {"status": "processing", "progress": 5}
        
        # STEP 1: Download all images
        logger.info("Downloading images...")
        image_files = []
        
        for i, scene in enumerate(request.scenes):
            image_filename = f"image_{i:03d}.jpg"
            image_path = await download_file(scene.image_url, temp_dir, image_filename)
            
            if not image_path:
                raise Exception(f"Failed to download image {i}: {scene.image_url}")
            
            image_files.append(image_path)
            
            # Update progress
            progress = 5 + (i + 1) * 15 // len(request.scenes)
            video_status[video_id] = {"status": "processing", "progress": progress}
        
        # STEP 2: Download audio if provided
        audio_path = None
        if request.background_audio_url:
            logger.info("Downloading background audio...")
            audio_path = await download_file(request.background_audio_url, temp_dir, "background_audio.mp3")
            
            if not audio_path:
                logger.warning("Failed to download audio, continuing without it")
        
        video_status[video_id] = {"status": "processing", "progress": 25}
        
        # STEP 3: Create complex ffmpeg filter chain - FIXED VERSION
        logger.info("Building ffmpeg filter chain...")
        
        # Build input parameters - FIXED
        ffmpeg_inputs = []
        filter_complex = []
        
        # Add all images as inputs - FIXED INDEXING
        for i, (image_path, scene) in enumerate(zip(image_files, request.scenes)):
            ffmpeg_inputs.extend(["-loop", "1", "-t", str(scene.duration), "-i", image_path])
        
        # Add audio input if available
        audio_input_index = len(image_files)
        if audio_path:
            ffmpeg_inputs.extend(["-i", audio_path])
        
        # Create filter for each scene - FIXED
        for i, scene in enumerate(request.scenes):
            # Create zoom/pan effect
            zoom_filter = create_zoom_filter(
                scene.zoom_effect, 
                scene.zoom_intensity, 
                scene.duration, 
                width, 
                height
            )
            
            # Add text overlay if specified
            if scene.overlay_text:
                text_filter = create_text_filter(
                    scene.overlay_text,
                    scene.text_position,
                    scene.text_size,
                    scene.text_style,
                    width,
                    height
                )
                full_filter = f"[{i}:v]{zoom_filter},{text_filter}[v{i}]"
            else:
                full_filter = f"[{i}:v]{zoom_filter}[v{i}]"
            
            filter_complex.append(full_filter)
        
        # Concatenate all video streams
        video_inputs = "".join([f"[v{i}]" for i in range(len(request.scenes))])
        concat_filter = f"{video_inputs}concat=n={len(request.scenes)}:v=1:a=0[vout]"
        filter_complex.append(concat_filter)
        
        # Join all filters
        filter_string = ";".join(filter_complex)
        
        video_status[video_id] = {"status": "processing", "progress": 45}
        
        # STEP 4: Build final ffmpeg command
        output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{video_id}.mp4")
        
        ffmpeg_cmd = ["ffmpeg", "-y"] + ffmpeg_inputs + [
            "-filter_complex", filter_string,
            "-map", "[vout]"
        ]
        
        # Add audio mapping if available
        if audio_path:
            ffmpeg_cmd.extend([
                "-map", f"{audio_input_index}:a",
                "-c:a", "aac",
                "-filter:a", f"volume={request.audio_volume}",
                "-shortest"  # Stop when shortest stream ends
            ])
        
        # Add video encoding settings
        ffmpeg_cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "23",  # Good quality
            "-r", str(request.fps),
            output_path
        ])
        
        video_status[video_id] = {"status": "processing", "progress": 60}
        
        # STEP 5: Execute ffmpeg command
        logger.info("Executing advanced ffmpeg command...")
        
        if not run_ffmpeg_command(ffmpeg_cmd):
            raise Exception("Failed to create advanced video with ffmpeg")
        
        video_status[video_id] = {"status": "processing", "progress": 95}
        
        # Calculate total duration
        total_duration = sum(scene.duration for scene in request.scenes)
        
        # Final status update
        video_status[video_id] = {
            "status": "completed",
            "progress": 100,
            "duration": total_duration,
            "scenes": len(request.scenes),
            "has_audio": audio_path is not None
        }
        
        logger.info(f"ADVANCED video {video_id} created successfully! Scenes: {len(request.scenes)}, Duration: {total_duration}s")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating advanced video {video_id}: {str(e)}")
        video_status[video_id] = {
            "status": "error",
            "progress": 0,
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=f"Error creating advanced video: {str(e)}")
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temp dir: {str(e)}")

@app.post("/generate-advanced-video", response_model=VideoResponse)
async def generate_advanced_video(request: AdvancedVideoRequest, background_tasks: BackgroundTasks):
    """Generate an ADVANCED video with images, zoom effects, text overlays and audio"""
    # Validate request
    if not request.scenes:
        raise HTTPException(status_code=400, detail="At least one scene is required")
    
    total_duration = sum(scene.duration for scene in request.scenes)
    if total_duration > MAX_VIDEO_DURATION:
        raise HTTPException(
            status_code=400, 
            detail=f"Total video duration ({total_duration}s) exceeds maximum ({MAX_VIDEO_DURATION}s)"
        )
    
    # Validate image URLs
    for i, scene in enumerate(request.scenes):
        if not scene.image_url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=400,
                detail=f"Scene {i}: Invalid image URL. Must start with http:// or https://"
            )
    
    # Generate unique video ID
    video_id = f"advanced_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"
    
    # Initialize status
    video_status[video_id] = {"status": "queued", "progress": 0}
    
    # Start ADVANCED video creation in background
    background_tasks.add_task(create_advanced_video, request, video_id)
    
    return VideoResponse(
        video_id=video_id,
        video_url=f"/videos/{video_id}",
        duration=total_duration,
        status="queued",
        message=f"ADVANCED video generation started - {len(request.scenes)} scenes with effects v3.1 FIXED"
    )

# Include all existing endpoints from the optimized version
@app.get("/status/{video_id}")
async def get_video_status(video_id: str):
    """Get the status of a video generation"""
    if video_id not in video_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return video_status[video_id]

@app.get("/videos/{video_id}")
async def download_video(video_id: str):
    """Download the generated video"""
    video_path = os.path.join(VIDEO_OUTPUT_DIR, f"{video_id}.mp4")
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{video_id}.mp4"
    )

@app.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a generated video"""
    video_path = os.path.join(VIDEO_OUTPUT_DIR, f"{video_id}.mp4")
    
    if os.path.exists(video_path):
        os.remove(video_path)
    
    if video_id in video_status:
        del video_status[video_id]
    
    return {"message": f"Video {video_id} deleted successfully"}

@app.get("/videos")
async def list_videos():
    """List all available videos"""
    videos = []
    
    if os.path.exists(VIDEO_OUTPUT_DIR):
        for filename in os.listdir(VIDEO_OUTPUT_DIR):
            if filename.endswith('.mp4'):
                video_id = filename[:-4]
                file_path = os.path.join(VIDEO_OUTPUT_DIR, filename)
                file_size = os.path.getsize(file_path)
                
                videos.append({
                    "video_id": video_id,
                    "filename": filename,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "url": f"/videos/{video_id}"
                })
    
    return {"videos": videos, "count": len(videos)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)