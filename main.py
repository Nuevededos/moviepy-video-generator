from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import uuid
import asyncio
import aiohttp
import logging
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap

# MoviePy imports
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip,
    concatenate_videoclips, ColorClip
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MoviePy Video Generator",
    description="Servicio gratuito para generar videos motivacionales con MoviePy + PIL",
    version="1.2.0"  # Simplified approach
)

# Configuration
VIDEO_OUTPUT_DIR = os.getenv("VIDEO_OUTPUT_DIR", "/app/videos")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", "300"))  # 5 minutes
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp")

# Create directories
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Pydantic models
class VideoClip(BaseModel):
    text: str = Field(..., description="Texto a mostrar en el clip")
    duration: float = Field(..., description="Duración del clip en segundos")
    background_color: Optional[str] = Field("#009c8c", description="Color de fondo del clip")
    text_color: Optional[str] = Field("#ffffff", description="Color del texto")
    font_size: Optional[int] = Field(60, description="Tamaño de fuente")
    position: Optional[str] = Field("center", description="Posición del texto")

class VideoRequest(BaseModel):
    title: str = Field(..., description="Título del video")
    clips: List[VideoClip] = Field(..., description="Lista de clips de texto")
    background_music_url: Optional[str] = Field(None, description="URL de música de fondo")
    resolution: Optional[str] = Field("1920x1080", description="Resolución del video")
    fps: Optional[int] = Field(30, description="Frames por segundo")
    transition_duration: Optional[float] = Field(0.5, description="Duración de transiciones")
    music_volume: Optional[float] = Field(0.3, description="Volumen de música (0.0 a 1.0)")

class VideoResponse(BaseModel):
    video_id: str
    video_url: str
    duration: float
    status: str
    message: Optional[str] = None

# In-memory storage for video status
video_status = {}

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_font(font_size: int):
    """Get font with fallback system"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)
        except Exception:
            continue
    
    # Fallback to default font
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def create_complete_frame(clip_data: VideoClip, resolution: tuple) -> np.ndarray:
    """Create complete frame with background + text using PIL - SIMPLIFIED"""
    width, height = resolution
    
    try:
        # Create RGB image with background color
        bg_color = hex_to_rgb(clip_data.background_color)
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Get font
        font = get_font(clip_data.font_size)
        
        # Text color
        text_color = hex_to_rgb(clip_data.text_color)
        
        # Word wrap
        max_chars = max(15, min(40, width // (clip_data.font_size // 3)))
        wrapped_text = textwrap.fill(clip_data.text, width=max_chars)
        lines = wrapped_text.split('\n')
        
        # Calculate total text height
        line_height = clip_data.font_size + 10
        total_text_height = len(lines) * line_height
        
        # Calculate starting Y position
        if clip_data.position == "center":
            start_y = (height - total_text_height) // 2
        elif clip_data.position == "top":
            start_y = 50
        elif clip_data.position == "bottom":
            start_y = height - total_text_height - 50
        else:
            start_y = (height - total_text_height) // 2
        
        # Draw text lines
        current_y = start_y
        for line in lines:
            if line.strip():  # Skip empty lines
                # Calculate line width for centering
                if font:
                    try:
                        bbox = draw.textbbox((0, 0), line, font=font)
                        line_width = bbox[2] - bbox[0]
                    except:
                        line_width = len(line) * (clip_data.font_size // 2)
                else:
                    line_width = len(line) * (clip_data.font_size // 2)
                
                x = (width - line_width) // 2  # Center horizontally
                
                # Draw text
                try:
                    if font:
                        draw.text((x, current_y), line, fill=text_color, font=font)
                    else:
                        draw.text((x, current_y), line, fill=text_color)
                except Exception as e:
                    logger.warning(f"Error drawing line '{line}': {e}")
            
            current_y += line_height
        
        # Convert to numpy array - EXPLICIT RGB
        img_array = np.array(img, dtype=np.uint8)
        
        # Verify shape
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            logger.error(f"Wrong shape: {img_array.shape}")
            # Force correct shape
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            img_array[:, :] = bg_color
        
        logger.info(f"Frame created successfully: {img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"Error creating frame: {e}")
        # Ultimate fallback - solid color array
        bg_color = hex_to_rgb(clip_data.background_color)
        fallback_array = np.full((height, width, 3), bg_color, dtype=np.uint8)
        return fallback_array

def create_video_clip_simple(clip_data: VideoClip, resolution: tuple) -> ImageClip:
    """Create video clip - SIMPLIFIED approach"""
    try:
        # Create complete frame (background + text)
        frame_array = create_complete_frame(clip_data, resolution)
        
        # Create ImageClip directly
        clip = ImageClip(frame_array, duration=clip_data.duration)
        
        logger.info(f"Video clip created: {frame_array.shape}, duration: {clip_data.duration}s")
        return clip
        
    except Exception as e:
        logger.error(f"Error creating video clip: {e}")
        # Solid color fallback
        width, height = resolution
        bg_color = hex_to_rgb(clip_data.background_color)
        solid_array = np.full((height, width, 3), bg_color, dtype=np.uint8)
        return ImageClip(solid_array, duration=clip_data.duration)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "moviepy-video-generator",
        "version": "1.2.0",
        "renderer": "PIL-Simplified"
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "MoviePy Video Generator API",
        "version": "1.2.0",
        "renderer": "PIL Simplified (Single frame approach)",
        "endpoints": {
            "generate": "/generate-video",
            "status": "/status/{video_id}",
            "download": "/videos/{video_id}",
            "health": "/health"
        }
    }

async def download_audio(url: str, temp_dir: str) -> Optional[str]:
    """Download audio file from URL"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    audio_path = os.path.join(temp_dir, f"audio_{uuid.uuid4().hex}.mp3")
                    with open(audio_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    return audio_path
                else:
                    logger.error(f"Failed to download audio: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error downloading audio: {str(e)}")
        return None

async def create_video_from_clips(request: VideoRequest, video_id: str) -> str:
    """Main function to create video from clips - SIMPLIFIED"""
    temp_dir = None
    final_video = None
    try:
        # Parse resolution
        width, height = map(int, request.resolution.split('x'))
        resolution = (width, height)
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"video_{video_id}_")
        logger.info(f"Creating video {video_id} with resolution {resolution} - Simplified PIL v1.2.0")
        
        # Update status
        video_status[video_id] = {"status": "processing", "progress": 10}
        
        # Create video clips - SIMPLIFIED
        video_clips = []
        
        for i, clip_data in enumerate(request.clips):
            logger.info(f"Processing clip {i+1}/{len(request.clips)}: '{clip_data.text[:30]}...'")
            
            # Create complete video clip (background + text in one)
            video_clip = create_video_clip_simple(clip_data, resolution)
            video_clips.append(video_clip)
            
            # Update progress
            progress = 10 + (i + 1) * 50 // len(request.clips)
            video_status[video_id] = {"status": "processing", "progress": progress}
        
        # Concatenate all clips
        logger.info("Concatenating video clips...")
        final_video = concatenate_videoclips(video_clips, method="compose")
        
        # Update status
        video_status[video_id] = {"status": "processing", "progress": 70}
        
        # Add background music if provided
        background_audio = None
        if request.background_music_url:
            logger.info("Downloading and adding background music...")
            audio_path = await download_audio(request.background_music_url, temp_dir)
            
            if audio_path and os.path.exists(audio_path):
                try:
                    background_audio = AudioFileClip(audio_path)
                    
                    # Adjust audio duration to match video
                    if background_audio.duration > final_video.duration:
                        background_audio = background_audio.subclip(0, final_video.duration)
                    else:
                        # Loop audio if shorter than video
                        loops_needed = int(final_video.duration / background_audio.duration) + 1
                        background_audio = concatenate_videoclips([background_audio] * loops_needed)
                        background_audio = background_audio.subclip(0, final_video.duration)
                    
                    # Set volume
                    background_audio = background_audio.volumex(request.music_volume)
                    
                    # Add audio to video
                    final_video = final_video.set_audio(background_audio)
                    
                except Exception as e:
                    logger.error(f"Error adding background music: {str(e)}")
        
        # Update status
        video_status[video_id] = {"status": "processing", "progress": 85}
        
        # Export video
        output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{video_id}.mp4")
        logger.info(f"Exporting video to {output_path}...")
        
        final_video.write_videofile(
            output_path,
            fps=request.fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=os.path.join(temp_dir, f"temp_audio_{video_id}.m4a"),
            remove_temp=True,
            verbose=False,
            logger=None  # Disable moviepy logger
        )
        
        # Update status
        video_status[video_id] = {
            "status": "completed",
            "progress": 100,
            "duration": final_video.duration
        }
        
        logger.info(f"Video {video_id} created successfully - Simplified PIL v1.2.0")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating video {video_id}: {str(e)}")
        video_status[video_id] = {
            "status": "error",
            "progress": 0,
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=f"Error creating video: {str(e)}")
    
    finally:
        # Clean up resources
        try:
            if final_video:
                final_video.close()
            if 'background_audio' in locals() and background_audio:
                background_audio.close()
            # Clean up individual clips
            if 'video_clips' in locals():
                for clip in video_clips:
                    if hasattr(clip, 'close'):
                        clip.close()
        except Exception as e:
            logger.error(f"Error cleaning up video resources: {str(e)}")
        
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temp dir: {str(e)}")

@app.post("/generate-video", response_model=VideoResponse)
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """Generate a video from text clips"""
    # Validate request
    if not request.clips:
        raise HTTPException(status_code=400, detail="At least one clip is required")
    
    total_duration = sum(clip.duration for clip in request.clips)
    if total_duration > MAX_VIDEO_DURATION:
        raise HTTPException(
            status_code=400, 
            detail=f"Total video duration ({total_duration}s) exceeds maximum ({MAX_VIDEO_DURATION}s)"
        )
    
    # Generate unique video ID
    video_id = f"video_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"
    
    # Initialize status
    video_status[video_id] = {"status": "queued", "progress": 0}
    
    # Start video creation in background
    background_tasks.add_task(create_video_from_clips, request, video_id)
    
    return VideoResponse(
        video_id=video_id,
        video_url=f"/videos/{video_id}",
        duration=total_duration,
        status="queued",
        message="Video generation started - Simplified PIL v1.2.0"
    )

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
                video_id = filename[:-4]  # Remove .mp4 extension
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