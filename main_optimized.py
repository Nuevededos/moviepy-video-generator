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
import subprocess
import json
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap

# MoviePy imports - SOLO para crear frames individuales
from moviepy.editor import AudioFileClip

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MoviePy Video Generator OPTIMIZED",
    description="Servicio híbrido MoviePy + FFmpeg para videos eficientes",
    version="2.0.0"  # OPTIMIZED VERSION
)

# Configuration
VIDEO_OUTPUT_DIR = os.getenv("VIDEO_OUTPUT_DIR", "/app/videos")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", "600"))  # 10 minutes now!
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp")

# Create directories
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Pydantic models (IDENTICAL API)
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
    
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def create_image_file(clip_data: VideoClip, resolution: tuple, output_path: str) -> bool:
    """Create image file directly using PIL - OPTIMIZED"""
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
            if line.strip():
                # Calculate line width for centering
                if font:
                    try:
                        bbox = draw.textbbox((0, 0), line, font=font)
                        line_width = bbox[2] - bbox[0]
                    except:
                        line_width = len(line) * (clip_data.font_size // 2)
                else:
                    line_width = len(line) * (clip_data.font_size // 2)
                
                x = (width - line_width) // 2
                
                try:
                    if font:
                        draw.text((x, current_y), line, fill=text_color, font=font)
                    else:
                        draw.text((x, current_y), line, fill=text_color)
                except Exception as e:
                    logger.warning(f"Error drawing line '{line}': {e}")
            
            current_y += line_height
        
        # Save image as PNG
        img.save(output_path, 'PNG', quality=95)
        logger.info(f"Image saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating image: {e}")
        return False

def run_ffmpeg_command(cmd: List[str]) -> bool:
    """Run ffmpeg command with error handling"""
    try:
        logger.info(f"Running ffmpeg: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
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
        "service": "moviepy-video-generator-optimized",
        "version": "2.0.0",
        "renderer": "PIL + FFmpeg Hybrid"
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "MoviePy Video Generator OPTIMIZED",
        "version": "2.0.0",
        "renderer": "PIL + FFmpeg Hybrid (5-10x faster)",
        "max_duration": f"{MAX_VIDEO_DURATION}s",
        "optimization": "Direct ffmpeg rendering",
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

async def create_video_optimized(request: VideoRequest, video_id: str) -> str:
    """OPTIMIZED video creation using PIL + FFmpeg hybrid approach"""
    temp_dir = None
    
    try:
        # Parse resolution
        width, height = map(int, request.resolution.split('x'))
        resolution = (width, height)
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"video_{video_id}_")
        logger.info(f"Creating OPTIMIZED video {video_id} with resolution {resolution}")
        
        # Update status
        video_status[video_id] = {"status": "processing", "progress": 10}
        
        # STEP 1: Create individual images using PIL
        image_files = []
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")
        
        with open(concat_list_path, 'w') as concat_file:
            for i, clip_data in enumerate(request.clips):
                logger.info(f"Creating image {i+1}/{len(request.clips)}: '{clip_data.text[:30]}...'")
                
                # Create image file
                image_path = os.path.join(temp_dir, f"clip_{i:03d}.png")
                
                if create_image_file(clip_data, resolution, image_path):
                    image_files.append((image_path, clip_data.duration))
                    
                    # Add to ffmpeg concat list
                    concat_file.write(f"file '{image_path}'\n")
                    concat_file.write(f"duration {clip_data.duration}\n")
                else:
                    raise Exception(f"Failed to create image for clip {i}")
                
                # Update progress
                progress = 10 + (i + 1) * 40 // len(request.clips)
                video_status[video_id] = {"status": "processing", "progress": progress}
        
        # Add final image for last frame (ffmpeg concat requirement)
        if image_files:
            final_image_path = image_files[-1][0]
            with open(concat_list_path, 'a') as concat_file:
                concat_file.write(f"file '{final_image_path}'\n")
        
        # Update status
        video_status[video_id] = {"status": "processing", "progress": 55}
        
        # STEP 2: Create video using FFmpeg directly
        temp_video_path = os.path.join(temp_dir, f"temp_video_{video_id}.mp4")
        
        # FFmpeg command for creating video from images
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-vf", f"fps={request.fps},scale={width}:{height}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",  # Balance between speed and quality
            temp_video_path
        ]
        
        if not run_ffmpeg_command(ffmpeg_cmd):
            raise Exception("Failed to create video with FFmpeg")
        
        # Update status
        video_status[video_id] = {"status": "processing", "progress": 75}
        
        # STEP 3: Add audio if provided
        final_output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{video_id}.mp4")
        
        if request.background_music_url:
            logger.info("Adding background music with FFmpeg...")
            audio_path = await download_audio(request.background_music_url, temp_dir)
            
            if audio_path and os.path.exists(audio_path):
                # FFmpeg command to add audio
                ffmpeg_audio_cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_video_path,
                    "-i", audio_path,
                    "-c:v", "copy",  # Don't re-encode video
                    "-c:a", "aac",
                    "-filter:a", f"volume={request.music_volume}",
                    "-shortest",  # Stop when shortest input ends
                    final_output_path
                ]
                
                if not run_ffmpeg_command(ffmpeg_audio_cmd):
                    # Fallback: copy video without audio
                    shutil.copy2(temp_video_path, final_output_path)
            else:
                # No audio, just copy the video
                shutil.copy2(temp_video_path, final_output_path)
        else:
            # No audio requested, copy the video
            shutil.copy2(temp_video_path, final_output_path)
        
        # Update status
        video_status[video_id] = {"status": "processing", "progress": 95}
        
        # Get final video duration
        total_duration = sum(clip.duration for clip in request.clips)
        
        # Final status update
        video_status[video_id] = {
            "status": "completed",
            "progress": 100,
            "duration": total_duration
        }
        
        logger.info(f"OPTIMIZED video {video_id} created successfully! Duration: {total_duration}s")
        return final_output_path
        
    except Exception as e:
        logger.error(f"Error creating optimized video {video_id}: {str(e)}")
        video_status[video_id] = {
            "status": "error",
            "progress": 0,
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=f"Error creating video: {str(e)}")
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temp dir: {str(e)}")

@app.post("/generate-video", response_model=VideoResponse)
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """Generate a video from text clips - OPTIMIZED VERSION"""
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
    
    # Start OPTIMIZED video creation in background
    background_tasks.add_task(create_video_optimized, request, video_id)
    
    return VideoResponse(
        video_id=video_id,
        video_url=f"/videos/{video_id}",
        duration=total_duration,
        status="queued",
        message="OPTIMIZED video generation started - PIL + FFmpeg Hybrid v2.0"
    )

# Keep all other endpoints IDENTICAL
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