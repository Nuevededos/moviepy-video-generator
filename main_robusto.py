from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import os
import uuid
import asyncio
import aiohttp
import logging
import subprocess
import tempfile
import shutil
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MoviePy Video Generator ROBUSTO",
    description="Sistema completo y robusto para videos motivacionales",
    version="4.0.0"
)

# Configuration
VIDEO_OUTPUT_DIR = os.getenv("VIDEO_OUTPUT_DIR", "/app/videos")
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", "600"))
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp")

# Create directories
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# PERSISTENT storage for video status (file-based)
STATUS_FILE = os.path.join(VIDEO_OUTPUT_DIR, "video_status.json")

def load_video_status():
    """Load video status from file"""
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading status: {e}")
    return {}

def save_video_status(status_dict):
    """Save video status to file"""
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(status_dict, f)
    except Exception as e:
        logger.error(f"Error saving status: {e}")

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
    music_volume: Optional[float] = Field(0.3, description="Volumen de música (0.0 a 1.0)")

class AdvancedScene(BaseModel):
    image_url: str = Field(..., description="URL de la imagen para esta escena")
    duration: float = Field(..., description="Duración de la escena en segundos")
    zoom_effect: Optional[Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "static"]] = Field("static")
    zoom_intensity: Optional[float] = Field(1.3, description="Intensidad del zoom")
    overlay_text: Optional[str] = Field(None, description="Texto superpuesto")
    text_position: Optional[Literal["top", "center", "bottom"]] = Field("bottom")
    text_size: Optional[int] = Field(60, description="Tamaño del texto")

class AdvancedVideoRequest(BaseModel):
    title: str = Field(..., description="Título del video")
    scenes: List[AdvancedScene] = Field(..., description="Lista de escenas del video")
    background_audio_url: Optional[str] = Field(None, description="URL del audio de fondo")
    resolution: Optional[str] = Field("1920x1080", description="Resolución del video")
    fps: Optional[int] = Field(30, description="Frames por segundo")
    audio_volume: Optional[float] = Field(0.8, description="Volumen del audio")

class VideoResponse(BaseModel):
    video_id: str
    video_url: str
    duration: float
    status: str
    message: Optional[str] = None

async def download_file(url: str, temp_dir: str, filename: str) -> Optional[str]:
    """Download file from URL with retries"""
    for attempt in range(3):
        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        file_path = os.path.join(temp_dir, filename)
                        with open(file_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        logger.info(f"Downloaded: {filename}")
                        return file_path
                    else:
                        logger.error(f"Download failed: {url} - Status: {response.status}")
        except Exception as e:
            logger.error(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                await asyncio.sleep(2)
    return None

def run_ffmpeg_command(cmd: List[str]) -> bool:
    """Run ffmpeg command with detailed logging"""
    try:
        logger.info(f"Executing FFmpeg command...")
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=900  # 15 minute timeout
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
        logger.error(f"Unexpected FFmpeg error: {e}")
        return False

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "moviepy-video-generator-robusto",
        "version": "4.0.0",
        "renderer": "ROBUSTO - FFmpeg + Persistent Storage"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MoviePy Video Generator ROBUSTO",
        "version": "4.0.0",
        "renderer": "Sistema robusto con FFmpeg optimizado",
        "max_duration": f"{MAX_VIDEO_DURATION}s",
        "features": [
            "Videos básicos optimizados 5-10x más rápidos",
            "Videos avanzados con imágenes reales", 
            "Efectos zoom/pan profesionales",
            "Audio/voz en off integrado",
            "Storage persistente de estados"
        ],
        "endpoints": {
            "generate_basic": "/generate-video",
            "generate_advanced": "/generate-advanced-video",
            "status": "/status/{video_id}",
            "download": "/videos/{video_id}",
            "health": "/health"
        }
    }

async def create_basic_video(request: VideoRequest, video_id: str) -> str:
    """Create basic optimized video"""
    temp_dir = None
    status_dict = load_video_status()
    
    try:
        width, height = map(int, request.resolution.split('x'))
        temp_dir = tempfile.mkdtemp(prefix=f"basic_video_{video_id}_")
        
        status_dict[video_id] = {"status": "processing", "progress": 10}
        save_video_status(status_dict)
        
        # Create concat file for FFmpeg
        concat_file = os.path.join(temp_dir, "concat_list.txt")
        
        with open(concat_file, 'w') as f:
            for i, clip in enumerate(request.clips):
                # Create solid color video for each clip
                color_hex = clip.background_color.lstrip('#')
                
                # FFmpeg command to create colored clip with text
                clip_output = os.path.join(temp_dir, f"clip_{i:03d}.mp4")
                
                text_escaped = clip.text.replace("'", "\\''").replace(":", "\\:")
                
                ffmpeg_clip_cmd = [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", f"color=c=#{color_hex}:size={width}x{height}:duration={clip.duration}:rate={request.fps}",
                    "-vf", f"drawtext=text='{text_escaped}':fontsize={clip.font_size}:fontcolor={clip.text_color}:x=(w-text_w)/2:y=(h-text_h)/2:fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    clip_output
                ]
                
                if not run_ffmpeg_command(ffmpeg_clip_cmd):
                    raise Exception(f"Failed to create clip {i}")
                
                f.write(f"file '{clip_output}'\n")
                
                progress = 10 + (i + 1) * 40 // len(request.clips)
                status_dict[video_id] = {"status": "processing", "progress": progress}
                save_video_status(status_dict)
        
        # Concatenate all clips
        temp_video = os.path.join(temp_dir, "temp_video.mp4")
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy",
            temp_video
        ]
        
        if not run_ffmpeg_command(concat_cmd):
            raise Exception("Failed to concatenate clips")
        
        status_dict[video_id] = {"status": "processing", "progress": 70}
        save_video_status(status_dict)
        
        # Add audio if provided
        final_output = os.path.join(VIDEO_OUTPUT_DIR, f"{video_id}.mp4")
        
        if request.background_music_url:
            audio_path = await download_file(request.background_music_url, temp_dir, "audio.mp3")
            
            if audio_path:
                audio_cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_video,
                    "-i", audio_path,
                    "-c:v", "copy", "-c:a", "aac",
                    "-filter:a", f"volume={request.music_volume}",
                    "-shortest",
                    final_output
                ]
                
                if not run_ffmpeg_command(audio_cmd):
                    shutil.copy2(temp_video, final_output)
            else:
                shutil.copy2(temp_video, final_output)
        else:
            shutil.copy2(temp_video, final_output)
        
        total_duration = sum(clip.duration for clip in request.clips)
        
        status_dict[video_id] = {
            "status": "completed",
            "progress": 100,
            "duration": total_duration
        }
        save_video_status(status_dict)
        
        logger.info(f"Basic video {video_id} completed successfully")
        return final_output
        
    except Exception as e:
        logger.error(f"Error creating basic video {video_id}: {str(e)}")
        status_dict[video_id] = {"status": "error", "progress": 0, "error": str(e)}
        save_video_status(status_dict)
        raise HTTPException(status_code=500, detail=f"Error creating video: {str(e)}")
    
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temp dir: {str(e)}")

@app.post("/generate-video", response_model=VideoResponse)
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """Generate basic optimized video"""
    if not request.clips:
        raise HTTPException(status_code=400, detail="At least one clip is required")
    
    total_duration = sum(clip.duration for clip in request.clips)
    if total_duration > MAX_VIDEO_DURATION:
        raise HTTPException(
            status_code=400, 
            detail=f"Total video duration ({total_duration}s) exceeds maximum ({MAX_VIDEO_DURATION}s)"
        )
    
    video_id = f"basic_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"
    
    status_dict = load_video_status()
    status_dict[video_id] = {"status": "queued", "progress": 0}
    save_video_status(status_dict)
    
    background_tasks.add_task(create_basic_video, request, video_id)
    
    return VideoResponse(
        video_id=video_id,
        video_url=f"/videos/{video_id}",
        duration=total_duration,
        status="queued",
        message="ROBUSTO basic video generation started v4.0"
    )

@app.post("/generate-advanced-video", response_model=VideoResponse)
async def generate_advanced_video(request: AdvancedVideoRequest, background_tasks: BackgroundTasks):
    """Generate advanced video with images and effects"""
    if not request.scenes:
        raise HTTPException(status_code=400, detail="At least one scene is required")
    
    total_duration = sum(scene.duration for scene in request.scenes)
    if total_duration > MAX_VIDEO_DURATION:
        raise HTTPException(
            status_code=400, 
            detail=f"Total video duration ({total_duration}s) exceeds maximum ({MAX_VIDEO_DURATION}s)"
        )
    
    video_id = f"advanced_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"
    
    status_dict = load_video_status()
    status_dict[video_id] = {"status": "queued", "progress": 0}
    save_video_status(status_dict)
    
    background_tasks.add_task(create_advanced_video_robusto, request, video_id)
    
    return VideoResponse(
        video_id=video_id,
        video_url=f"/videos/{video_id}",
        duration=total_duration,
        status="queued",
        message="ROBUSTO advanced video generation started v4.0"
    )

async def create_advanced_video_robusto(request: AdvancedVideoRequest, video_id: str) -> str:
    """Create advanced video with robust error handling"""
    temp_dir = None
    status_dict = load_video_status()
    
    try:
        width, height = map(int, request.resolution.split('x'))
        temp_dir = tempfile.mkdtemp(prefix=f"advanced_video_{video_id}_")
        
        status_dict[video_id] = {"status": "processing", "progress": 5}
        save_video_status(status_dict)
        
        # Download all images
        image_files = []
        for i, scene in enumerate(request.scenes):
            image_filename = f"image_{i:03d}.jpg"
            image_path = await download_file(scene.image_url, temp_dir, image_filename)
            
            if not image_path:
                raise Exception(f"Failed to download image {i}: {scene.image_url}")
            
            image_files.append(image_path)
            
            progress = 5 + (i + 1) * 20 // len(request.scenes)
            status_dict[video_id] = {"status": "processing", "progress": progress}
            save_video_status(status_dict)
        
        # Download audio if provided
        audio_path = None
        if request.background_audio_url:
            audio_path = await download_file(request.background_audio_url, temp_dir, "audio.mp3")
        
        status_dict[video_id] = {"status": "processing", "progress": 30}
        save_video_status(status_dict)
        
        # Create video clips for each scene
        clip_files = []
        for i, (scene, image_path) in enumerate(zip(request.scenes, image_files)):
            clip_output = os.path.join(temp_dir, f"scene_{i:03d}.mp4")
            
            # Create zoom filter
            if scene.zoom_effect == "zoom_in":
                zoom_filter = f"zoompan=z='min(zoom+0.0015,{scene.zoom_intensity})':d={int(scene.duration*request.fps)}:s={width}x{height}"
            elif scene.zoom_effect == "zoom_out":
                zoom_filter = f"zoompan=z='max(zoom-0.0015,1.0)':d={int(scene.duration*request.fps)}:s={width}x{height}"
            elif scene.zoom_effect == "pan_right":
                zoom_filter = f"zoompan=z='min(zoom+0.001,{scene.zoom_intensity})':x='0':d={int(scene.duration*request.fps)}:s={width}x{height}"
            elif scene.zoom_effect == "pan_left":
                zoom_filter = f"zoompan=z='min(zoom+0.001,{scene.zoom_intensity})':x='iw-iw/zoom':d={int(scene.duration*request.fps)}:s={width}x{height}"
            else:
                zoom_filter = f"scale={width}:{height}"
            
            # Build ffmpeg command for this scene
            scene_cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-t", str(scene.duration), "-i", image_path,
                "-vf", zoom_filter,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-r", str(request.fps),
                clip_output
            ]
            
            # Add text overlay if specified
            if scene.overlay_text:
                text_escaped = scene.overlay_text.replace("'", "\\''").replace(":", "\\:")
                
                if scene.text_position == "top":
                    y_pos = "50"
                elif scene.text_position == "center":
                    y_pos = "(h-text_h)/2"
                else:
                    y_pos = "h-text_h-50"
                
                text_filter = f"drawtext=text='{text_escaped}':fontsize={scene.text_size}:fontcolor=white:borderw=3:bordercolor=black:x=(w-text_w)/2:y={y_pos}:fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                
                # Update video filter to include text
                scene_cmd[scene_cmd.index("-vf") + 1] = f"{zoom_filter},{text_filter}"
            
            if not run_ffmpeg_command(scene_cmd):
                raise Exception(f"Failed to create scene {i}")
            
            clip_files.append(clip_output)
            
            progress = 30 + (i + 1) * 40 // len(request.scenes)
            status_dict[video_id] = {"status": "processing", "progress": progress}
            save_video_status(status_dict)
        
        # Concatenate all scenes
        concat_file = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_file, 'w') as f:
            for clip_file in clip_files:
                f.write(f"file '{clip_file}'\n")
        
        temp_video = os.path.join(temp_dir, "temp_video.mp4")
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy",
            temp_video
        ]
        
        if not run_ffmpeg_command(concat_cmd):
            raise Exception("Failed to concatenate scenes")
        
        status_dict[video_id] = {"status": "processing", "progress": 80}
        save_video_status(status_dict)
        
        # Add audio if provided
        final_output = os.path.join(VIDEO_OUTPUT_DIR, f"{video_id}.mp4")
        
        if audio_path:
            audio_cmd = [
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac",
                "-filter:a", f"volume={request.audio_volume}",
                "-shortest",
                final_output
            ]
            
            if not run_ffmpeg_command(audio_cmd):
                shutil.copy2(temp_video, final_output)
        else:
            shutil.copy2(temp_video, final_output)
        
        total_duration = sum(scene.duration for scene in request.scenes)
        
        status_dict[video_id] = {
            "status": "completed",
            "progress": 100,
            "duration": total_duration,
            "scenes": len(request.scenes),
            "has_audio": audio_path is not None
        }
        save_video_status(status_dict)
        
        logger.info(f"Advanced video {video_id} completed successfully")
        return final_output
        
    except Exception as e:
        logger.error(f"Error creating advanced video {video_id}: {str(e)}")
        status_dict[video_id] = {"status": "error", "progress": 0, "error": str(e)}
        save_video_status(status_dict)
        raise HTTPException(status_code=500, detail=f"Error creating advanced video: {str(e)}")
    
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temp dir: {str(e)}")

@app.get("/status/{video_id}")
async def get_video_status(video_id: str):
    """Get video status with persistent storage"""
    status_dict = load_video_status()
    
    if video_id not in status_dict:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return status_dict[video_id]

@app.get("/videos/{video_id}")
async def download_video(video_id: str):
    """Download generated video"""
    video_path = os.path.join(VIDEO_OUTPUT_DIR, f"{video_id}.mp4")
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{video_id}.mp4"
    )

@app.get("/videos")
async def list_videos():
    """List all generated videos"""
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