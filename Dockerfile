# Use Python 3.11 slim for ARM64 compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ARM64
RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    wget \
    curl \
    fonts-dejavu-core \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Configure ImageMagick policy for video processing
RUN sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml || true

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/videos /app/temp

# Set environment variables - OPTIMIZED VERSION
ENV VIDEO_OUTPUT_DIR=/app/videos
ENV TEMP_DIR=/app/temp
ENV MAX_WORKERS=2
ENV MAX_VIDEO_DURATION=600
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the OPTIMIZED application
CMD ["uvicorn", "main_optimized:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]