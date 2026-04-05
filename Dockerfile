# ─────────────────────────────────────────────────────────────────────────────
# Neuro-Drive: Human-Centric Driver Monitoring System
# Docker Image
# ─────────────────────────────────────────────────────────────────────────────
#
# Build:
#   docker build -t neuro-drive:latest .
#
# Run (Linux with webcam):
#   docker run --rm \
#     --device=/dev/video0:/dev/video0 \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -p 8000:8000 \
#     neuro-drive:latest
#
# Run with video file:
#   docker run --rm \
#     -v $(pwd)/videos:/app/videos \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -p 8000:8000 \
#     neuro-drive:latest python main.py --video /app/videos/test.mp4
#
# API only (no display):
#   docker run --rm -p 8000:8000 neuro-drive:latest \
#     python main.py --no-calibration --source /dev/null
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bullseye

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    # OpenCV GUI dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk-3-0 \
    # V4L2 (Video4Linux) for webcam access
    v4l-utils \
    libv4l-dev \
    # Audio (PortAudio for sounddevice)
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    # Utilities
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────
# Copy requirements first for better layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────
COPY . .

# ── Create output directory ───────────────────────────────────────────────
RUN mkdir -p /app/logs /app/models

# ── Environment variables ─────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DISPLAY=:0

# ── Ports ─────────────────────────────────────────────────────────────────
# FastAPI REST API
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Default command ───────────────────────────────────────────────────────
CMD ["python", "main.py", "--source", "0", "--no-calibration"]
