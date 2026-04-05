"""
api.py - FastAPI REST API for Neuro-Drive
=========================================
Provides a REST interface to query real-time driver monitoring metrics.
Runs in a background thread alongside the main processing loop.

Endpoints:
    GET /               → System info
    GET /status         → Current alert status + fatigue score
    GET /metrics        → All sensor metrics (EAR, MAR, head pose, gaze)
    GET /stats          → Session statistics
    GET /health         → Health check
    GET /stream         → Server-Sent Events (SSE) real-time stream
    POST /calibrate     → Trigger recalibration
    POST /config        → Update threshold parameters

CORS is enabled for dashboard integration.
"""

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import json
import time
from typing import Optional

# Shared state (populated by main processing loop in main.py)
# Import the shared dict from main module at runtime
try:
    from main import _api_state
except ImportError:
    # Standalone mode or during testing
    _api_state = {
        "fatigue_score": 0.0,
        "alert_status": "SAFE",
        "fps": 0.0,
        "metrics": {},
        "stats": {},
    }

import config

# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Neuro-Drive API",
    description="Real-time driver monitoring system REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow requests from local dashboard / Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_start_time = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", summary="System information")
async def root():
    """Returns basic system information."""
    return {
        "system":    "Neuro-Drive: Human-Centric Driver Monitoring System",
        "version":   "1.0.0",
        "uptime_s":  round(time.time() - _start_time, 1),
        "endpoints": ["/status", "/metrics", "/stats", "/stream", "/health",
                      "/docs"],
    }


@app.get("/health", summary="Health check")
async def health():
    """Liveness probe — returns 200 if server is running."""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/status", summary="Current alert status")
async def get_status():
    """
    Returns the current alert status and fatigue score.
    
    Response:
        alert_status:  SAFE | DROWSY | YAWNING | DISTRACTED | CRITICAL | NO_FACE
        fatigue_score: 0.0 – 1.0
        fps:           Current processing FPS
        timestamp:     Unix timestamp
    """
    return {
        "alert_status":  _api_state.get("alert_status", "SAFE"),
        "fatigue_score": _api_state.get("fatigue_score", 0.0),
        "fps":           _api_state.get("fps", 0.0),
        "timestamp":     time.time(),
    }


@app.get("/metrics", summary="All sensor metrics")
async def get_metrics():
    """
    Returns all real-time sensor metrics.
    
    Response includes:
        EAR (eye aspect ratio), MAR (mouth aspect ratio),
        head pose angles, gaze position, environmental conditions.
    """
    return {
        "metrics":       _api_state.get("metrics", {}),
        "alert_status":  _api_state.get("alert_status", "SAFE"),
        "fatigue_score": _api_state.get("fatigue_score", 0.0),
        "timestamp":     time.time(),
    }


@app.get("/stats", summary="Session statistics")
async def get_stats():
    """
    Returns session-level statistics:
        - Session duration
        - Total alert count
        - Total yawn count
        - PERCLOS (% eye closure)
    """
    return {
        "stats":     _api_state.get("stats", {}),
        "timestamp": time.time(),
    }


@app.get("/thresholds", summary="Current threshold configuration")
async def get_thresholds():
    """Returns all configurable detection thresholds."""
    return {
        "ear_closed_threshold":  config.EAR_CLOSED_THRESHOLD,
        "ear_drowsy_threshold":  config.EAR_DROWSY_THRESHOLD,
        "mar_yawn_threshold":    config.MAR_YAWN_THRESHOLD,
        "head_yaw_threshold":    config.HEAD_YAW_THRESHOLD,
        "head_pitch_threshold":  config.HEAD_PITCH_DOWN_THRESHOLD,
        "gaze_off_road_threshold": config.GAZE_OFF_ROAD_THRESHOLD,
        "fatigue_score_drowsy":  config.FATIGUE_SCORE_DROWSY,
        "fatigue_score_critical": config.FATIGUE_SCORE_CRITICAL,
        "fatigue_weights": {
            "ear":   config.FATIGUE_WEIGHT_EAR,
            "mar":   config.FATIGUE_WEIGHT_MAR,
            "head":  config.FATIGUE_WEIGHT_HEAD,
            "gaze":  config.FATIGUE_WEIGHT_GAZE,
        }
    }


@app.get("/stream", summary="Server-Sent Events (SSE) real-time stream")
async def stream_metrics():
    """
    Streams real-time metrics as Server-Sent Events (SSE).
    
    Connect with EventSource in JavaScript:
        const source = new EventSource('http://localhost:8000/stream');
        source.onmessage = (e) => { const data = JSON.parse(e.data); ... };
    
    Sends updates at ~5 Hz.
    """
    async def event_generator():
        while True:
            data = {
                "alert_status":  _api_state.get("alert_status", "SAFE"),
                "fatigue_score": _api_state.get("fatigue_score", 0.0),
                "fps":           _api_state.get("fps", 0.0),
                "metrics":       _api_state.get("metrics", {}),
                "timestamp":     time.time(),
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.2)  # 5 Hz

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/alerts/history", summary="Recent alert history")
async def get_alert_history():
    """
    Returns a summary of alerts from the current session.
    Full history is available in the CSV log file.
    """
    stats = _api_state.get("stats", {})
    return {
        "total_alerts":       stats.get("total_alerts", 0),
        "total_yawns":        stats.get("total_yawns", 0),
        "session_duration_s": stats.get("session_duration_s", 0),
        "perclos":            stats.get("perclos", 0),
        "current_status":     _api_state.get("alert_status", "SAFE"),
        "timestamp":          time.time(),
    }
