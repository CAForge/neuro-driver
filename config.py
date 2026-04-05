"""
config.py - Central configuration for Neuro-Drive: Human-Centric Driver Monitoring System
All thresholds, weights, and system parameters are defined here for easy tuning.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA & DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0               # Webcam index (0 = default)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30
DISPLAY_SCALE = 1.0            # Scale factor for display window

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE FACE MESH
# ─────────────────────────────────────────────────────────────────────────────
MAX_NUM_FACES = 1
REFINE_LANDMARKS = True        # Enables iris tracking (468 → 478 landmarks)
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# ─────────────────────────────────────────────────────────────────────────────
# EYE ASPECT RATIO (EAR)
# ─────────────────────────────────────────────────────────────────────────────
# EAR < EAR_CLOSED_THRESHOLD → eyes are closing
EAR_CLOSED_THRESHOLD = 0.22       # Below this = eye is considered closed
EAR_DROWSY_THRESHOLD = 0.27       # Below this = eye is partially closed (drowsy)
EAR_CLOSED_FRAMES = 15            # Consecutive frames with closed eyes → alert
EAR_DROWSY_FRAMES = 30            # Consecutive frames with drowsy eyes → alert

# Adaptive EAR: per-user calibration offset
EAR_CALIBRATION_FRAMES = 60       # Number of frames to calibrate baseline
EAR_ADAPTIVE_FACTOR = 0.75        # Threshold = baseline * adaptive_factor

# MediaPipe landmark indices for eyes
# Left eye (from driver's perspective)
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]  # P1..P6 for EAR formula

# Right eye
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]  # P1..P6 for EAR formula

# ─────────────────────────────────────────────────────────────────────────────
# MOUTH ASPECT RATIO (MAR) — Yawn Detection
# ─────────────────────────────────────────────────────────────────────────────
MAR_YAWN_THRESHOLD = 0.65         # Above this = yawning
MAR_YAWN_FRAMES = 20              # Consecutive frames with open mouth → yawn counted
MAR_YAWN_COUNT_ALERT = 2          # Number of yawns in window → trigger alert
MAR_YAWN_WINDOW_SECONDS = 120     # Time window for counting yawns (2 minutes)

# MediaPipe landmark indices for mouth
MOUTH_OUTER_INDICES = [61, 291, 39, 181, 0, 17, 269, 405]
MOUTH_MAR_INDICES = [61, 291, 39, 181, 0, 17, 269, 405]  # for MAR

# Specific vertical/horizontal mouth landmarks
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
MOUTH_TOP_INNER = [82, 87]
MOUTH_BOTTOM_INNER = [312, 317]

# ─────────────────────────────────────────────────────────────────────────────
# HEAD POSE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────
# Pitch: up/down tilt | Yaw: left/right turn | Roll: head tilt
HEAD_PITCH_DOWN_THRESHOLD = 15.0   # degrees down → nodding off
HEAD_PITCH_UP_THRESHOLD = 20.0     # degrees up → unusual
HEAD_YAW_THRESHOLD = 25.0          # degrees left/right → looking away
HEAD_ROLL_THRESHOLD = 20.0         # degrees roll → unusual head tilt

HEAD_POSE_ALERT_FRAMES = 20        # Consecutive frames off-road → distraction alert

# 3D model points for head pose (standard face model)
HEAD_MODEL_POINTS_INDICES = [1, 9, 57, 130, 287, 359]  # nose, chin, corners, etc.

# ─────────────────────────────────────────────────────────────────────────────
# GAZE TRACKING
# ─────────────────────────────────────────────────────────────────────────────
GAZE_OFF_ROAD_THRESHOLD = 0.35     # Iris deviation ratio → looking off-road
GAZE_ALERT_FRAMES = 25             # Consecutive frames with off-road gaze → alert
IRIS_CENTER_TOLERANCE = 0.25       # Normalized tolerance for center gaze

# Iris landmark indices (requires refine_landmarks=True)
LEFT_IRIS_INDICES = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]

# ─────────────────────────────────────────────────────────────────────────────
# FATIGUE SCORE (Composite)
# ─────────────────────────────────────────────────────────────────────────────
# fatigue_score = w1*EAR_score + w2*MAR_score + w3*head_pose_score + w4*gaze_score
FATIGUE_WEIGHT_EAR = 0.40          # EAR contribution
FATIGUE_WEIGHT_MAR = 0.20          # Yawning contribution
FATIGUE_WEIGHT_HEAD = 0.25         # Head pose contribution
FATIGUE_WEIGHT_GAZE = 0.15         # Gaze contribution

FATIGUE_SCORE_DROWSY = 0.45        # Score above this → DROWSY warning
FATIGUE_SCORE_CRITICAL = 0.70      # Score above this → CRITICAL alert
FATIGUE_SMOOTHING_ALPHA = 0.3      # Exponential smoothing factor (0=no smooth, 1=no history)

# ─────────────────────────────────────────────────────────────────────────────
# DISTRACTION DETECTION
# ─────────────────────────────────────────────────────────────────────────────
DISTRACTION_TIME_THRESHOLD = 2.5   # Seconds of distraction before alert
NO_FACE_TIME_THRESHOLD = 3.0       # Seconds with no face detected → alert

# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE THRESHOLDING (Environmental Robustness)
# ─────────────────────────────────────────────────────────────────────────────
BRIGHTNESS_LOW_THRESHOLD = 40      # Below this (0-255 mean) → low light mode
BRIGHTNESS_HIGH_THRESHOLD = 200    # Above this → overexposed
ADAPTIVE_EAR_SCALE_LOW_LIGHT = 0.9 # Relax EAR threshold in low light
ADAPTIVE_NOISE_KERNEL_SIZE = 3     # Gaussian blur kernel for noisy frames
MOTION_BLUR_THRESHOLD = 80.0       # Laplacian variance below this → blurry/motion

# ─────────────────────────────────────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 5.0       # Min seconds between same alert type
AUDIO_ALERT_ENABLED = True
AUDIO_ALERT_FREQUENCY = 880        # Hz for beep
AUDIO_ALERT_DURATION_MS = 500      # Duration of beep

# Alert level thresholds
ALERT_LEVELS = {
    "SAFE":      {"color": (0, 200, 0),   "label": "✓ SAFE"},
    "DROWSY":    {"color": (0, 165, 255), "label": "⚠ DROWSY"},
    "YAWNING":   {"color": (0, 165, 255), "label": "⚠ YAWNING"},
    "DISTRACTED":{"color": (0, 0, 255),   "label": "⚠ DISTRACTED"},
    "CRITICAL":  {"color": (0, 0, 200),   "label": "🚨 CRITICAL"},
    "NO_FACE":   {"color": (128, 0, 128), "label": "⚠ NO FACE DETECTED"},
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
LOG_ENABLED = True
LOG_FILE = "neuro_drive_events.log"
LOG_LEVEL = "INFO"                 # DEBUG, INFO, WARNING, ERROR
LOG_CSV_ENABLED = True
LOG_CSV_FILE = "neuro_drive_data.csv"
LOG_INTERVAL_SECONDS = 1.0         # How often to log metrics to CSV

# ─────────────────────────────────────────────────────────────────────────────
# OPENVINO OPTIMIZATION (Optional)
# ─────────────────────────────────────────────────────────────────────────────
USE_OPENVINO = False               # Enable OpenVINO acceleration if available
OPENVINO_DEVICE = "CPU"            # CPU, GPU, MYRIAD
OPENVINO_MODEL_DIR = "models/"

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
SKIP_FRAMES = 0                    # Process every Nth frame (0 = process all)
FACE_DETECTION_INTERVAL = 1        # Run face detection every N frames
DRAW_FACE_MESH = True              # Draw full 468-point mesh (can reduce FPS)
DRAW_IRIS = True                   # Draw iris tracking overlay

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
CALIBRATION_ENABLED = True
CALIBRATION_DURATION_SECONDS = 5   # How long to run calibration on startup

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI / REST API
# ─────────────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_ENABLE = True                  # Launch FastAPI alongside main loop
