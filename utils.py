"""
utils.py - Shared utility functions for Neuro-Drive
Includes: EAR/MAR computation, drawing helpers, adaptive thresholding,
          audio alert, logging setup, FPS counter, and calibration.
"""

import cv2
import numpy as np
import math
import time
import logging
import csv
import os
import threading
from collections import deque
from typing import List, Tuple, Optional, Dict

import config


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(name: str = "neuro_drive") -> logging.Logger:
    """Configure and return a logger that writes to both console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if config.LOG_ENABLED:
        fh = logging.FileHandler(config.LOG_FILE)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


logger = setup_logger()


# ─────────────────────────────────────────────────────────────────────────────
# CSV DATA LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class CSVDataLogger:
    """Logs per-frame metrics to CSV for offline analysis and dashboards."""

    HEADERS = [
        "timestamp", "fps", "ear_left", "ear_right", "ear_avg",
        "mar", "pitch", "yaw", "roll",
        "gaze_x", "gaze_y", "fatigue_score",
        "alert_status", "brightness", "is_blurry"
    ]

    def __init__(self, filepath: str = config.LOG_CSV_FILE):
        self.filepath = filepath
        self._last_log_time = 0.0
        self._file = None
        self._writer = None
        if config.LOG_CSV_ENABLED:
            self._init_file()

    def _init_file(self):
        file_exists = os.path.isfile(self.filepath)
        self._file = open(self.filepath, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.HEADERS)
        if not file_exists:
            self._writer.writeheader()

    def log(self, metrics: dict):
        if not config.LOG_CSV_ENABLED or self._writer is None:
            return
        now = time.time()
        if now - self._last_log_time < config.LOG_INTERVAL_SECONDS:
            return
        self._last_log_time = now
        row = {k: metrics.get(k, "") for k in self.HEADERS}
        row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()


# ─────────────────────────────────────────────────────────────────────────────
# FPS COUNTER
# ─────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    """Rolling-window FPS counter for stable display."""

    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)

    def tick(self):
        self._times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# EYE ASPECT RATIO (EAR)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ear(landmarks_2d: np.ndarray, eye_indices: List[int]) -> float:
    """
    Compute Eye Aspect Ratio (EAR) using the Soukupová & Čech formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Where p1..p6 are the 6 key eye landmarks:
        p1 = outer corner, p4 = inner corner
        p2/p3 = upper eyelid, p5/p6 = lower eyelid

    Args:
        landmarks_2d: Array of shape (N, 2) with all face landmark (x, y) coords
        eye_indices:  List of 6 landmark indices [p1, p2, p3, p4, p5, p6]

    Returns:
        EAR float value (typically 0.15–0.40; below 0.20 = closed)
    """
    if len(eye_indices) < 6:
        return 0.30  # fallback

    pts = landmarks_2d[eye_indices]  # shape (6, 2)
    p1, p2, p3, p4, p5, p6 = pts

    # Vertical distances
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    # Horizontal distance
    C = np.linalg.norm(p1 - p4)

    if C < 1e-6:
        return 0.30

    ear = (A + B) / (2.0 * C)
    return float(ear)


# ─────────────────────────────────────────────────────────────────────────────
# MOUTH ASPECT RATIO (MAR)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mar(landmarks_2d: np.ndarray) -> float:
    """
    Compute Mouth Aspect Ratio (MAR) for yawn detection.
    Similar concept to EAR but for the mouth.

    Uses:
        - Vertical: distance between top and bottom inner lip
        - Horizontal: distance between mouth corners

    Returns:
        MAR float (typically 0.1–0.3 closed; > 0.6 = yawning)
    """
    try:
        # Inner lip vertical
        top    = landmarks_2d[config.MOUTH_TOP]
        bottom = landmarks_2d[config.MOUTH_BOTTOM]
        left   = landmarks_2d[config.MOUTH_LEFT]
        right  = landmarks_2d[config.MOUTH_RIGHT]

        vertical   = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(left - right)

        if horizontal < 1e-6:
            return 0.0

        mar = vertical / horizontal
        return float(mar)
    except (IndexError, Exception):
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# GAZE DEVIATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_gaze_ratio(landmarks_2d: np.ndarray,
                       eye_indices: List[int],
                       iris_indices: List[int]) -> Tuple[float, float]:
    """
    Compute normalized iris position within the eye bounding box.

    Returns:
        (gaze_x, gaze_y) in range [0, 1]
        (0.5, 0.5) = center / looking forward
        < 0.35 or > 0.65 = looking to the side
    """
    try:
        eye_pts  = landmarks_2d[eye_indices]
        iris_pts = landmarks_2d[iris_indices]

        # Eye bounding box
        ex_min, ey_min = eye_pts.min(axis=0)
        ex_max, ey_max = eye_pts.max(axis=0)

        # Iris center
        iris_center = iris_pts.mean(axis=0)

        eye_w = ex_max - ex_min
        eye_h = ey_max - ey_min

        if eye_w < 1e-6 or eye_h < 1e-6:
            return 0.5, 0.5

        gaze_x = (iris_center[0] - ex_min) / eye_w
        gaze_y = (iris_center[1] - ey_min) / eye_h

        # Clamp to [0, 1]
        gaze_x = float(np.clip(gaze_x, 0.0, 1.0))
        gaze_y = float(np.clip(gaze_y, 0.0, 1.0))

        return gaze_x, gaze_y
    except (IndexError, Exception):
        return 0.5, 0.5


# ─────────────────────────────────────────────────────────────────────────────
# FRAME QUALITY ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

def assess_frame_quality(frame: np.ndarray) -> Dict[str, float]:
    """
    Assess environmental conditions from the frame.

    Returns dict with:
        brightness (0–255 mean), is_blurry (bool), blur_score (float)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))

    # Laplacian variance: low = blurry/motion
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry  = blur_score < config.MOTION_BLUR_THRESHOLD

    return {
        "brightness": brightness,
        "blur_score": blur_score,
        "is_blurry":  is_blurry,
    }


def get_adaptive_ear_threshold(base_threshold: float, brightness: float) -> float:
    """
    Adjust EAR threshold based on lighting conditions.
    In very low light, landmarks are less reliable → relax threshold slightly.
    """
    if brightness < config.BRIGHTNESS_LOW_THRESHOLD:
        return base_threshold * config.ADAPTIVE_EAR_SCALE_LOW_LIGHT
    return base_threshold


def preprocess_frame(frame: np.ndarray, brightness: float) -> np.ndarray:
    """
    Apply adaptive preprocessing:
    - CLAHE histogram equalization in low light
    - Mild blur reduction in noisy conditions
    """
    if brightness < config.BRIGHTNESS_LOW_THRESHOLD:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO ALERT
# ─────────────────────────────────────────────────────────────────────────────

def _generate_beep(frequency: int = 880, duration_ms: int = 500,
                   sample_rate: int = 44100) -> np.ndarray:
    """Generate a numpy audio array for a sine-wave beep."""
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000),
                    endpoint=False)
    wave = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    return wave


_audio_lock = threading.Lock()
_last_beep_time: Dict[str, float] = {}


def play_alert_sound(alert_type: str = "DROWSY"):
    """
    Play an audio alert beep in a background thread.
    Respects cooldown to avoid spamming.
    
    Falls back gracefully if sounddevice is unavailable.
    """
    if not config.AUDIO_ALERT_ENABLED:
        return

    now = time.time()
    with _audio_lock:
        last = _last_beep_time.get(alert_type, 0)
        if now - last < config.ALERT_COOLDOWN_SECONDS:
            return
        _last_beep_time[alert_type] = now

    def _beep():
        try:
            import sounddevice as sd
            wave = _generate_beep(
                frequency=config.AUDIO_ALERT_FREQUENCY,
                duration_ms=config.AUDIO_ALERT_DURATION_MS
            )
            sd.play(np.column_stack([wave, wave]), samplerate=44100)
            sd.wait()
        except ImportError:
            # sounddevice not available — try os beep
            try:
                import winsound
                winsound.Beep(config.AUDIO_ALERT_FREQUENCY,
                              config.AUDIO_ALERT_DURATION_MS)
            except Exception:
                pass  # Silent fallback — alert still shown visually
        except Exception:
            pass

    threading.Thread(target=_beep, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_landmarks_subset(frame: np.ndarray, landmarks_2d: np.ndarray,
                           indices: List[int], color: Tuple[int, int, int],
                           radius: int = 2, thickness: int = -1):
    """Draw a subset of face mesh landmarks as filled circles."""
    for idx in indices:
        if 0 <= idx < len(landmarks_2d):
            pt = tuple(landmarks_2d[idx].astype(int))
            cv2.circle(frame, pt, radius, color, thickness)


def draw_eye_contour(frame: np.ndarray, landmarks_2d: np.ndarray,
                     eye_indices: List[int], color: Tuple[int, int, int],
                     thickness: int = 1):
    """Draw eye contour polygon."""
    pts = landmarks_2d[eye_indices].astype(np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness,
                  lineType=cv2.LINE_AA)


def draw_iris(frame: np.ndarray, landmarks_2d: np.ndarray,
              iris_indices: List[int], color: Tuple[int, int, int]):
    """Draw iris as a circle fitted around the 4 iris landmarks."""
    iris_pts = landmarks_2d[iris_indices].astype(np.float32)
    center = iris_pts.mean(axis=0).astype(int)
    # Radius = half of iris diameter
    diam = np.linalg.norm(iris_pts[0] - iris_pts[2])
    radius = max(2, int(diam / 2))
    cv2.circle(frame, tuple(center), radius, color, 1, lineType=cv2.LINE_AA)
    cv2.circle(frame, tuple(center), 2, color, -1)


def draw_head_pose_axes(frame: np.ndarray,
                        nose_pt: Tuple[int, int],
                        rotation_vector: np.ndarray,
                        translation_vector: np.ndarray,
                        camera_matrix: np.ndarray,
                        dist_coeffs: np.ndarray,
                        axis_length: float = 50.0):
    """
    Project 3D coordinate axes onto the frame to visualize head pose.
    X=red (right), Y=green (up), Z=blue (out of screen).
    """
    axis_3d = np.float32([
        [axis_length, 0, 0],   # X axis — red
        [0, axis_length, 0],   # Y axis — green
        [0, 0, axis_length],   # Z axis — blue
    ])
    origin_3d = np.float32([[0, 0, 0]])

    try:
        img_pts, _ = cv2.projectPoints(
            np.vstack([origin_3d, axis_3d]),
            rotation_vector, translation_vector,
            camera_matrix, dist_coeffs
        )
        img_pts = img_pts.reshape(-1, 2).astype(int)

        origin = tuple(img_pts[0])
        cv2.arrowedLine(frame, origin, tuple(img_pts[1]), (0, 0, 255), 2, tipLength=0.3)   # X red
        cv2.arrowedLine(frame, origin, tuple(img_pts[2]), (0, 255, 0), 2, tipLength=0.3)   # Y green
        cv2.arrowedLine(frame, origin, tuple(img_pts[3]), (255, 0, 0), 2, tipLength=0.3)   # Z blue
    except cv2.error:
        pass


def draw_hud_panel(frame: np.ndarray, metrics: dict, alert_status: str,
                   fatigue_score: float, fps: float):
    """
    Draw the main heads-up display panel with:
    - Fatigue score bar
    - EAR / MAR values
    - Head pose angles
    - Alert status banner
    - FPS counter
    """
    h, w = frame.shape[:2]

    # ── Semi-transparent dark panel on the left ────────────────────────────
    panel_w = 280
    panel_h = h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # ── Title ──────────────────────────────────────────────────────────────
    cv2.putText(frame, "NEURO-DRIVE", (10, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Driver Monitor", (10, 48),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (150, 150, 180), 1, cv2.LINE_AA)
    cv2.line(frame, (10, 55), (panel_w - 10, 55), (60, 60, 80), 1)

    y = 80
    def label(key, val, color=(200, 200, 220), fmt="{:.3f}"):
        nonlocal y
        cv2.putText(frame, f"{key}:", (12, y),
                    cv2.FONT_HERSHEY_PLAIN, 0.95, (120, 120, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, fmt.format(val) if isinstance(val, float) else str(val),
                    (130, y), cv2.FONT_HERSHEY_PLAIN, 0.95, color, 1, cv2.LINE_AA)
        y += 22

    # EAR
    ear_avg = metrics.get("ear_avg", 0.0)
    ear_color = (0, 200, 0) if ear_avg > config.EAR_DROWSY_THRESHOLD else (0, 100, 255)
    label("EAR L", metrics.get("ear_left", 0.0), ear_color)
    label("EAR R", metrics.get("ear_right", 0.0), ear_color)
    label("EAR Avg", ear_avg, ear_color)

    y += 5
    # MAR
    mar = metrics.get("mar", 0.0)
    mar_color = (0, 165, 255) if mar > config.MAR_YAWN_THRESHOLD else (200, 200, 220)
    label("MAR", mar, mar_color)
    label("Yawns", metrics.get("yawn_count", 0), fmt="{}")

    y += 5
    # Head pose
    pitch = metrics.get("pitch", 0.0)
    yaw   = metrics.get("yaw",   0.0)
    roll  = metrics.get("roll",  0.0)
    hp_color = (0, 100, 255) if (abs(yaw) > config.HEAD_YAW_THRESHOLD or
                                  pitch < -config.HEAD_PITCH_DOWN_THRESHOLD) \
               else (200, 200, 220)
    label("Pitch", pitch, hp_color, "{:.1f}°")
    label("Yaw",   yaw,   hp_color, "{:.1f}°")
    label("Roll",  roll,  (200, 200, 220), "{:.1f}°")

    y += 5
    # Gaze
    gx = metrics.get("gaze_x", 0.5)
    gy = metrics.get("gaze_y", 0.5)
    gaze_ok = (abs(gx - 0.5) < config.IRIS_CENTER_TOLERANCE and
               abs(gy - 0.5) < config.IRIS_CENTER_TOLERANCE)
    gaze_color = (0, 200, 0) if gaze_ok else (0, 100, 255)
    label("Gaze X", gx, gaze_color)
    label("Gaze Y", gy, gaze_color)

    y += 10
    cv2.line(frame, (10, y), (panel_w - 10, y), (60, 60, 80), 1)
    y += 16

    # ── Fatigue Score Bar ──────────────────────────────────────────────────
    cv2.putText(frame, "FATIGUE SCORE", (10, y),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (180, 180, 210), 1, cv2.LINE_AA)
    y += 18
    bar_w = panel_w - 24
    bar_h = 14
    # Background
    cv2.rectangle(frame, (12, y), (12 + bar_w, y + bar_h), (50, 50, 60), -1)
    # Filled portion
    fill_w = int(bar_w * min(fatigue_score, 1.0))
    bar_color = (
        (0, 200, 0)    if fatigue_score < config.FATIGUE_SCORE_DROWSY else
        (0, 165, 255)  if fatigue_score < config.FATIGUE_SCORE_CRITICAL else
        (0, 0, 220)
    )
    if fill_w > 0:
        cv2.rectangle(frame, (12, y), (12 + fill_w, y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (12, y), (12 + bar_w, y + bar_h), (100, 100, 120), 1)
    # Score text on bar
    score_text = f"{fatigue_score:.2f}"
    cv2.putText(frame, score_text,
                (12 + bar_w // 2 - 16, y + bar_h - 2),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    y += bar_h + 8

    # ── Env info ──────────────────────────────────────────────────────────
    bright = metrics.get("brightness", 128)
    blurry = metrics.get("is_blurry", False)
    label("Brightness", bright, fmt="{:.0f}")
    label("Blurry", "YES" if blurry else "NO",
          (0, 100, 255) if blurry else (0, 200, 0), fmt="{}")

    # FPS bottom
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (12, h - 12), cv2.FONT_HERSHEY_PLAIN, 0.95,
                (0, 200, 200), 1, cv2.LINE_AA)

    # ── Alert Banner (top-right) ──────────────────────────────────────────
    alert_cfg = config.ALERT_LEVELS.get(alert_status, config.ALERT_LEVELS["SAFE"])
    alert_color = alert_cfg["color"]
    alert_label = alert_cfg["label"]

    banner_x1 = panel_w + 10
    banner_y1 = 12
    banner_h  = 50
    banner_w  = w - panel_w - 20

    if alert_status != "SAFE":
        # Blinking effect: toggle visibility every ~15 frames using time
        blink_on = int(time.time() * 3) % 2 == 0
        if blink_on:
            cv2.rectangle(frame, (banner_x1, banner_y1),
                          (banner_x1 + banner_w, banner_y1 + banner_h),
                          alert_color, -1)
            cv2.putText(frame, alert_label,
                        (banner_x1 + 20, banner_y1 + 34),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (banner_x1, banner_y1),
                      (banner_x1 + banner_w, banner_y1 + banner_h),
                      (20, 80, 20), -1)
        cv2.putText(frame, alert_label,
                    (banner_x1 + 20, banner_y1 + 34),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

class DriverCalibration:
    """
    Calibrates per-user baseline EAR and MAR by observing
    the driver for N seconds in a neutral state.
    """

    def __init__(self, duration: float = config.CALIBRATION_DURATION_SECONDS):
        self.duration = duration
        self.ear_samples: List[float] = []
        self.mar_samples: List[float] = []
        self.start_time: Optional[float] = None
        self.done = False

        # Calibrated thresholds (default = config values)
        self.ear_threshold = config.EAR_CLOSED_THRESHOLD
        self.mar_threshold = config.MAR_YAWN_THRESHOLD

    def update(self, ear: float, mar: float):
        if self.done:
            return
        if self.start_time is None:
            self.start_time = time.time()

        self.ear_samples.append(ear)
        self.mar_samples.append(mar)

        if time.time() - self.start_time >= self.duration:
            self._finalize()

    def _finalize(self):
        if len(self.ear_samples) > 10:
            mean_ear = np.mean(self.ear_samples)
            self.ear_threshold = mean_ear * config.EAR_ADAPTIVE_FACTOR
            logger.info(f"[Calibration] EAR baseline={mean_ear:.3f}  "
                        f"threshold={self.ear_threshold:.3f}")
        if len(self.mar_samples) > 10:
            mean_mar = np.mean(self.mar_samples)
            self.mar_threshold = mean_mar * 2.5  # yawn = 2.5× resting
            logger.info(f"[Calibration] MAR baseline={mean_mar:.3f}  "
                        f"threshold={self.mar_threshold:.3f}")
        self.done = True

    @property
    def progress(self) -> float:
        if self.start_time is None:
            return 0.0
        return min(1.0, (time.time() - self.start_time) / self.duration)


def draw_calibration_screen(frame: np.ndarray, progress: float):
    """Draw calibration overlay asking driver to look forward."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "CALIBRATING...", (w // 2 - 160, h // 2 - 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Please look straight ahead",
                (w // 2 - 180, h // 2 - 20),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "Keep a relaxed, neutral expression",
                (w // 2 - 200, h // 2 + 10),
                cv2.FONT_HERSHEY_PLAIN, 1.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Progress bar
    bar_x, bar_y = w // 2 - 200, h // 2 + 50
    bar_w, bar_h = 400, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * progress), bar_y + bar_h),
                  (0, 200, 255), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (120, 120, 120), 1)
    pct = int(progress * 100)
    cv2.putText(frame, f"{pct}%", (bar_x + bar_w // 2 - 20, bar_y + 15),
                cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 1, cv2.LINE_AA)

    return frame
