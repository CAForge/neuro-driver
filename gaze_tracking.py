"""
gaze_tracking.py - Gaze Tracking for Neuro-Drive
=================================================
Tracks driver gaze direction using MediaPipe iris landmarks (requires
refine_landmarks=True, which extends the mesh to 478 points including
4 iris landmarks per eye).

Approach:
    1. Locate iris centre using the 4 iris landmark points
    2. Locate eye bounding box using the eye contour landmarks
    3. Compute normalised iris position within the eye bounding box
       → (0.5, 0.5) means looking straight ahead
    4. Threshold deviation from centre to classify as "off-road"

Gaze Score (0–1):
    Represents how far the gaze is from the forward-looking position.
    Used as one component in the composite fatigue_score.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import time
import logging

import config
from utils import compute_gaze_ratio

logger = logging.getLogger("neuro_drive")


# ─────────────────────────────────────────────────────────────────────────────
# GAZE TRACKER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class GazeTracker:
    """
    Real-time gaze tracker using MediaPipe iris landmarks.
    
    Tracks:
        - Horizontal gaze direction (left / centre / right)
        - Vertical gaze direction (up / centre / down)
        - Combined off-road gaze score
        - Consecutive frames of distracted gaze
    """

    # Gaze region labels
    GAZE_LABELS = {
        (0, 0): "UP-LEFT",    (0, 1): "UP",        (0, 2): "UP-RIGHT",
        (1, 0): "LEFT",       (1, 1): "CENTER",     (1, 2): "RIGHT",
        (2, 0): "DOWN-LEFT",  (2, 1): "DOWN",       (2, 2): "DOWN-RIGHT",
    }

    def __init__(self):
        # Smoothing buffers (rolling average over N frames)
        self._gaze_x_buf: List[float] = []
        self._gaze_y_buf: List[float] = []
        self._buf_size = 8  # frames for smoothing

        # Alert state
        self.off_gaze_frames = 0
        self.gaze_alert = False

        # Calibration offset (set during calibration phase)
        self._offset_x = 0.5
        self._offset_y = 0.5
        self._calibrated = False
        self._calib_samples_x: List[float] = []
        self._calib_samples_y: List[float] = []

        # Current gaze state
        self.gaze_x = 0.5
        self.gaze_y = 0.5
        self.gaze_direction = "CENTER"
        self.gaze_score = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # PRIMARY UPDATE METHOD
    # ─────────────────────────────────────────────────────────────────────

    def update(self, landmarks_2d: np.ndarray) -> Tuple[float, float, str, float]:
        """
        Compute gaze from 2D landmarks.
        
        Args:
            landmarks_2d: (N, 2) array of pixel coordinates from MediaPipe
                          Must include iris landmarks (indices 469-477)
        
        Returns:
            gaze_x, gaze_y, direction (str), gaze_score (float)
        """
        try:
            # Compute gaze for each eye
            lx, ly = compute_gaze_ratio(
                landmarks_2d,
                config.LEFT_EYE_INDICES,
                config.LEFT_IRIS_INDICES
            )
            rx, ry = compute_gaze_ratio(
                landmarks_2d,
                config.RIGHT_EYE_INDICES,
                config.RIGHT_IRIS_INDICES
            )

            # Average both eyes
            raw_x = (lx + rx) / 2.0
            raw_y = (ly + ry) / 2.0

        except (IndexError, Exception) as e:
            logger.debug(f"[Gaze] Landmark error: {e}")
            raw_x, raw_y = 0.5, 0.5

        # Apply calibration offset
        if self._calibrated:
            raw_x = raw_x - self._offset_x + 0.5
            raw_y = raw_y - self._offset_y + 0.5
            raw_x = float(np.clip(raw_x, 0.0, 1.0))
            raw_y = float(np.clip(raw_y, 0.0, 1.0))

        # Smooth via rolling buffer
        self._gaze_x_buf.append(raw_x)
        self._gaze_y_buf.append(raw_y)
        if len(self._gaze_x_buf) > self._buf_size:
            self._gaze_x_buf.pop(0)
            self._gaze_y_buf.pop(0)

        smooth_x = float(np.mean(self._gaze_x_buf))
        smooth_y = float(np.mean(self._gaze_y_buf))

        self.gaze_x = smooth_x
        self.gaze_y = smooth_y

        # Classify direction
        self.gaze_direction = self._classify_direction(smooth_x, smooth_y)

        # Compute gaze score
        self.gaze_score = self._compute_score(smooth_x, smooth_y)

        # Update alert counter
        is_off_road = self.gaze_direction != "CENTER"
        if is_off_road:
            self.off_gaze_frames += 1
        else:
            self.off_gaze_frames = max(0, self.off_gaze_frames - 2)

        self.gaze_alert = self.off_gaze_frames >= config.GAZE_ALERT_FRAMES

        return smooth_x, smooth_y, self.gaze_direction, self.gaze_score

    # ─────────────────────────────────────────────────────────────────────
    # GAZE DIRECTION CLASSIFICATION
    # ─────────────────────────────────────────────────────────────────────

    def _classify_direction(self, gx: float, gy: float) -> str:
        """
        Classify gaze into one of 9 regions using the tolerance band.
        
        Horizontal bands: LEFT < (0.5 - tol), CENTER, (0.5 + tol) < RIGHT
        Vertical bands:   UP < (0.5 - tol), CENTER, (0.5 + tol) < DOWN
        """
        tol = config.IRIS_CENTER_TOLERANCE

        if gx < (0.5 - tol):
            h_idx = 0   # LEFT
        elif gx > (0.5 + tol):
            h_idx = 2   # RIGHT
        else:
            h_idx = 1   # CENTER

        if gy < (0.5 - tol):
            v_idx = 0   # UP
        elif gy > (0.5 + tol):
            v_idx = 2   # DOWN
        else:
            v_idx = 1   # CENTER

        # Only CENTER vertically + horizontally = safe
        label = self.GAZE_LABELS.get((v_idx, h_idx), "CENTER")
        return label

    def _compute_score(self, gx: float, gy: float) -> float:
        """
        Compute normalised gaze-off-road score in [0, 1].
        
        Score 0.0 → looking perfectly forward
        Score 1.0 → maximum gaze deviation
        """
        dx = abs(gx - 0.5)
        dy = abs(gy - 0.5)
        # Euclidean distance from centre, normalised by max possible deviation
        deviation = np.sqrt(dx**2 + dy**2)
        max_dev = np.sqrt(0.5**2 + 0.5**2)  # corner = 0.707
        score = deviation / max_dev
        return float(np.clip(score, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────
    # CALIBRATION
    # ─────────────────────────────────────────────────────────────────────

    def calibrate(self, gx: float, gy: float):
        """
        Accumulate calibration samples while driver looks forward.
        Call finalize_calibration() when enough samples are collected.
        """
        self._calib_samples_x.append(gx)
        self._calib_samples_y.append(gy)

    def finalize_calibration(self):
        """Compute gaze offset from calibration samples."""
        if len(self._calib_samples_x) < 5:
            return
        self._offset_x = float(np.mean(self._calib_samples_x))
        self._offset_y = float(np.mean(self._calib_samples_y))
        self._calibrated = True
        logger.info(f"[Gaze] Calibrated offset: x={self._offset_x:.3f}, y={self._offset_y:.3f}")

    # ─────────────────────────────────────────────────────────────────────
    # DRAWING
    # ─────────────────────────────────────────────────────────────────────

    def draw_gaze_indicator(self, frame: np.ndarray, panel_offset_x: int = 290):
        """
        Draw a small gaze indicator compass in the corner of the frame.
        Shows a dot representing current gaze position within a 3×3 grid.
        """
        h, w = frame.shape[:2]
        size = 80
        x0 = panel_offset_x + 20
        y0 = h - size - 20

        # Background grid
        cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (30, 30, 40), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (80, 80, 100), 1)

        # Grid lines (3x3)
        third = size // 3
        for i in [1, 2]:
            cv2.line(frame, (x0 + i*third, y0), (x0 + i*third, y0 + size),
                     (60, 60, 80), 1)
            cv2.line(frame, (x0, y0 + i*third), (x0 + size, y0 + i*third),
                     (60, 60, 80), 1)

        # Centre crosshair
        cx = x0 + size // 2
        cy = y0 + size // 2
        cv2.line(frame, (cx - 5, cy), (cx + 5, cy), (100, 100, 120), 1)
        cv2.line(frame, (cx, cy - 5), (cx, cy + 5), (100, 100, 120), 1)

        # Gaze dot position
        dot_x = int(x0 + self.gaze_x * size)
        dot_y = int(y0 + self.gaze_y * size)
        dot_x = np.clip(dot_x, x0 + 3, x0 + size - 3)
        dot_y = np.clip(dot_y, y0 + 3, y0 + size - 3)

        # Dot colour: green=center, orange=off-center, red=far
        score = self.gaze_score
        dot_color = (
            (0, 220, 0)    if score < 0.25 else
            (0, 165, 255)  if score < 0.50 else
            (0, 0, 220)
        )
        cv2.circle(frame, (dot_x, dot_y), 6, dot_color, -1)
        cv2.circle(frame, (dot_x, dot_y), 6, (255, 255, 255), 1)

        # Label
        cv2.putText(frame, "GAZE", (x0, y0 - 5),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (180, 180, 200), 1, cv2.LINE_AA)
        dir_color = (0, 220, 0) if self.gaze_direction == "CENTER" else (0, 100, 255)
        cv2.putText(frame, self.gaze_direction, (x0, y0 + size + 14),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, dir_color, 1, cv2.LINE_AA)

        return frame


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tracker = GazeTracker()
    # Simulate looking left
    dummy_gaze = (0.2, 0.5)
    score = tracker._compute_score(*dummy_gaze)
    direction = tracker._classify_direction(*dummy_gaze)
    print(f"Gaze {dummy_gaze} → direction={direction}, score={score:.3f}")
