"""
head_pose.py - Head Pose Estimation for Neuro-Drive
====================================================
Uses solvePnP with a calibrated 3D face model and MediaPipe landmarks
to estimate the driver's head orientation in Euler angles (pitch, yaw, roll).

Theory:
    solvePnP finds the rotation (R) and translation (T) vectors that map
    known 3D face model points to 2D image projections.
    We then decompose R into Euler angles using Rodrigues + RQ decomposition.

    Pitch:  +ve = looking up,     -ve = looking down (nodding)
    Yaw:    +ve = turning right,  -ve = turning left
    Roll:   +ve = tilting right,  -ve = tilting left
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

import config

logger = logging.getLogger("neuro_drive")


# ─────────────────────────────────────────────────────────────────────────────
# 3D FACE MODEL POINTS
# ─────────────────────────────────────────────────────────────────────────────

# Standard anthropometric 3D face model (in mm, centred at nose tip).
# These correspond to well-known anatomical landmarks.
# Indices reference MediaPipe Face Mesh (468-point) landmark IDs.

FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0],      # Nose tip         → landmark 1
    [0.0,   -330.0, -65.0],     # Chin             → landmark 152
    [-225.0, 170.0, -135.0],    # Left eye corner  → landmark 33
    [225.0,  170.0, -135.0],    # Right eye corner → landmark 263
    [-150.0, -150.0, -125.0],   # Left mouth corner→ landmark 61
    [150.0,  -150.0, -125.0],   # Right mouth corner→landmark 291
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
FACE_3D_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


# ─────────────────────────────────────────────────────────────────────────────
# HEAD POSE ESTIMATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class HeadPoseEstimator:
    """
    Estimates head pose (pitch, yaw, roll) from MediaPipe face landmarks.
    
    Uses:
        - cv2.solvePnP (SOLVEPNP_ITERATIVE or SOLVEPNP_EPNP)
        - Rodrigues rotation decomposition
        - Euler angle extraction via RQ decomposition
    
    Maintains:
        - Rolling smoothing of angles to reduce jitter
        - Frame counter for consecutive off-pose detection
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width  = frame_width
        self.frame_height = frame_height

        # Camera intrinsics (estimated from image size — no calibration needed)
        self.camera_matrix, self.dist_coeffs = self._build_camera_matrix()

        # Smoothed output angles
        self._smooth_pitch = 0.0
        self._smooth_yaw   = 0.0
        self._smooth_roll  = 0.0
        self._alpha = 0.4  # Smoothing factor

        # State for alert logic
        self.off_pose_frames = 0
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None

    def _build_camera_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build an approximate camera intrinsic matrix.
        Focal length estimated as image width (reasonable for webcams).
        
        Camera matrix K:
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        focal_length = self.frame_width  # in pixels
        cx = self.frame_width  / 2.0
        cy = self.frame_height / 2.0

        camera_matrix = np.array([
            [focal_length, 0,            cx],
            [0,            focal_length, cy],
            [0,            0,            1.0]
        ], dtype=np.float64)

        # Assuming no lens distortion (good enough for webcams)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        return camera_matrix, dist_coeffs

    def update_camera_size(self, width: int, height: int):
        """Recalibrate if frame size changes (e.g., different camera)."""
        self.frame_width  = width
        self.frame_height = height
        self.camera_matrix, self.dist_coeffs = self._build_camera_matrix()

    def estimate(self, landmarks_2d: np.ndarray) -> Tuple[float, float, float,
                                                           Optional[np.ndarray],
                                                           Optional[np.ndarray]]:
        """
        Estimate head pose from 2D landmark positions.
        
        Args:
            landmarks_2d: Array (N, 2) of face landmark pixel coordinates
        
        Returns:
            pitch (float): degrees, + = up, - = down
            yaw   (float): degrees, + = right, - = left
            roll  (float): degrees, + = right tilt, - = left tilt
            rvec  (ndarray): rotation vector (for axis drawing)
            tvec  (ndarray): translation vector (for axis drawing)
        """
        try:
            # Extract the 6 model landmark positions from detected 2D landmarks
            image_points = np.array(
                [landmarks_2d[idx] for idx in FACE_3D_LANDMARK_IDS],
                dtype=np.float64
            )

            # Solve PnP: find camera-relative rotation and translation
            success, rvec, tvec = cv2.solvePnP(
                FACE_3D_MODEL,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return self._smooth_pitch, self._smooth_yaw, self._smooth_roll, None, None

            self.rvec = rvec
            self.tvec = tvec

            # Convert rotation vector → rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)

            # Decompose rotation matrix into Euler angles via RQ decomposition
            pitch, yaw, roll = self._rotation_matrix_to_euler(rmat)

            # Smooth angles
            self._smooth_pitch = self._smooth(self._smooth_pitch, pitch)
            self._smooth_yaw   = self._smooth(self._smooth_yaw,   yaw)
            self._smooth_roll  = self._smooth(self._smooth_roll,  roll)

            return self._smooth_pitch, self._smooth_yaw, self._smooth_roll, rvec, tvec

        except (cv2.error, IndexError, Exception) as e:
            logger.debug(f"[HeadPose] solvePnP error: {e}")
            return self._smooth_pitch, self._smooth_yaw, self._smooth_roll, None, None

    def _rotation_matrix_to_euler(self, rmat: np.ndarray) -> Tuple[float, float, float]:
        """
        Decompose 3×3 rotation matrix to Euler angles (degrees).
        
        Uses OpenCV's RQDecomp3x3 which gives stable angles across
        the full rotation range. Returns angles as (pitch, yaw, roll).
        """
        # projectMatrix decomposition returns angles in degrees
        _, _, _, _, _, _, euler = cv2.RQDecomp3x3(rmat)
        # euler = (rx, ry, rz) in degrees already
        pitch = euler[0]   # X rotation = pitch (up/down)
        yaw   = euler[1]   # Y rotation = yaw  (left/right)
        roll  = euler[2]   # Z rotation = roll (tilt)
        return pitch, yaw, roll

    def _smooth(self, prev: float, current: float) -> float:
        """Exponential moving average smoothing."""
        return prev * (1 - self._alpha) + current * self._alpha

    def assess_pose(self, pitch: float, yaw: float, roll: float) -> dict:
        """
        Assess head pose and return status flags.
        
        Returns:
            dict with keys: is_looking_away, is_nodding, direction
        """
        is_looking_away = abs(yaw) > config.HEAD_YAW_THRESHOLD
        is_nodding_down = pitch < -config.HEAD_PITCH_DOWN_THRESHOLD
        is_nodding_up   = pitch >  config.HEAD_PITCH_UP_THRESHOLD
        is_tilted       = abs(roll) > config.HEAD_ROLL_THRESHOLD

        if abs(yaw) > config.HEAD_YAW_THRESHOLD:
            direction = "RIGHT" if yaw > 0 else "LEFT"
        elif is_nodding_down:
            direction = "DOWN"
        elif is_nodding_up:
            direction = "UP"
        else:
            direction = "FORWARD"

        # Update consecutive off-pose frame counter
        if is_looking_away or is_nodding_down:
            self.off_pose_frames += 1
        else:
            self.off_pose_frames = max(0, self.off_pose_frames - 1)

        head_pose_alert = self.off_pose_frames >= config.HEAD_POSE_ALERT_FRAMES

        return {
            "is_looking_away": is_looking_away,
            "is_nodding_down": is_nodding_down,
            "is_nodding_up":   is_nodding_up,
            "is_tilted":       is_tilted,
            "direction":       direction,
            "head_pose_alert": head_pose_alert,
            "off_pose_frames": self.off_pose_frames,
        }

    def compute_head_pose_score(self, pitch: float, yaw: float) -> float:
        """
        Compute normalised head pose risk score in [0, 1].
        
        Higher score = head is further from forward-facing position.
        Used as input to the composite fatigue_score formula.
        """
        yaw_norm   = min(1.0, abs(yaw)   / (config.HEAD_YAW_THRESHOLD * 2))
        pitch_norm = min(1.0, abs(min(pitch, 0)) / (config.HEAD_PITCH_DOWN_THRESHOLD * 2))
        # Weighted combination (yaw is generally more indicative of distraction)
        score = 0.6 * yaw_norm + 0.4 * pitch_norm
        return float(np.clip(score, 0.0, 1.0))

    def get_camera_matrix(self) -> np.ndarray:
        return self.camera_matrix

    def get_dist_coeffs(self) -> np.ndarray:
        return self.dist_coeffs


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("HeadPoseEstimator module loaded successfully.")
    estimator = HeadPoseEstimator(1280, 720)
    print(f"Camera matrix:\n{estimator.camera_matrix}")
    print(f"Dist coeffs: {estimator.dist_coeffs.T}")
