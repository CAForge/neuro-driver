"""
fatigue_detection.py - Fatigue Detection Engine for Neuro-Drive
===============================================================
Implements the multi-factor fatigue scoring model:

    fatigue_score = w1*EAR_score + w2*MAR_score + w3*head_pose_score + w4*gaze_score

Where each sub-score is normalised to [0, 1] and the weights sum to 1.

Components:
    EAR score:        Derived from eye aspect ratio (eye closure level)
    MAR score:        Derived from mouth aspect ratio (yawning)
    Head pose score:  Derived from pitch/yaw angles
    Gaze score:       Derived from iris deviation from center

Alerts:
    fatigue_score >= FATIGUE_SCORE_DROWSY   → DROWSY warning
    fatigue_score >= FATIGUE_SCORE_CRITICAL → CRITICAL alert
"""

import time
import numpy as np
import logging
from typing import Tuple, List, Optional, Dict
from collections import deque

import config
from utils import compute_ear, compute_mar, play_alert_sound, CSVDataLogger

logger = logging.getLogger("neuro_drive")


# ─────────────────────────────────────────────────────────────────────────────
# YAWN EVENT TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class YawnTracker:
    """
    Tracks yawn events over a rolling time window.
    A "yawn" is detected when MAR exceeds threshold for a minimum number
    of consecutive frames, then falls back below threshold.
    """

    def __init__(self):
        self._in_yawn     = False
        self._yawn_frames = 0
        self._yawn_events: deque = deque()   # timestamps of completed yawns
        self.yawn_count   = 0                # yawns in current window

    def update(self, mar: float, threshold: float) -> bool:
        """
        Update yawn state. Returns True when a new yawn is completed.
        
        A yawn is recorded when:
            - MAR rises above threshold for >= MAR_YAWN_FRAMES frames (onset)
            - MAR then falls below threshold (completion)
        """
        is_mouth_open = mar > threshold
        new_yawn = False

        if is_mouth_open:
            self._yawn_frames += 1
            self._in_yawn = True
        else:
            if self._in_yawn and self._yawn_frames >= config.MAR_YAWN_FRAMES:
                # Completed yawn
                now = time.time()
                self._yawn_events.append(now)
                self.yawn_count += 1
                new_yawn = True
                logger.info(f"[Fatigue] Yawn detected! Total in window: {self._count_recent_yawns()}")
                play_alert_sound("YAWNING")
            self._in_yawn     = False
            self._yawn_frames = 0

        # Prune old yawn events outside the time window
        self._prune_events()
        return new_yawn

    def _prune_events(self):
        cutoff = time.time() - config.MAR_YAWN_WINDOW_SECONDS
        while self._yawn_events and self._yawn_events[0] < cutoff:
            self._yawn_events.popleft()

    def _count_recent_yawns(self) -> int:
        self._prune_events()
        return len(self._yawn_events)

    @property
    def is_yawning(self) -> bool:
        return self._in_yawn and self._yawn_frames >= config.MAR_YAWN_FRAMES

    @property
    def excess_yawning_alert(self) -> bool:
        return self._count_recent_yawns() >= config.MAR_YAWN_COUNT_ALERT

    def get_mar_score(self, mar: float, threshold: float) -> float:
        """
        Return normalised MAR score [0, 1].
        
        Combines:
            - Immediate mouth openness (is actively yawning?)
            - Cumulative yawn frequency in the time window
        """
        # Immediate score: how far is MAR above baseline?
        immediate = min(1.0, max(0.0, (mar - threshold * 0.5) / threshold))

        # Cumulative score: ratio of recent yawns to alert threshold
        cumulative = min(1.0, self._count_recent_yawns() / config.MAR_YAWN_COUNT_ALERT)

        # Combine: immediate has more weight during active yawn
        score = 0.6 * immediate + 0.4 * cumulative
        return float(score)


# ─────────────────────────────────────────────────────────────────────────────
# EYE STATE TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class EyeStateTracker:
    """
    Tracks EAR state and computes eye-closure-based drowsiness score.
    Handles PERCLOS (Percentage of Eye Closure) as a secondary metric.
    """

    def __init__(self):
        self._closed_frames      = 0
        self._drowsy_frames      = 0
        self._total_frames       = 0
        self._closed_frames_buf: deque = deque(maxlen=300)  # last 300 frames for PERCLOS

    def update(self, ear: float, closed_threshold: float,
               drowsy_threshold: float) -> Tuple[bool, bool]:
        """
        Update eye state.
        
        Returns:
            (is_closed, is_drowsy) booleans
        """
        self._total_frames += 1
        is_closed = ear < closed_threshold
        is_drowsy = ear < drowsy_threshold and not is_closed

        self._closed_frames_buf.append(1 if is_closed else 0)

        if is_closed:
            self._closed_frames += 1
            self._drowsy_frames = 0
        elif is_drowsy:
            self._drowsy_frames += 1
            self._closed_frames = 0
        else:
            self._closed_frames = max(0, self._closed_frames - 1)
            self._drowsy_frames = max(0, self._drowsy_frames - 2)

        return is_closed, is_drowsy

    @property
    def prolonged_closure_alert(self) -> bool:
        return self._closed_frames >= config.EAR_CLOSED_FRAMES

    @property
    def prolonged_drowsy_alert(self) -> bool:
        return self._drowsy_frames >= config.EAR_DROWSY_FRAMES

    @property
    def perclos(self) -> float:
        """PERCLOS: fraction of recent frames where eyes are closed (0–1)."""
        if not self._closed_frames_buf:
            return 0.0
        return float(np.mean(self._closed_frames_buf))

    def get_ear_score(self, ear: float, closed_threshold: float,
                      drowsy_threshold: float) -> float:
        """
        Normalised EAR risk score [0, 1].
        
        Combines:
            - Immediate EAR value (lower = worse)
            - PERCLOS (fraction of time eyes are closed)
            - Consecutive closure frames
        """
        # Immediate: map EAR to [0, 1] where EAR=0 → score=1, EAR=open → score=0
        open_ear = drowsy_threshold * 1.4  # approximate open EAR
        immediate = 1.0 - min(1.0, max(0.0, (ear - closed_threshold) /
                                        (open_ear - closed_threshold + 1e-6)))

        # PERCLOS contribution
        perclos_score = min(1.0, self.perclos * 5.0)  # 20% closure = max score

        # Consecutive frames
        consec_score = min(1.0, self._closed_frames / config.EAR_CLOSED_FRAMES)

        score = 0.5 * immediate + 0.3 * perclos_score + 0.2 * consec_score
        return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# DISTRACTION TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class DistractionTracker:
    """
    Tracks distraction events based on:
        1. Face not visible (no detection)
        2. Head pose off-road for > threshold seconds
        3. Gaze off-road for > threshold seconds
    """

    def __init__(self):
        self._distraction_start: Optional[float] = None
        self._no_face_start: Optional[float]     = None
        self.distraction_duration = 0.0
        self.no_face_duration     = 0.0
        self.is_distracted        = False
        self.no_face_alert        = False

    def update(self, face_detected: bool, head_away: bool, gaze_away: bool):
        now = time.time()

        # ── No face detected ──────────────────────────────────────────────
        if not face_detected:
            if self._no_face_start is None:
                self._no_face_start = now
            self.no_face_duration = now - self._no_face_start
            self.no_face_alert    = self.no_face_duration >= config.NO_FACE_TIME_THRESHOLD
        else:
            self._no_face_start   = None
            self.no_face_duration = 0.0
            self.no_face_alert    = False

        # ── Distraction (head/gaze off-road) ──────────────────────────────
        is_distracted_now = face_detected and (head_away or gaze_away)
        if is_distracted_now:
            if self._distraction_start is None:
                self._distraction_start = now
            self.distraction_duration = now - self._distraction_start
            self.is_distracted = (
                self.distraction_duration >= config.DISTRACTION_TIME_THRESHOLD
            )
        else:
            self._distraction_start   = None
            self.distraction_duration = 0.0
            self.is_distracted        = False


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE FATIGUE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class FatigueEngine:
    """
    Main fatigue detection engine.
    
    Computes the composite fatigue score:
        score = w1*ear_score + w2*mar_score + w3*head_score + w4*gaze_score
    
    And derives the alert level:
        SAFE / DROWSY / YAWNING / DISTRACTED / CRITICAL / NO_FACE
    """

    def __init__(self, csv_logger: Optional[CSVDataLogger] = None):
        self.eye_tracker        = EyeStateTracker()
        self.yawn_tracker       = YawnTracker()
        self.distraction_tracker = DistractionTracker()

        # Calibrated thresholds (updated after calibration)
        self.ear_closed_thresh  = config.EAR_CLOSED_THRESHOLD
        self.ear_drowsy_thresh  = config.EAR_DROWSY_THRESHOLD
        self.mar_yawn_thresh    = config.MAR_YAWN_THRESHOLD

        # Smoothed fatigue score
        self._fatigue_score     = 0.0

        # Alert cooldown
        self._last_alert_times: Dict[str, float] = {}

        # CSV logger
        self._csv_logger = csv_logger

        # Statistics
        self.total_alerts = 0
        self.session_start = time.time()

    def set_calibration(self, ear_threshold: float, mar_threshold: float):
        """Apply calibrated thresholds from DriverCalibration."""
        self.ear_closed_thresh = ear_threshold
        self.ear_drowsy_thresh = ear_threshold * (
            config.EAR_DROWSY_THRESHOLD / config.EAR_CLOSED_THRESHOLD
        )
        self.mar_yawn_thresh = mar_threshold
        logger.info(f"[Fatigue] Using calibrated thresholds: "
                    f"EAR={self.ear_closed_thresh:.3f}, MAR={self.mar_yawn_thresh:.3f}")

    def update(self,
               ear_left: float, ear_right: float,
               mar: float,
               pitch: float, yaw: float,
               gaze_x: float, gaze_y: float,
               head_pose_score: float,
               gaze_score: float,
               face_detected: bool,
               head_away: bool,
               brightness: float = 128.0,
               is_blurry: bool = False) -> Tuple[float, str]:
        """
        Full fatigue update cycle.
        
        Args:
            ear_left, ear_right: Eye aspect ratios
            mar:                 Mouth aspect ratio
            pitch, yaw:          Head pose angles (degrees)
            gaze_x, gaze_y:      Normalised iris position
            head_pose_score:     Normalised head pose risk [0-1]
            gaze_score:          Normalised gaze deviation [0-1]
            face_detected:       Whether a face is visible
            head_away:           Whether head is off-road
            brightness:          Frame mean brightness
            is_blurry:           Whether frame quality is poor
        
        Returns:
            (fatigue_score, alert_status)
        """
        # Adaptive EAR threshold (low light)
        from utils import get_adaptive_ear_threshold
        closed_thresh = get_adaptive_ear_threshold(self.ear_closed_thresh, brightness)
        drowsy_thresh = get_adaptive_ear_threshold(self.ear_drowsy_thresh, brightness)

        # ── EAR ──────────────────────────────────────────────────────────
        ear_avg = (ear_left + ear_right) / 2.0
        is_closed, is_drowsy = self.eye_tracker.update(ear_avg, closed_thresh, drowsy_thresh)
        ear_score = self.eye_tracker.get_ear_score(ear_avg, closed_thresh, drowsy_thresh)

        # ── MAR ──────────────────────────────────────────────────────────
        self.yawn_tracker.update(mar, self.mar_yawn_thresh)
        mar_score = self.yawn_tracker.get_mar_score(mar, self.mar_yawn_thresh)

        # ── Distraction ───────────────────────────────────────────────────
        gaze_away = (abs(gaze_x - 0.5) > config.GAZE_OFF_ROAD_THRESHOLD or
                     abs(gaze_y - 0.5) > config.GAZE_OFF_ROAD_THRESHOLD)
        self.distraction_tracker.update(face_detected, head_away, gaze_away)

        # ── Composite Fatigue Score ───────────────────────────────────────
        raw_score = (
            config.FATIGUE_WEIGHT_EAR  * ear_score   +
            config.FATIGUE_WEIGHT_MAR  * mar_score   +
            config.FATIGUE_WEIGHT_HEAD * head_pose_score +
            config.FATIGUE_WEIGHT_GAZE * gaze_score
        )
        raw_score = float(np.clip(raw_score, 0.0, 1.0))

        # Exponential smoothing
        alpha = config.FATIGUE_SMOOTHING_ALPHA
        self._fatigue_score = (alpha * raw_score +
                               (1 - alpha) * self._fatigue_score)

        # ── Determine Alert Status ────────────────────────────────────────
        alert = self._determine_alert(is_blurry=is_blurry)

        # ── Logging ───────────────────────────────────────────────────────
        if self._csv_logger:
            self._csv_logger.log({
                "fps":           0,   # filled by main loop
                "ear_left":      round(ear_left, 4),
                "ear_right":     round(ear_right, 4),
                "ear_avg":       round(ear_avg, 4),
                "mar":           round(mar, 4),
                "pitch":         round(pitch, 2),
                "yaw":           round(yaw, 2),
                "roll":          0,
                "gaze_x":        round(gaze_x, 3),
                "gaze_y":        round(gaze_y, 3),
                "fatigue_score": round(self._fatigue_score, 4),
                "alert_status":  alert,
                "brightness":    round(brightness, 1),
                "is_blurry":     is_blurry,
            })

        return self._fatigue_score, alert

    def _determine_alert(self, is_blurry: bool = False) -> str:
        """
        Determine the current alert level based on all subsystem states.
        Priority (high to low): NO_FACE → CRITICAL → DISTRACTED → DROWSY → YAWNING → SAFE
        """
        # No face detected
        if self.distraction_tracker.no_face_alert:
            self._fire_alert("NO_FACE")
            return "NO_FACE"

        # Critical fatigue
        if self._fatigue_score >= config.FATIGUE_SCORE_CRITICAL:
            self._fire_alert("CRITICAL")
            return "CRITICAL"

        # Distracted
        if self.distraction_tracker.is_distracted:
            self._fire_alert("DISTRACTED")
            return "DISTRACTED"

        # Drowsy
        if (self._fatigue_score >= config.FATIGUE_SCORE_DROWSY or
                self.eye_tracker.prolonged_closure_alert or
                self.eye_tracker.prolonged_drowsy_alert):
            self._fire_alert("DROWSY")
            return "DROWSY"

        # Yawning
        if self.yawn_tracker.is_yawning or self.yawn_tracker.excess_yawning_alert:
            self._fire_alert("YAWNING")
            return "YAWNING"

        return "SAFE"

    def _fire_alert(self, alert_type: str):
        """
        Fire audio/log alert if cooldown has elapsed.
        Prevents alert spam.
        """
        now = time.time()
        last = self._last_alert_times.get(alert_type, 0)
        if now - last < config.ALERT_COOLDOWN_SECONDS:
            return
        self._last_alert_times[alert_type] = now
        self.total_alerts += 1

        logger.warning(f"[ALERT] *** {alert_type} *** | "
                       f"fatigue={self._fatigue_score:.3f} | "
                       f"session={int(time.time() - self.session_start)}s")
        play_alert_sound(alert_type)

    @property
    def fatigue_score(self) -> float:
        return self._fatigue_score

    def get_stats(self) -> dict:
        return {
            "session_duration_s": int(time.time() - self.session_start),
            "total_alerts":       self.total_alerts,
            "total_yawns":        self.yawn_tracker.yawn_count,
            "perclos":            round(self.eye_tracker.perclos, 3),
            "fatigue_score":      round(self._fatigue_score, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = FatigueEngine()

    # Simulate drowsy driving (low EAR)
    print("=== Simulating drowsiness ===")
    for i in range(40):
        score, alert = engine.update(
            ear_left=0.18, ear_right=0.19,  # eyes nearly closed
            mar=0.2,
            pitch=0, yaw=5,
            gaze_x=0.5, gaze_y=0.5,
            head_pose_score=0.05,
            gaze_score=0.05,
            face_detected=True,
            head_away=False
        )
        if i % 10 == 0:
            print(f"Frame {i:3d}: score={score:.3f}  alert={alert}")

    print("\n=== Simulating yawning ===")
    for i in range(30):
        score, alert = engine.update(
            ear_left=0.30, ear_right=0.30,
            mar=0.75,  # wide open mouth
            pitch=0, yaw=0,
            gaze_x=0.5, gaze_y=0.5,
            head_pose_score=0.0,
            gaze_score=0.0,
            face_detected=True,
            head_away=False
        )
        if i % 10 == 0:
            print(f"Frame {i:3d}: score={score:.3f}  alert={alert}")

    print(f"\nSession stats: {engine.get_stats()}")
