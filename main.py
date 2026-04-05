"""
main.py - Neuro-Drive: Human-Centric Driver Monitoring System
=============================================================
Main entry point. Orchestrates:
    - MediaPipe face mesh
    - Head pose estimation
    - Gaze tracking
    - Fatigue detection engine
    - Visual overlay rendering
    - Optional FastAPI REST server (background thread)

Usage:
    python main.py [--source 0] [--width 1280] [--height 720]
                   [--no-api] [--no-calibration] [--video path/to/file]
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import threading
import time
import sys
import os
import logging
from typing import Optional

import config
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS,
    LEFT_EYE_EAR_INDICES, RIGHT_EYE_EAR_INDICES,
    LEFT_EYE_INDICES, RIGHT_EYE_INDICES,
    LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES,
    DRAW_FACE_MESH, DRAW_IRIS, CALIBRATION_ENABLED,
)
from utils import (
    setup_logger, FPSCounter, CSVDataLogger, DriverCalibration,
    compute_ear, compute_mar, assess_frame_quality, preprocess_frame,
    draw_landmarks_subset, draw_eye_contour, draw_iris, draw_head_pose_axes,
    draw_hud_panel, draw_calibration_screen,
)
from head_pose import HeadPoseEstimator
from gaze_tracking import GazeTracker
from fatigue_detection import FatigueEngine

logger = setup_logger("neuro_drive")


# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────────────────────────────────────

def build_face_mesh():
    """Initialise and return a MediaPipe FaceMesh instance."""
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        max_num_faces=config.MAX_NUM_FACES,
        refine_landmarks=config.REFINE_LANDMARKS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_landmarks(face_landmarks, w: int, h: int) -> np.ndarray:
    """
    Convert MediaPipe NormalisedLandmarkList to pixel-coordinate numpy array.
    
    Returns:
        Array of shape (N, 2) with (x, y) in pixels
    """
    landmarks = face_landmarks.landmark
    coords = np.array(
        [(lm.x * w, lm.y * h) for lm in landmarks],
        dtype=np.float32
    )
    return coords


# ─────────────────────────────────────────────────────────────────────────────
# DRAW FACE OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def draw_face_overlay(frame: np.ndarray, landmarks_2d: np.ndarray,
                      ear_avg: float, mar: float, alert_status: str):
    """
    Draw face tracking overlay:
        - Eye contours (colour-coded by EAR state)
        - Iris circles
        - Mouth outline (colour-coded by MAR state)
        - Nose bridge point
    """
    # ── Eye contours ──────────────────────────────────────────────────────
    eye_color = (0, 220, 0) if ear_avg > config.EAR_DROWSY_THRESHOLD else (0, 100, 255)
    draw_eye_contour(frame, landmarks_2d, LEFT_EYE_INDICES,  eye_color, thickness=1)
    draw_eye_contour(frame, landmarks_2d, RIGHT_EYE_INDICES, eye_color, thickness=1)

    # ── Iris (requires refine_landmarks=True) ─────────────────────────────
    if DRAW_IRIS and len(landmarks_2d) > 477:
        iris_color = (0, 255, 255)
        draw_iris(frame, landmarks_2d, LEFT_IRIS_INDICES,  iris_color)
        draw_iris(frame, landmarks_2d, RIGHT_IRIS_INDICES, iris_color)

    # ── Mouth ─────────────────────────────────────────────────────────────
    mouth_color = (0, 165, 255) if mar > config.MAR_YAWN_THRESHOLD else (200, 200, 50)
    mouth_outer = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                   375, 321, 405, 314, 17, 84, 181, 91, 146]
    draw_eye_contour(frame, landmarks_2d, mouth_outer, mouth_color, thickness=1)

    # ── Nose tip marker ───────────────────────────────────────────────────
    if len(landmarks_2d) > 1:
        nose_pt = tuple(landmarks_2d[1].astype(int))
        cv2.circle(frame, nose_pt, 3, (200, 200, 255), -1)

    # ── EAR/MAR text next to eyes ─────────────────────────────────────────
    if len(landmarks_2d) > 362:
        le_pt = landmarks_2d[362].astype(int)
        re_pt = landmarks_2d[133].astype(int)
        cv2.putText(frame, f"{ear_avg:.2f}", (le_pt[0] - 40, le_pt[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, eye_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{ear_avg:.2f}", (re_pt[0] + 5, re_pt[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, eye_color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL FASTAPI SERVER
# ─────────────────────────────────────────────────────────────────────────────

# Shared state for REST API
_api_state: dict = {
    "fatigue_score": 0.0,
    "alert_status":  "SAFE",
    "fps":           0.0,
    "metrics":       {},
    "stats":         {},
}


def start_api_server():
    """Start the FastAPI server in a background thread."""
    try:
        import uvicorn
        from api import app
        logger.info(f"[API] Starting REST API on {config.API_HOST}:{config.API_PORT}")
        uvicorn.run(app, host=config.API_HOST, port=config.API_PORT,
                    log_level="error")
    except ImportError:
        logger.warning("[API] uvicorn/fastapi not installed. REST API disabled.")
    except Exception as e:
        logger.error(f"[API] Failed to start: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run(source: int | str = CAMERA_INDEX,
        width:  int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        enable_api: bool = True,
        enable_calibration: bool = True):
    """
    Main driver monitoring loop.
    
    Args:
        source:             Camera index or video file path
        width/height:       Frame resolution
        enable_api:         Launch FastAPI server
        enable_calibration: Run per-user EAR/MAR calibration on startup
    """
    logger.info("=" * 60)
    logger.info("  NEURO-DRIVE: Human-Centric Driver Monitoring System")
    logger.info("=" * 60)
    logger.info(f"  Source: {source} | Resolution: {width}×{height}")

    # ── Optional API server ────────────────────────────────────────────────
    if enable_api and config.API_ENABLE:
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()

    # ── Camera / Video capture ─────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"  Camera opened: actual resolution {actual_w}×{actual_h}")

    # ── Subsystem initialisation ───────────────────────────────────────────
    face_mesh        = build_face_mesh()
    head_estimator   = HeadPoseEstimator(actual_w, actual_h)
    gaze_tracker     = GazeTracker()
    csv_logger       = CSVDataLogger()
    fatigue_engine   = FatigueEngine(csv_logger=csv_logger)
    fps_counter      = FPSCounter()
    calibration      = DriverCalibration() if enable_calibration else None

    # ── State variables ────────────────────────────────────────────────────
    frame_idx       = 0
    alert_status    = "SAFE"
    fatigue_score   = 0.0
    metrics         = {}
    no_face_since   = None
    last_ear_avg    = config.EAR_DROWSY_THRESHOLD  # sensible default
    last_mar        = 0.2
    last_pitch      = 0.0
    last_yaw        = 0.0
    last_roll       = 0.0
    last_gaze_x     = 0.5
    last_gaze_y     = 0.5
    last_hp_score   = 0.0
    last_gz_score   = 0.0
    last_rvec       = None
    last_tvec       = None

    logger.info("  Press 'q' to quit | 'c' to recalibrate | 's' for stats")
    logger.info("=" * 60)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh_connections = mp.solutions.face_mesh

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):
                    logger.info("Video file ended.")
                    break
                logger.warning("Frame capture failed, retrying...")
                time.sleep(0.01)
                continue

            frame_idx += 1
            fps_counter.tick()
            current_fps = fps_counter.fps

            # ── Frame skip logic ───────────────────────────────────────────
            if config.SKIP_FRAMES > 0 and frame_idx % (config.SKIP_FRAMES + 1) != 0:
                # Still draw HUD on skipped frames using cached values
                draw_hud_panel(frame, metrics, alert_status, fatigue_score, current_fps)
                cv2.imshow("Neuro-Drive", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ── Frame quality assessment ───────────────────────────────────
            quality = assess_frame_quality(frame)
            brightness = quality["brightness"]
            is_blurry  = quality["is_blurry"]

            # ── Adaptive preprocessing ─────────────────────────────────────
            display_frame = preprocess_frame(frame.copy(), brightness)

            # ── MediaPipe inference ────────────────────────────────────────
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True

            face_detected = bool(results.multi_face_landmarks)

            if face_detected:
                no_face_since = None
                face_lms = results.multi_face_landmarks[0]
                lm2d = extract_landmarks(face_lms, actual_w, actual_h)

                # ── Calibration phase ──────────────────────────────────────
                if calibration and not calibration.done:
                    ear_l_raw = compute_ear(lm2d, LEFT_EYE_EAR_INDICES)
                    ear_r_raw = compute_ear(lm2d, RIGHT_EYE_EAR_INDICES)
                    mar_raw   = compute_mar(lm2d)
                    calibration.update((ear_l_raw + ear_r_raw) / 2, mar_raw)
                    draw_calibration_screen(display_frame, calibration.progress)
                    cv2.imshow("Neuro-Drive", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                elif calibration and calibration.done and not hasattr(calibration, '_applied'):
                    fatigue_engine.set_calibration(
                        calibration.ear_threshold,
                        calibration.mar_threshold
                    )
                    gaze_tracker.finalize_calibration()
                    calibration._applied = True
                    logger.info("[Main] Calibration applied.")

                # ── EAR ────────────────────────────────────────────────────
                ear_left  = compute_ear(lm2d, LEFT_EYE_EAR_INDICES)
                ear_right = compute_ear(lm2d, RIGHT_EYE_EAR_INDICES)
                ear_avg   = (ear_left + ear_right) / 2.0

                # ── MAR ────────────────────────────────────────────────────
                mar = compute_mar(lm2d)

                # ── Head Pose ──────────────────────────────────────────────
                pitch, yaw, roll, rvec, tvec = head_estimator.estimate(lm2d)
                pose_assessment = head_estimator.assess_pose(pitch, yaw, roll)
                hp_score = head_estimator.compute_head_pose_score(pitch, yaw)
                head_away = pose_assessment["is_looking_away"] or pose_assessment["is_nodding_down"]

                # ── Gaze ───────────────────────────────────────────────────
                gaze_x, gaze_y, gaze_dir, gz_score = gaze_tracker.update(lm2d)

                # Accumulate gaze calibration samples (first N seconds)
                if calibration and not calibration.done:
                    gaze_tracker.calibrate(gaze_x, gaze_y)

                # ── Fatigue Engine ─────────────────────────────────────────
                fatigue_score, alert_status = fatigue_engine.update(
                    ear_left=ear_left, ear_right=ear_right,
                    mar=mar,
                    pitch=pitch, yaw=yaw,
                    gaze_x=gaze_x, gaze_y=gaze_y,
                    head_pose_score=hp_score,
                    gaze_score=gz_score,
                    face_detected=True,
                    head_away=head_away,
                    brightness=brightness,
                    is_blurry=is_blurry,
                )

                # Cache for skip frames
                last_ear_avg  = ear_avg
                last_mar      = mar
                last_pitch    = pitch
                last_yaw      = yaw
                last_roll     = roll
                last_gaze_x   = gaze_x
                last_gaze_y   = gaze_y
                last_hp_score = hp_score
                last_gz_score = gz_score
                last_rvec     = rvec
                last_tvec     = tvec

                # ── Face Overlay ───────────────────────────────────────────
                if DRAW_FACE_MESH:
                    # Draw full mesh (subtle, low opacity)
                    mesh_overlay = display_frame.copy()
                    mp_drawing.draw_landmarks(
                        image=mesh_overlay,
                        landmark_list=face_lms,
                        connections=mp_face_mesh_connections.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    cv2.addWeighted(mesh_overlay, 0.15, display_frame, 0.85, 0, display_frame)

                draw_face_overlay(display_frame, lm2d, ear_avg, mar, alert_status)

                # ── Head Pose Axes ─────────────────────────────────────────
                if rvec is not None and tvec is not None:
                    nose_pt = tuple(lm2d[1].astype(int))
                    draw_head_pose_axes(
                        display_frame, nose_pt, rvec, tvec,
                        head_estimator.get_camera_matrix(),
                        head_estimator.get_dist_coeffs(),
                        axis_length=40.0
                    )

                # Head direction label
                dir_label = pose_assessment["direction"]
                dir_color = (0, 220, 0) if dir_label == "FORWARD" else (0, 100, 255)
                h_frame, w_frame = display_frame.shape[:2]
                cv2.putText(display_frame, f"HEAD: {dir_label}",
                            (w_frame - 240, 35),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, dir_color, 1, cv2.LINE_AA)

                # Gaze indicator widget
                gaze_tracker.draw_gaze_indicator(display_frame)

                metrics = {
                    "ear_left":   ear_left,
                    "ear_right":  ear_right,
                    "ear_avg":    ear_avg,
                    "mar":        mar,
                    "yawn_count": fatigue_engine.yawn_tracker.yawn_count,
                    "pitch":      pitch,
                    "yaw":        yaw,
                    "roll":       roll,
                    "gaze_x":     gaze_x,
                    "gaze_y":     gaze_y,
                    "brightness": brightness,
                    "is_blurry":  is_blurry,
                }

            else:
                # ── No face detected ───────────────────────────────────────
                if no_face_since is None:
                    no_face_since = time.time()

                fatigue_score, alert_status = fatigue_engine.update(
                    ear_left=last_ear_avg, ear_right=last_ear_avg,
                    mar=last_mar,
                    pitch=last_pitch, yaw=last_yaw,
                    gaze_x=last_gaze_x, gaze_y=last_gaze_y,
                    head_pose_score=last_hp_score,
                    gaze_score=last_gz_score,
                    face_detected=False,
                    head_away=False,
                    brightness=brightness,
                    is_blurry=is_blurry,
                )

                no_face_duration = time.time() - no_face_since
                h_frame, w_frame = display_frame.shape[:2]
                cv2.putText(display_frame, f"NO FACE ({no_face_duration:.1f}s)",
                            (w_frame // 2 - 120, h_frame // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            # ── HUD Panel ─────────────────────────────────────────────────
            draw_hud_panel(display_frame, metrics, alert_status,
                           fatigue_score, current_fps)

            # ── Update API state ───────────────────────────────────────────
            _api_state.update({
                "fatigue_score": round(fatigue_score, 4),
                "alert_status":  alert_status,
                "fps":           round(current_fps, 1),
                "metrics":       {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in metrics.items()},
                "stats":         fatigue_engine.get_stats(),
            })

            # ── Display ───────────────────────────────────────────────────
            cv2.imshow("Neuro-Drive", display_frame)

            # ── Key handling ───────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested by user.")
                break
            elif key == ord('c'):
                logger.info("Recalibrating...")
                calibration = DriverCalibration()
            elif key == ord('s'):
                stats = fatigue_engine.get_stats()
                logger.info(f"[Stats] {stats}")
                print(f"\n{'='*40}\nSession Stats:\n{stats}\n{'='*40}\n")

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        csv_logger.close()
        stats = fatigue_engine.get_stats()
        logger.info(f"Session complete. Final stats: {stats}")
        print(f"\n{'='*50}")
        print("  NEURO-DRIVE SESSION SUMMARY")
        print(f"{'='*50}")
        for k, v in stats.items():
            print(f"  {k:25s}: {v}")
        print(f"{'='*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Neuro-Drive: Human-Centric Driver Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Use default webcam
  python main.py --source 1             # Use second camera
  python main.py --video test.mp4       # Use video file
  python main.py --no-api               # Disable REST API
  python main.py --no-calibration       # Skip calibration
  python main.py --width 640 --height 480  # Lower resolution for faster FPS
        """
    )
    parser.add_argument("--source",          type=int,   default=CAMERA_INDEX,
                        help="Camera device index (default: 0)")
    parser.add_argument("--video",           type=str,   default=None,
                        help="Path to video file (overrides --source)")
    parser.add_argument("--width",           type=int,   default=FRAME_WIDTH,
                        help=f"Frame width (default: {FRAME_WIDTH})")
    parser.add_argument("--height",          type=int,   default=FRAME_HEIGHT,
                        help=f"Frame height (default: {FRAME_HEIGHT})")
    parser.add_argument("--no-api",          action="store_true",
                        help="Disable FastAPI REST server")
    parser.add_argument("--no-calibration",  action="store_true",
                        help="Skip EAR/MAR calibration phase")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source = args.video if args.video else args.source
    run(
        source=source,
        width=args.width,
        height=args.height,
        enable_api=not args.no_api,
        enable_calibration=not args.no_calibration,
    )
