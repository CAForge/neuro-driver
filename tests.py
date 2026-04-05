"""
tests.py - Test Suite for Neuro-Drive
======================================
Covers:
    1. Unit tests for EAR/MAR computation
    2. Simulated drowsiness scenario
    3. Simulated yawning scenario
    4. Simulated head-turn (distraction) scenario
    5. Edge cases (no face, blurry frame, low light)
    6. Fatigue score formula validation
    7. Head pose estimation (mock)
    8. Gaze tracker direction classification
    9. Integration test: full FatigueEngine pipeline
   10. Config validation

Run:
    python tests.py
    python -m pytest tests.py -v
"""

import sys
import math
import time
import numpy as np
import unittest

# We import the modules under test
from utils import compute_ear, compute_mar, FPSCounter, DriverCalibration, assess_frame_quality
from head_pose import HeadPoseEstimator
from gaze_tracking import GazeTracker
from fatigue_detection import FatigueEngine, YawnTracker, EyeStateTracker
import config


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_landmarks(n: int = 478) -> np.ndarray:
    """Create dummy landmark array of shape (n, 2)."""
    return np.random.rand(n, 2).astype(np.float32) * 100 + 200


def set_eye_landmarks(lm: np.ndarray, indices: list,
                      ear_target: float, width: float = 30.0) -> np.ndarray:
    """
    Modify eye landmark positions to achieve a target EAR.
    
    EAR = (A + B) / (2C) where A, B = vertical, C = horizontal
    So set vertical separation = ear_target * width
    """
    lm = lm.copy()
    p1, p2, p3, p4, p5, p6 = indices
    cx, cy = 300.0, 300.0

    # Horizontal: p1 (left), p4 (right)
    lm[p1] = [cx - width / 2, cy]
    lm[p4] = [cx + width / 2, cy]

    # Vertical (p2 above, p6 below top-right area; p3 above, p5 below top-left)
    v = ear_target * width / 2  # half of desired (A+B)/2
    lm[p2] = [cx - width / 4, cy - v]
    lm[p6] = [cx - width / 4, cy + v]
    lm[p3] = [cx + width / 4, cy - v]
    lm[p5] = [cx + width / 4, cy + v]

    return lm


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEAR(unittest.TestCase):
    """Tests for Eye Aspect Ratio computation."""

    def _make_eye(self, ear_value: float) -> tuple:
        """Create 6 landmark points achieving the given EAR."""
        width = 40.0
        height = ear_value * width  # EAR ≈ height/width for symmetric eye

        # p1 (left), p4 (right), p2,p3 (upper), p5,p6 (lower)
        p1 = np.array([0.0, 0.0])
        p4 = np.array([width, 0.0])
        p2 = np.array([width * 0.3, -height / 2])
        p3 = np.array([width * 0.7, -height / 2])
        p5 = np.array([width * 0.7,  height / 2])
        p6 = np.array([width * 0.3,  height / 2])
        pts = np.array([p1, p2, p3, p4, p5, p6])

        lm = np.zeros((500, 2), dtype=np.float32)
        for i, pt in enumerate(pts):
            lm[i] = pt

        indices = [0, 1, 2, 3, 4, 5]
        return lm, indices

    def test_open_eye(self):
        lm, idx = self._make_eye(0.35)
        ear = compute_ear(lm, idx)
        self.assertAlmostEqual(ear, 0.35, places=2)

    def test_closed_eye(self):
        lm, idx = self._make_eye(0.10)
        ear = compute_ear(lm, idx)
        self.assertAlmostEqual(ear, 0.10, places=2)

    def test_below_threshold(self):
        lm, idx = self._make_eye(0.15)
        ear = compute_ear(lm, idx)
        self.assertLess(ear, config.EAR_CLOSED_THRESHOLD)

    def test_above_threshold(self):
        lm, idx = self._make_eye(0.32)
        ear = compute_ear(lm, idx)
        self.assertGreater(ear, config.EAR_DROWSY_THRESHOLD)

    def test_fallback_on_insufficient_indices(self):
        lm = np.zeros((10, 2))
        ear = compute_ear(lm, [0, 1, 2])  # Only 3, should fallback
        self.assertEqual(ear, 0.30)


class TestMAR(unittest.TestCase):
    """Tests for Mouth Aspect Ratio (yawn detection)."""

    def _make_mouth_landmarks(self, mar_value: float) -> np.ndarray:
        """Create fake landmarks with specified MAR."""
        lm = np.zeros((500, 2), dtype=np.float32)
        mouth_width = 60.0
        mouth_height = mar_value * mouth_width

        # mouth_left, mouth_right, mouth_top, mouth_bottom
        lm[config.MOUTH_LEFT]   = [200.0, 300.0]
        lm[config.MOUTH_RIGHT]  = [200.0 + mouth_width, 300.0]
        lm[config.MOUTH_TOP]    = [200.0 + mouth_width / 2, 300.0 - mouth_height / 2]
        lm[config.MOUTH_BOTTOM] = [200.0 + mouth_width / 2, 300.0 + mouth_height / 2]
        return lm

    def test_closed_mouth(self):
        lm = self._make_mouth_landmarks(0.1)
        mar = compute_mar(lm)
        self.assertLess(mar, config.MAR_YAWN_THRESHOLD)

    def test_yawning_mouth(self):
        lm = self._make_mouth_landmarks(0.75)
        mar = compute_mar(lm)
        self.assertGreater(mar, config.MAR_YAWN_THRESHOLD)

    def test_mar_range(self):
        for expected in [0.1, 0.3, 0.5, 0.7]:
            lm = self._make_mouth_landmarks(expected)
            mar = compute_mar(lm)
            self.assertAlmostEqual(mar, expected, places=1)


class TestFPSCounter(unittest.TestCase):
    def test_fps_approx(self):
        counter = FPSCounter(window=10)
        interval = 1.0 / 30.0  # 30 FPS
        for _ in range(10):
            counter.tick()
            time.sleep(interval)
        fps = counter.fps
        self.assertGreater(fps, 25)
        self.assertLess(fps, 35)

    def test_empty_counter(self):
        counter = FPSCounter()
        self.assertEqual(counter.fps, 0.0)


class TestHeadPoseEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = HeadPoseEstimator(1280, 720)

    def test_camera_matrix_shape(self):
        cm = self.estimator.camera_matrix
        self.assertEqual(cm.shape, (3, 3))
        # Focal length should be frame width
        self.assertEqual(cm[0, 0], 1280.0)
        self.assertEqual(cm[1, 1], 1280.0)

    def test_centre_points(self):
        cm = self.estimator.camera_matrix
        self.assertEqual(cm[0, 2], 640.0)
        self.assertEqual(cm[1, 2], 360.0)

    def test_assess_forward(self):
        result = self.estimator.assess_pose(0.0, 0.0, 0.0)
        self.assertEqual(result["direction"], "FORWARD")
        self.assertFalse(result["is_looking_away"])

    def test_assess_looking_left(self):
        result = self.estimator.assess_pose(0.0, -30.0, 0.0)
        self.assertTrue(result["is_looking_away"])
        self.assertEqual(result["direction"], "LEFT")

    def test_assess_looking_right(self):
        result = self.estimator.assess_pose(0.0, 30.0, 0.0)
        self.assertTrue(result["is_looking_away"])
        self.assertEqual(result["direction"], "RIGHT")

    def test_assess_nodding_down(self):
        result = self.estimator.assess_pose(-20.0, 0.0, 0.0)
        self.assertTrue(result["is_nodding_down"])
        self.assertEqual(result["direction"], "DOWN")

    def test_head_pose_score_forward(self):
        score = self.estimator.compute_head_pose_score(0.0, 0.0)
        self.assertAlmostEqual(score, 0.0, places=1)

    def test_head_pose_score_extreme(self):
        score = self.estimator.compute_head_pose_score(-30.0, 45.0)
        self.assertGreater(score, 0.5)


class TestGazeTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = GazeTracker()

    def test_center_gaze(self):
        direction = self.tracker._classify_direction(0.5, 0.5)
        self.assertEqual(direction, "CENTER")

    def test_left_gaze(self):
        direction = self.tracker._classify_direction(0.1, 0.5)
        self.assertEqual(direction, "LEFT")

    def test_right_gaze(self):
        direction = self.tracker._classify_direction(0.9, 0.5)
        self.assertEqual(direction, "RIGHT")

    def test_up_gaze(self):
        direction = self.tracker._classify_direction(0.5, 0.1)
        self.assertEqual(direction, "UP")

    def test_down_gaze(self):
        direction = self.tracker._classify_direction(0.5, 0.9)
        self.assertEqual(direction, "DOWN")

    def test_score_center(self):
        score = self.tracker._compute_score(0.5, 0.5)
        self.assertAlmostEqual(score, 0.0, places=2)

    def test_score_corner(self):
        score = self.tracker._compute_score(0.0, 0.0)
        self.assertAlmostEqual(score, 1.0, places=1)


class TestYawnTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = YawnTracker()

    def test_no_yawn(self):
        for _ in range(100):
            self.tracker.update(0.2, config.MAR_YAWN_THRESHOLD)
        self.assertEqual(self.tracker.yawn_count, 0)

    def test_yawn_detection(self):
        # Mouth opens above threshold for sufficient frames, then closes
        for _ in range(config.MAR_YAWN_FRAMES + 5):
            self.tracker.update(0.75, config.MAR_YAWN_THRESHOLD)
        # Close mouth to complete yawn
        self.tracker.update(0.1, config.MAR_YAWN_THRESHOLD)
        self.assertEqual(self.tracker.yawn_count, 1)

    def test_incomplete_yawn_not_counted(self):
        # Opens briefly (less than required frames)
        for _ in range(5):
            self.tracker.update(0.75, config.MAR_YAWN_THRESHOLD)
        self.tracker.update(0.1, config.MAR_YAWN_THRESHOLD)
        self.assertEqual(self.tracker.yawn_count, 0)

    def test_is_yawning_property(self):
        for _ in range(config.MAR_YAWN_FRAMES):
            self.tracker.update(0.75, config.MAR_YAWN_THRESHOLD)
        self.assertTrue(self.tracker.is_yawning)


class TestEyeStateTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = EyeStateTracker()

    def test_open_eyes_no_alert(self):
        for _ in range(50):
            self.tracker.update(0.35, config.EAR_CLOSED_THRESHOLD,
                                 config.EAR_DROWSY_THRESHOLD)
        self.assertFalse(self.tracker.prolonged_closure_alert)

    def test_closed_eyes_alert(self):
        for _ in range(config.EAR_CLOSED_FRAMES + 5):
            self.tracker.update(0.15, config.EAR_CLOSED_THRESHOLD,
                                 config.EAR_DROWSY_THRESHOLD)
        self.assertTrue(self.tracker.prolonged_closure_alert)

    def test_perclos_computation(self):
        # 50% of frames closed
        for i in range(100):
            ear = 0.15 if i % 2 == 0 else 0.35
            self.tracker.update(ear, config.EAR_CLOSED_THRESHOLD,
                                 config.EAR_DROWSY_THRESHOLD)
        self.assertAlmostEqual(self.tracker.perclos, 0.5, delta=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO TESTS (Integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestDrowsinessScenario(unittest.TestCase):
    """
    Simulates a driver whose eyes slowly close over time.
    Expected: fatigue score rises, DROWSY alert triggered.
    """

    def test_gradual_drowsiness(self):
        engine = FatigueEngine()

        # Normal driving (30 frames)
        for _ in range(30):
            score, alert = engine.update(
                ear_left=0.35, ear_right=0.34,
                mar=0.15, pitch=0.0, yaw=0.0,
                gaze_x=0.5, gaze_y=0.5,
                head_pose_score=0.0, gaze_score=0.0,
                face_detected=True, head_away=False
            )
        self.assertEqual(alert, "SAFE")
        normal_score = score

        # Eyes starting to close (30 more frames)
        for _ in range(30):
            engine.update(
                ear_left=0.21, ear_right=0.20,
                mar=0.15, pitch=0.0, yaw=0.0,
                gaze_x=0.5, gaze_y=0.5,
                head_pose_score=0.0, gaze_score=0.0,
                face_detected=True, head_away=False
            )

        # Eyes closed (30 more frames)
        for _ in range(30):
            score, alert = engine.update(
                ear_left=0.14, ear_right=0.13,
                mar=0.15, pitch=0.0, yaw=0.0,
                gaze_x=0.5, gaze_y=0.5,
                head_pose_score=0.0, gaze_score=0.0,
                face_detected=True, head_away=False
            )

        self.assertGreater(score, normal_score)
        self.assertIn(alert, ["DROWSY", "CRITICAL"])
        print(f"\n[PASS] Drowsiness scenario: score={score:.3f}, alert={alert}")


class TestYawnScenario(unittest.TestCase):
    """
    Simulates repeated yawning events.
    Expected: YAWNING alert triggered after threshold yawns.
    """

    def test_yawn_detection(self):
        engine = FatigueEngine()

        # Simulate 3 yawns
        for yawn_idx in range(3):
            # Yawn onset
            for _ in range(config.MAR_YAWN_FRAMES + 10):
                engine.update(
                    ear_left=0.30, ear_right=0.30,
                    mar=0.80, pitch=0.0, yaw=0.0,
                    gaze_x=0.5, gaze_y=0.5,
                    head_pose_score=0.0, gaze_score=0.0,
                    face_detected=True, head_away=False
                )
            # Yawn completion (mouth closes)
            for _ in range(10):
                engine.update(
                    ear_left=0.30, ear_right=0.30,
                    mar=0.10, pitch=0.0, yaw=0.0,
                    gaze_x=0.5, gaze_y=0.5,
                    head_pose_score=0.0, gaze_score=0.0,
                    face_detected=True, head_away=False
                )

        yawn_count = engine.yawn_tracker.yawn_count
        self.assertGreaterEqual(yawn_count, 2)
        print(f"\n[PASS] Yawn scenario: {yawn_count} yawns detected")


class TestDistractionScenario(unittest.TestCase):
    """
    Simulates driver looking away from road for > threshold time.
    Expected: DISTRACTED alert triggered.
    """

    def test_head_turn_distraction(self):
        engine = FatigueEngine()

        # Looking away (head turned right) for extended time
        for _ in range(config.HEAD_POSE_ALERT_FRAMES + 20):
            score, alert = engine.update(
                ear_left=0.32, ear_right=0.32,
                mar=0.10, pitch=0.0, yaw=35.0,  # turned right beyond threshold
                gaze_x=0.85, gaze_y=0.5,         # gaze also off-road
                head_pose_score=0.8, gaze_score=0.7,
                face_detected=True, head_away=True
            )

        self.assertIn(alert, ["DISTRACTED", "DROWSY", "CRITICAL"])
        print(f"\n[PASS] Distraction scenario: score={score:.3f}, alert={alert}")

    def test_no_face_alert(self):
        engine = FatigueEngine()

        # No face detected for > threshold time
        from fatigue_detection import DistractionTracker
        tracker = DistractionTracker()
        for _ in range(100):
            tracker.update(face_detected=False, head_away=False, gaze_away=False)
            time.sleep(0.05)  # ~5 seconds

        self.assertTrue(tracker.no_face_alert)
        print(f"\n[PASS] No-face scenario: alert after {tracker.no_face_duration:.1f}s")


class TestFatigueFormula(unittest.TestCase):
    """Validates the fatigue score formula and weight normalization."""

    def test_weights_sum_to_one(self):
        total = (config.FATIGUE_WEIGHT_EAR + config.FATIGUE_WEIGHT_MAR +
                 config.FATIGUE_WEIGHT_HEAD + config.FATIGUE_WEIGHT_GAZE)
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_zero_input_low_score(self):
        engine = FatigueEngine()
        for _ in range(5):
            score, alert = engine.update(
                ear_left=0.35, ear_right=0.35,
                mar=0.15, pitch=0.0, yaw=0.0,
                gaze_x=0.5, gaze_y=0.5,
                head_pose_score=0.0, gaze_score=0.0,
                face_detected=True, head_away=False
            )
        self.assertLess(score, config.FATIGUE_SCORE_DROWSY)
        self.assertEqual(alert, "SAFE")

    def test_max_input_high_score(self):
        engine = FatigueEngine()
        # Force all sub-scores to maximum
        for _ in range(50):
            score, alert = engine.update(
                ear_left=0.10, ear_right=0.10,
                mar=0.90, pitch=-20.0, yaw=40.0,
                gaze_x=0.0, gaze_y=0.0,
                head_pose_score=1.0, gaze_score=1.0,
                face_detected=True, head_away=True
            )
        self.assertGreater(score, config.FATIGUE_SCORE_DROWSY)

    def test_score_range(self):
        engine = FatigueEngine()
        for _ in range(10):
            score, _ = engine.update(
                ear_left=0.25, ear_right=0.25,
                mar=0.3, pitch=-5.0, yaw=10.0,
                gaze_x=0.6, gaze_y=0.5,
                head_pose_score=0.2, gaze_score=0.15,
                face_detected=True, head_away=False
            )
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestFrameQuality(unittest.TestCase):
    """Tests for environmental robustness helpers."""

    def test_bright_frame(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        quality = assess_frame_quality(frame)
        self.assertGreater(quality["brightness"], 100)
        self.assertFalse(quality["is_blurry"])

    def test_dark_frame(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 20
        quality = assess_frame_quality(frame)
        self.assertLess(quality["brightness"], config.BRIGHTNESS_LOW_THRESHOLD)

    def test_blurry_frame(self):
        # Uniform frame = zero Laplacian variance = blurry
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        quality = assess_frame_quality(frame)
        self.assertTrue(quality["is_blurry"])

    def test_sharp_frame(self):
        # Checkerboard = high Laplacian variance = sharp
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[::2, ::2] = 255
        quality = assess_frame_quality(frame)
        self.assertFalse(quality["is_blurry"])


class TestConfigValidation(unittest.TestCase):
    """Validates config parameter relationships."""

    def test_ear_threshold_ordering(self):
        self.assertLess(config.EAR_CLOSED_THRESHOLD, config.EAR_DROWSY_THRESHOLD)

    def test_fatigue_score_ordering(self):
        self.assertLess(config.FATIGUE_SCORE_DROWSY, config.FATIGUE_SCORE_CRITICAL)
        self.assertLessEqual(config.FATIGUE_SCORE_CRITICAL, 1.0)

    def test_alert_levels_completeness(self):
        required = ["SAFE", "DROWSY", "YAWNING", "DISTRACTED", "CRITICAL", "NO_FACE"]
        for level in required:
            self.assertIn(level, config.ALERT_LEVELS)

    def test_iris_indices_count(self):
        self.assertEqual(len(config.LEFT_IRIS_INDICES), 4)
        self.assertEqual(len(config.RIGHT_IRIS_INDICES), 4)

    def test_ear_indices_count(self):
        self.assertEqual(len(config.LEFT_EYE_EAR_INDICES), 6)
        self.assertEqual(len(config.RIGHT_EYE_EAR_INDICES), 6)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  NEURO-DRIVE Test Suite")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestEAR,
        TestMAR,
        TestFPSCounter,
        TestHeadPoseEstimator,
        TestGazeTracker,
        TestYawnTracker,
        TestEyeStateTracker,
        TestDrowsinessScenario,
        TestYawnScenario,
        TestDistractionScenario,
        TestFatigueFormula,
        TestFrameQuality,
        TestConfigValidation,
    ]

    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    total  = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    print(f"  Results: {passed}/{total} tests passed")
    print("=" * 60)

    sys.exit(0 if result.wasSuccessful() else 1)
