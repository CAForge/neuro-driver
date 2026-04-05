# 🧠 Neuro-Drive: Human-Centric Driver Monitoring System

> Real-time AI-powered driver fatigue and distraction detection using computer vision, MediaPipe Face Mesh, and a multi-factor composite scoring model.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Detection Algorithms Explained](#detection-algorithms)
   - [Eye Aspect Ratio (EAR)](#ear)
   - [Mouth Aspect Ratio (MAR)](#mar)
   - [Head Pose Estimation](#head-pose)
   - [Gaze Tracking](#gaze)
   - [Composite Fatigue Score](#fatigue-score)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Running the System](#running)
7. [REST API](#api)
8. [Docker Deployment](#docker)
9. [Configuration](#configuration)
10. [Testing](#testing)
11. [Expected Output](#output)
12. [Performance](#performance)
13. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview {#overview}

**Neuro-Drive** monitors a driver in real-time using a standard webcam and detects:

| Condition | Detection Method | Alert |
|-----------|-----------------|-------|
| Eye closure / Drowsiness | EAR + PERCLOS | ⚠ DROWSY |
| Yawning | MAR + event counting | ⚠ YAWNING |
| Head turned away | solvePnP + Euler angles | ⚠ DISTRACTED |
| Eyes off-road | Iris position tracking | ⚠ DISTRACTED |
| Critical fatigue | Composite score | 🚨 CRITICAL |
| No face visible | Detection timeout | ⚠ NO FACE |

### Key Features

- **MediaPipe Face Mesh** — 468 (+ 10 iris) landmarks at real-time speed
- **Multi-factor fatigue model** — Weighted combination of 4 detection signals
- **Per-user calibration** — Auto-adapts EAR/MAR thresholds on startup
- **Environmental robustness** — CLAHE enhancement in low light, adaptive thresholds
- **REST API** — FastAPI server with Server-Sent Events (SSE) for dashboards
- **CSV logging** — Timestamped event log for post-session analysis
- **Docker ready** — Full containerised deployment

---

## 🏗️ Architecture {#architecture}

```
webcam / video file
        │
        ▼
┌──────────────────┐
│  Frame Capture   │  OpenCV VideoCapture
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Quality Assessment│  Brightness, blur detection
│ + Preprocessing  │  CLAHE, adaptive thresholds
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  MediaPipe Face  │  468-point face mesh
│     Mesh         │  + iris landmarks (478 pts)
└────────┬─────────┘
         │
    ┌────┴────────────────────────────┐
    │                                 │
    ▼                                 ▼
┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  EAR    │  │   MAR    │  │  Head    │  │  Gaze    │
│ Module  │  │  Module  │  │  Pose   │  │ Tracker  │
│         │  │          │  │Estimator│  │          │
└────┬────┘  └────┬─────┘  └────┬────┘  └────┬─────┘
     │            │              │              │
     └────────────┴──────────────┴──────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │   Fatigue Engine     │
                    │                      │
                    │  score = w1·EAR      │
                    │        + w2·MAR      │
                    │        + w3·Head     │
                    │        + w4·Gaze     │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼──────────────┐
              ▼                ▼               ▼
        Visual Alert     Audio Alert     CSV + Log
         (OpenCV)       (sounddevice)    (File I/O)
              │
              ▼
         ┌─────────┐
         │ REST API│  FastAPI + SSE
         └─────────┘
```

---

## 🔬 Detection Algorithms Explained {#detection-algorithms}

### 1. Eye Aspect Ratio (EAR) {#ear}

The EAR formula (Soukupová & Čech, 2016) computes a ratio of vertical to horizontal eye distances:

```
        ||p2 - p6|| + ||p3 - p5||
EAR = ─────────────────────────────
              2 · ||p1 - p4||
```

Where:
- `p1, p4` = horizontal eye corners (left, right)
- `p2, p3` = upper eyelid points
- `p5, p6` = lower eyelid points

```
        p2    p3
    p1           p4
        p6    p5
```

| EAR Value | State |
|-----------|-------|
| > 0.27    | Eyes open (alert) |
| 0.22–0.27 | Eyes partially closed (drowsy) |
| < 0.22    | Eyes closed |

**PERCLOS** (Percentage of Eye Closure): tracks the fraction of frames within a rolling 300-frame window where eyes are closed. This is a clinical standard for drowsiness measurement.

**Adaptive Thresholding**: EAR thresholds are relaxed by 10% in low-light conditions (frame brightness < 40/255) since landmark detection is less reliable.

**Per-user Calibration**: During the 5-second startup calibration, the driver's resting EAR is recorded. The closure threshold is then set to 75% of that baseline, making it personalised.

---

### 2. Mouth Aspect Ratio (MAR) {#mar}

Similar to EAR, the MAR quantifies how open the mouth is:

```
          ||top - bottom||
MAR = ─────────────────────────
         ||left - right||
```

| MAR Value | State |
|-----------|-------|
| < 0.3     | Mouth closed |
| > 0.65    | Yawning |

**Yawn Event Logic**:
1. MAR must exceed threshold for ≥ 20 consecutive frames (onset)
2. MAR must then fall below threshold (completion)
3. Yawn count is maintained over a 2-minute sliding window
4. ≥ 2 yawns in the window → YAWNING alert

---

### 3. Head Pose Estimation {#head-pose}

Uses OpenCV's `solvePnP` (Iterative) to solve the Perspective-n-Point problem:

**Input**: 6 known 3D face model points mapped to their 2D image projections
```
3D Model Points (mm):         Corresponding Landmarks:
  Nose tip    [0, 0, 0]       → Landmark 1
  Chin        [0, -330, -65]  → Landmark 152
  Left eye    [-225, 170,-135]→ Landmark 33
  Right eye   [225, 170,-135] → Landmark 263
  Left mouth  [-150,-150,-125]→ Landmark 61
  Right mouth [150, -150,-125]→ Landmark 291
```

**Camera Intrinsics**: Approximated from image size (focal length ≈ frame width). This is a standard approximation for webcams without calibration.

**Output**: Rotation + translation vectors → converted to Euler angles via `RQDecomp3x3`:
- **Pitch**: Head tilt up/down (nodding). Below −15° → falling asleep
- **Yaw**: Head turn left/right. Beyond ±25° → looking away from road
- **Roll**: Head tilt sideways

---

### 4. Gaze Tracking {#gaze}

Uses MediaPipe's `refine_landmarks=True` which extends the 468-point mesh to 478 points, adding 4 iris landmarks per eye (indices 469–477).

**Algorithm**:
1. Find the eye bounding box from contour landmarks
2. Find iris centre = mean of the 4 iris landmark positions
3. Compute normalised iris position within the bounding box:
   - `(0.5, 0.5)` = looking straight ahead
   - `(0.1, 0.5)` = looking far left
   - `(0.9, 0.5)` = looking far right

**Calibration**: During the startup phase, the resting forward-facing iris position is recorded and subtracted as an offset, compensating for camera angle or positioning asymmetries.

---

### 5. Composite Fatigue Score {#fatigue-score}

All four signals are combined into a single normalised score:

```
fatigue_score = w₁·EAR_score + w₂·MAR_score + w₃·head_score + w₄·gaze_score
```

**Default weights** (tunable in `config.py`):

| Component | Weight | Rationale |
|-----------|--------|-----------|
| EAR       | 0.40   | Primary drowsiness indicator |
| Head Pose | 0.25   | Strong distraction signal |
| MAR       | 0.20   | Cumulative fatigue indicator |
| Gaze      | 0.15   | Secondary distraction signal |

**Alert Thresholds**:
- `score ≥ 0.45` → ⚠ DROWSY
- `score ≥ 0.70` → 🚨 CRITICAL

**Smoothing**: Exponential moving average (α = 0.3) is applied to avoid flickering alerts from momentary landmark noise.

---

## 📁 Project Structure {#project-structure}

```
neuro-drive/
├── main.py               # Entry point, main processing loop
├── fatigue_detection.py  # FatigueEngine, EyeStateTracker, YawnTracker
├── head_pose.py          # HeadPoseEstimator (solvePnP + Euler angles)
├── gaze_tracking.py      # GazeTracker (iris landmark analysis)
├── utils.py              # EAR/MAR computation, drawing, logging, audio
├── config.py             # All tunable parameters and thresholds
├── api.py                # FastAPI REST server
├── tests.py              # Full test suite (13 test classes, ~50 tests)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container build
├── README.md             # This file
│
├── neuro_drive_events.log  (created at runtime)
└── neuro_drive_data.csv    (created at runtime)
```

---

## 🛠 Installation {#installation}

### Prerequisites

- Python 3.9+ (3.11 recommended)
- Webcam or video file
- Linux / macOS / Windows 10+

### Step 1: Clone / Download

```bash
git clone https://github.com/your-org/neuro-drive.git
cd neuro-drive
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
python -m venv venv
```

### Step 3: Instal

l Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Audio Support (Optional)

**Linux**:
```bash
sudo apt-get install libportaudio2
pip install sounddevice
```

**macOS**:
```bash
brew install portaudio
pip install sounddevice
```

**Windows**: `sounddevice` installs without extras via pip.

### Step 5: Verify Installation

```bash
python -c "import mediapipe; import cv2; print('OK')"
```

---

## 🚀 Running the System {#running}

### Default Webcam

```bash
python main.py
```

### Specific Camera Index

```bash
python main.py --source 1
```

### Video File

```bash
python main.py --video path/to/drive_footage.mp4
```

### Lower Resolution (for slower hardware)

```bash
python main.py --width 640 --height 480
```

### Skip Calibration

```bash
python main.py --no-calibration
```

### Disable REST API

```bash
python main.py --no-api
```

### Keyboard Controls (while running)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Re-run calibration |
| `s` | Print session statistics to console |

---

## 🌐 REST API {#api}

The FastAPI server starts automatically at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

### Endpoints

#### `GET /status` — Current alert status
```json
{
  "alert_status": "DROWSY",
  "fatigue_score": 0.512,
  "fps": 28.4,
  "timestamp": 1711234567.89
}
```

#### `GET /metrics` — All sensor values
```json
{
  "metrics": {
    "ear_left": 0.201,
    "ear_right": 0.198,
    "ear_avg": 0.200,
    "mar": 0.145,
    "yawn_count": 2,
    "pitch": -3.2,
    "yaw": 8.1,
    "roll": 1.4,
    "gaze_x": 0.52,
    "gaze_y": 0.49,
    "brightness": 142.3,
    "is_blurry": false
  },
  "alert_status": "SAFE",
  "fatigue_score": 0.183
}
```

#### `GET /stats` — Session statistics
```json
{
  "stats": {
    "session_duration_s": 3600,
    "total_alerts": 5,
    "total_yawns": 3,
    "perclos": 0.04,
    "fatigue_score": 0.23
  }
}
```

#### `GET /stream` — SSE real-time stream
```javascript
const source = new EventSource('http://localhost:8000/stream');
source.onmessage = (e) => {
    const data = JSON.parse(e.data);
    console.log(data.alert_status, data.fatigue_score);
};
```

---

## 🐳 Docker Deployment {#docker}

### Build

```bash
docker build -t neuro-drive:latest .
```

### Run with Webcam (Linux)

```bash
xhost +local:docker

docker run --rm \
  --device=/dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -p 8000:8000 \
  neuro-drive:latest
```

### Run with Video File

```bash
docker run --rm \
  -v $(pwd)/videos:/app/videos \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -p 8000:8000 \
  neuro-drive:latest python main.py --video /app/videos/test.mp4
```

### Docker Compose

```yaml
version: '3.8'
services:
  neuro-drive:
    build: .
    devices:
      - /dev/video0:/dev/video0
    environment:
      - DISPLAY=${DISPLAY}
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    restart: unless-stopped
```

---

## ⚙️ Configuration {#configuration}

All parameters are in `config.py`. Key tunable values:

### Detection Thresholds

```python
# Eye Aspect Ratio
EAR_CLOSED_THRESHOLD = 0.22   # Below this = eye closed
EAR_DROWSY_THRESHOLD = 0.27   # Below this = eye drowsy
EAR_CLOSED_FRAMES = 15        # Frames with closed eyes before alert

# Mouth (Yawning)
MAR_YAWN_THRESHOLD = 0.65     # Above this = yawning
MAR_YAWN_COUNT_ALERT = 2      # Yawns in 2 min before alert

# Head Pose
HEAD_YAW_THRESHOLD = 25.0     # ° left/right before alert
HEAD_PITCH_DOWN_THRESHOLD = 15.0  # ° down before alert

# Gaze
GAZE_OFF_ROAD_THRESHOLD = 0.35  # Iris deviation ratio
```

### Fatigue Weights

```python
FATIGUE_WEIGHT_EAR  = 0.40
FATIGUE_WEIGHT_MAR  = 0.20
FATIGUE_WEIGHT_HEAD = 0.25
FATIGUE_WEIGHT_GAZE = 0.15
# Must sum to 1.0
```

### Alert Thresholds

```python
FATIGUE_SCORE_DROWSY   = 0.45
FATIGUE_SCORE_CRITICAL = 0.70
ALERT_COOLDOWN_SECONDS = 5.0
```

---

## 🧪 Testing {#testing}

### Run All Tests

```bash
python tests.py
```

### Run with pytest

```bash
pytest tests.py -v
pytest tests.py -v --tb=short
pytest tests.py -v -k "Drowsy"   # Run only drowsiness tests
```

### Test Coverage

```bash
pytest tests.py --cov=. --cov-report=term-missing
```

### Test Scenarios

| Test Class | What it tests |
|------------|--------------|
| `TestEAR` | EAR formula correctness, open/closed detection |
| `TestMAR` | MAR formula, yawn threshold detection |
| `TestFPSCounter` | FPS accuracy at known intervals |
| `TestHeadPoseEstimator` | Camera matrix, angle direction logic |
| `TestGazeTracker` | Direction classification, score computation |
| `TestYawnTracker` | Yawn event detection, incomplete yawn exclusion |
| `TestEyeStateTracker` | PERCLOS calculation, alert frame counting |
| `TestDrowsinessScenario` | Gradual drowsiness → DROWSY alert |
| `TestYawnScenario` | Multiple yawns → YAWNING alert |
| `TestDistractionScenario` | Head turn / no face → DISTRACTED alert |
| `TestFatigueFormula` | Weight normalisation, score range validity |
| `TestFrameQuality` | Brightness / blur detection |
| `TestConfigValidation` | Config parameter integrity |

---

## 📊 Expected Output {#output}

### Display Layout

```
┌──────────────┬────────────────────────────────────┐
│              │                                    │
│ NEURO-DRIVE  │  ✓ SAFE  (or ⚠ DROWSY etc.)       │
│ Driver Monitor│                                   │
│──────────────│                                    │
│ EAR L: 0.312 │                                    │
│ EAR R: 0.308 │    [Live face mesh overlay]        │
│ EAR Avg:0.310│    [Eye contours: green/orange]    │
│ MAR:   0.142 │    [Iris circles]                  │
│ Yawns: 0     │    [Head pose XYZ axes]            │
│              │                                    │
│ Pitch:  -2.1°│                                    │
│ Yaw:     3.4°│    [Gaze grid indicator]           │
│ Roll:    0.8°│                                    │
│              │                                    │
│ Gaze X: 0.51 │                                    │
│ Gaze Y: 0.49 │                                    │
│              │                                    │
│ [Fatigue Bar]│                                    │
│ ████░░░  0.18│                                    │
│              │                                    │
│ Bright: 145  │                                    │
│ Blurry: NO   │                                    │
│              │                                    │
│ FPS: 28.4    │                                    │
└──────────────┴────────────────────────────────────┘
```

### Alert States

| Status | Display | Trigger |
|--------|---------|---------|
| `✓ SAFE` | Green banner | Normal driving |
| `⚠ DROWSY` | Orange blinking | EAR low or fatigue score ≥ 0.45 |
| `⚠ YAWNING` | Orange blinking | Active yawn or ≥ 2 yawns in 2 min |
| `⚠ DISTRACTED` | Red blinking | Head/gaze off-road for ≥ 2.5s |
| `🚨 CRITICAL` | Dark red blinking | Fatigue score ≥ 0.70 |
| `⚠ NO FACE` | Purple blinking | No face detected for ≥ 3s |

---

## ⚡ Performance {#performance}

### Benchmark Results (tested on Intel Core i7-12th Gen)

| Resolution | FPS (no skip) | CPU Usage |
|------------|--------------|-----------|
| 1280×720   | 22–28 FPS    | 40–55%    |
| 640×480    | 30–40 FPS    | 25–35%    |

### Optimisation Tips

1. **Lower resolution**: `--width 640 --height 480`
2. **Frame skipping**: Set `SKIP_FRAMES = 1` in `config.py` (processes every other frame)
3. **Disable mesh drawing**: Set `DRAW_FACE_MESH = False`
4. **OpenVINO** (Intel hardware): Set `USE_OPENVINO = True` in `config.py`
5. **Single core**: `export OMP_NUM_THREADS=2`

---

## 🔧 Troubleshooting {#troubleshooting}

### Camera not found
```bash
# List available cameras
python -c "import cv2; [print(f'Camera {i}') for i in range(5) if cv2.VideoCapture(i).isOpened()]"
```

### MediaPipe import error
```bash
pip install --upgrade mediapipe
```

### No audio alert
- Install `sounddevice`: `pip install sounddevice`
- Linux: `sudo apt-get install libportaudio2`
- Alerts are still visual; audio is optional

### Low FPS
- Reduce resolution: `--width 640 --height 480`
- Set `DRAW_FACE_MESH = False` in `config.py`
- Set `SKIP_FRAMES = 1` in `config.py`

### Docker display issues (Linux)
```bash
xhost +local:docker
docker run ... -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ...
```

### Poor detection in low light
- The system automatically applies CLAHE enhancement in low light
- Use a well-lit environment or add ambient lighting
- Infrared cameras significantly improve performance in darkness

---

## 📄 Logs & Data

**Event Log** (`neuro_drive_events.log`):
```
2024-11-01 14:23:01 | WARNING  | [ALERT] *** DROWSY *** | fatigue=0.521 | session=342s
2024-11-01 14:23:10 | INFO     | [Fatigue] Yawn detected! Total in window: 1
```

**CSV Data** (`neuro_drive_data.csv`):
```csv
timestamp,fps,ear_left,ear_right,ear_avg,mar,pitch,yaw,roll,gaze_x,gaze_y,fatigue_score,alert_status,brightness,is_blurry
2024-11-01 14:23:01,27.3,0.312,0.308,0.310,0.142,-2.1,3.4,0.8,0.51,0.49,0.183,SAFE,145.2,False
```

---

## 📜 License

MIT License — Free to use, modify, and distribute with attribution.

---

## 🙏 References

1. Soukupová, T., & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* CVWW.
2. Lugaresi, C. et al. (2019). *MediaPipe: A Framework for Perceiving and Processing Reality.* Workshop on ML for AR at CVPR.
3. OpenCV Documentation: [solvePnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
4. PERCLOS Standard (SAE J3016, 2021)
#   n e u r o - d r i v e r  
 