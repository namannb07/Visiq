# Dual-Camera Real-Time Person Analytics & Line Tracking

An optimized, multi-camera real-time computer vision system built with **YOLO26**, **ByteTrack**, and **OpenVINO CPU Acceleration**. 

Designed for standard CPU execution, this project processes two video input sources concurrently with dedicated analytics pipelines and displays a combined side-by-side dashboard monitor.

---

## 🌟 Key Features

* 🎥 **Dual Video Input Sources**: Supports simultaneous capture from multiple USB webcams, RTSP IP camera streams, or video files.
* 🚶 **Camera 1 — Line Crossing Tracking Pipeline**:
  * Powered by **ByteTrack** for persistent object ID tracking across frames.
  * Configurable vertical boundary line (default $X = 0.5 \times \text{width}$).
  * **Left-to-Right Crossing**: Increments **Count IN (+1)** & **Net Present (+1)**.
  * **Right-to-Left Crossing**: Increments **Count OUT (+1)** & Decrements **Net Present (-1)**.
  * Visual track history trails, bounding boxes, and direction indicators (`IN ->` / `<- OUT`).
* 📊 **Camera 2 — Frame Occupancy Pipeline**:
  * Real-time count of total persons currently visible within Camera 2's frame viewport.
* ⚡ **CPU Performance Optimization**:
  * **OpenVINO Engine Export**: Automatically exports PyTorch YOLO weights into Intel OpenVINO IR format on CPU for **2x to 4x higher FPS** using SIMD vector instructions (AVX-512 / AVX2).
  * **Threaded Bufferless Streaming**: Daemon threads fetch frames with `CAP_PROP_BUFFERSIZE=1` to eliminate camera I/O blocking and buffer latency.
  * **Dynamic Inference Scaling**: Default `imgsz=320` maintains high detection accuracy for people while drastically reducing matrix computation overhead on CPU.
* 📺 **Unified Dashboard Canvas**: Displays Camera 1 and Camera 2 side-by-side in a single window with a global top status bar showing live FPS, latency (ms), engine type, and active metrics.

---

## 🏗️ Architecture Overview

```
                          ┌────────────────────────┐
                          │   Camera Stream 1      │───┐ (Left Frame)
                          │  (Threaded Capture)    │   │
                          └────────────────────────┘   │
                                                       ▼
┌──────────────────┐      ┌────────────────────────┐  ┌──────────────────────────────────┐
│ YOLO26 Model     │ ───> │ OpenVINO CPU Engine    │─>│ Cam 1: ByteTrack Line Counter    │──┐
│ (yolo26n.pt)     │      │ (SIMD Vector Acceler.) │  └──────────────────────────────────┘  │
└──────────────────┘      └────────────────────────┘  ┌──────────────────────────────────┐  │   ┌────────────────────┐
                          │   Camera Stream 2      │─>│ Cam 2: Person Occupancy Counter  │──┼──>│ Side-by-Side Canvas│
                          │  (Threaded Capture)    │  └──────────────────────────────────┘  │   │     Dashboard      │
                          └────────────────────────┘                                        │   └────────────────────┘
                                                       ▲                                    │
                                                       │ (Right Frame)                      │
                                                       └────────────────────────────────────┘
```

---

## 📋 System Requirements

* **OS**: Windows 10/11, Linux, or macOS
* **Python**: `3.12+`
* **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Environment

Using **uv**:
```bash
# Clone repository
git clone https://github.com/namannb07/Visiq.git
cd Person_Detection

# Install dependencies (OpenCV, Ultralytics, OpenVINO, python-dotenv)
uv sync
```

Using **pip**:
```bash
pip install -r pyproject.toml
```

---

## 💻 Usage

### 1. List Available Cameras
Find connected camera indices on your system:
```bash
uv run python main.py --list-cameras
```
*Tip: Built-in laptop webcams are usually index `0`. External USB webcams are usually index `1` or `2`.*

---

### 2. Run Dual-Camera Pipeline
Specify your two camera indices (e.g., Camera 0 and Camera 1):
```bash
uv run python main.py --source1 0 --source2 1
```

---

### 3. CLI Command Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--source1` | Camera 1 source index / video path / RTSP URL (Line Tracker) | `0` |
| `--source2` | Camera 2 source index / video path / RTSP URL (Occupancy) | `1` |
| `--imgsz` | Inference image size (`320`, `416`, `640`) | `320` |
| `--conf` | Confidence detection threshold (`0.10` to `1.0`) | `0.40` |
| `--line-x` | Vertical line ratio across frame width (`0.0` to `1.0`) | `0.5` |
| `--no-openvino` | Disable OpenVINO CPU export and run PyTorch fallback | `False` |
| `--list-cameras`| List available hardware webcams and exit | — |

---

### 4. Custom Configuration Examples

**Adjust Line Position (e.g. 40% from left edge)**:
```bash
uv run python main.py --source1 0 --source2 1 --line-x 0.4
```

**Run with Higher Inference Precision (`416x416`)**:
```bash
uv run python main.py --source1 0 --source2 1 --imgsz 416
```

**Run using Video Files or RTSP Streams**:
```bash
uv run python main.py --source1 entrance.mp4 --source2 rtsp://192.168.1.100:554/stream1
```

---

## ⚙️ Environment Configuration (`.env`)

You can also create or edit a `.env` file in the project root to set defaults:

```env
MODEL_NAME=yolo26n.pt
CAMERA1_SOURCE=0
CAMERA2_SOURCE=1
IMAGE_SIZE=320
CONFIDENCE=0.40
LINE_POSITION=0.5
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30
```

---

## 📁 Project Structure

```
Person_Detection/
├── main.py                             # CLI entry point & dual camera pipeline loop
├── pyproject.toml                      # Dependencies & project metadata
├── yolo26n.pt                          # YOLO model weights
├── .env                                # Environment variables (optional)
└── src/
    └── person_detection/
        ├── __init__.py
        ├── camera.py                   # ThreadedCameraStream background reader
        └── line_counter.py             # LineCrossingTracker & overlay renderer
```

---

## ⌨️ Controls

* Press **`Q`** while focusing on the dashboard window to safely close camera streams and exit the application.

---

## 📜 License

This project is open-source under the MIT License.
