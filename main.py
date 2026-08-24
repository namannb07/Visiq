"""
YOLO26n Real-Time Person Detection (USB Webcam)
------------------------------------------------
Optimized for external USB webcams on machines that also have a built-in
laptop camera. Key optimizations over a naive capture->predict->show loop:

  1. Threaded capture (CameraStream) — camera I/O runs on its own thread,
     so the main loop always grabs the LATEST frame instead of blocking
     on cap.read(). This removes capture latency from the inference loop
     and prevents frame queue buildup.
  2. Auto device selection — uses CUDA + FP16 if a GPU is available,
     falls back to CPU cleanly otherwise.
  3. model.fuse() — fuses Conv+BN layers for faster inference.
  4. MJPG FOURCC — most USB webcams only hit their higher FPS/resolution
     modes when MJPG compression is requested; without it many cams cap
     out at ~10-15 FPS at 640x480.
  5. --list-cameras utility — since laptops usually enumerate the
     built-in camera at index 0, this lets you confirm which index your
     USB webcam actually is before running detection.
  6. Zero redundant re-encoding of overlay strings / minimal per-frame
     Python overhead in the hot loop.

Usage:
    python yolo_person_detection_optimized.py --list-cameras
    python yolo_person_detection_optimized.py
"""

import argparse
import os
import platform
import threading
import time

import cv2
import torch
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()


# ============================================================
# Configuration (env-overridable)
# ============================================================

MODEL_NAME = os.getenv("MODEL_NAME", "yolo26n.pt")

# Reads CAMERA_PATH (used in .env), falling back to CAMERA_SOURCE / CAMERA_INDEX
RAW_CAMERA_PATH = (
    os.getenv("CAMERA_PATH")
    or os.getenv("CAMERA_SOURCE")
    or os.getenv("CAMERA_INDEX")
    or "1"
)

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "416"))
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.40"))

CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))

PERSON_CLASS = 0  # COCO "person"
WINDOW_NAME = "YOLO26n Person Detection"

FPS_SMOOTHING = 0.10
LOG_EVERY_N_FRAMES = 100


def parse_source(source_val):
    """
    Parses a source value into either an integer (webcam index) or string (RTSP stream URL / file path).
    """
    if source_val is None:
        return 0
    source_str = str(source_val).strip()
    if source_str.isdigit():
        return int(source_str)
    try:
        return int(source_str)
    except ValueError:
        return source_str


def get_backend(source) -> int:
    """DirectShow on Windows for local webcam indices; default OpenCV backend for streams/files."""
    if isinstance(source, int) and platform.system() == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY



# ============================================================
# Threaded camera reader — always serves the newest frame
# ============================================================

class CameraStream:
    def __init__(self, source, width=None, height=None, fps=None):
        self.source = source
        backend = get_backend(source)
        self.cap = cv2.VideoCapture(source, backend)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        # MJPG and fixed resolution settings apply to local hardware webcams (int index)
        if isinstance(source, int):
            if width and height:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps:
                self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize stale-frame latency

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.stopped = False

        # Prime with one synchronous read so the first frame is ready
        # before the background thread starts.
        self.ret, self.frame = self.cap.read()

        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


# ============================================================
# Utility: enumerate local cameras
# ============================================================

def list_cameras(max_index: int = 6) -> None:
    print("Scanning camera indices 0.." + str(max_index - 1))

    for i in range(max_index):
        backend = get_backend(i)
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            status = "OK, frame read" if ok else "opened, no frame"
            print(f"  [{i}] available - {w}x{h} ({status})")
            cap.release()
        else:
            print(f"  [{i}] not available")

    print()
    print("Tip: the built-in laptop webcam is almost always index 0.")
    print("Your external USB webcam is typically index 1 (or higher if")
    print("you have more than one extra camera/virtual camera installed).")
    print("Set CAMERA_INDEX in your .env to the index you want to use.")


# ============================================================
# Device selection
# ============================================================

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Optimized YOLO person detection")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available local camera indices and exit.",
    )
    parser.add_argument(
        "--source",
        "--camera-path",
        "--camera-index",
        dest="source",
        type=str,
        default=None,
        help="Override CAMERA_PATH from .env for this run (camera index e.g. 0, 1 or stream URL/file path).",
    )
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    raw_source = args.source if args.source is not None else RAW_CAMERA_PATH
    camera_source = parse_source(raw_source)

    device = get_device()
    use_half = device == "cuda"

    print("=" * 60)
    print("YOLO26n Person Detection")
    print("=" * 60)
    print(f"Device: {device}{' (FP16)' if use_half else ''}")
    print(f"Video Source: {camera_source}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    t0 = time.perf_counter()
    model = YOLO(MODEL_NAME)
    model.to(device)
    model.fuse()  # fold Conv+BatchNorm for faster inference
    if use_half:
        model.half()  # set FP16 weights once, instead of passing half= per call
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

    # --------------------------------------------------------
    # Open camera / stream (threaded)
    # --------------------------------------------------------
    print(f"\nOpening video source: {camera_source}...")
    try:
        stream = CameraStream(
            camera_source, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
        ).start()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("Run with --list-cameras to see available camera indices.")
        return

    print(
        f"Stream ready: {stream.actual_width}x{stream.actual_height} "
        f"@ {stream.actual_fps:.1f} FPS"
    )

    ret, frame = stream.read()
    if not ret or frame is None:
        print("ERROR: Could not read an initial frame from the video source.")
        stream.stop()
        return

    # --------------------------------------------------------
    # Warm-up (first inference is always slower — CUDA context /
    # graph build / kernel autotune, so do it before timing loop)
    # --------------------------------------------------------
    print("Warming up model...")
    t0 = time.perf_counter()
    model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        classes=[PERSON_CLASS],
        device=device,
        verbose=False,
    )
    print(f"Warm-up done in {time.perf_counter() - t0:.2f}s")

    print("\nDetection started. Press Q to quit.\n")

    previous_time = time.perf_counter()
    fps = 0.0
    frame_count = 0
    inference_time = 0.0

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                # Camera thread hasn't produced a new frame yet / died.
                if stream.stopped:
                    print("ERROR: Camera stream stopped unexpectedly.")
                    break
                continue

            infer_start = time.perf_counter()
            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                classes=[PERSON_CLASS],
                device=device,
                verbose=False,
            )
            inference_time = time.perf_counter() - infer_start

            person_count = len(results[0].boxes)
            annotated = results[0].plot()

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = current_fps if fps == 0 else (
                    FPS_SMOOTHING * current_fps + (1 - FPS_SMOOTHING) * fps
                )

            source_disp = str(camera_source)
            if len(source_disp) > 25:
                source_disp = source_disp[:22] + "..."

            cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Persons: {person_count}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Source: {source_disp}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Inference: {inference_time * 1000:.0f} ms ({device})",
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1
            if frame_count % LOG_EVERY_N_FRAMES == 0:
                print(
                    f"FPS: {fps:.1f} | Inference: {inference_time * 1000:.1f} ms "
                    f"| Persons: {person_count}"
                )

    finally:
        print("\nStopping...")
        stream.stop()
        cv2.destroyAllWindows()
        print("Program stopped.")


if __name__ == "__main__":
    main()