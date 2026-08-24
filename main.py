"""
YOLO Dual-Camera Real-Time Analytics Pipeline (OpenVINO CPU Optimized)
----------------------------------------------------------------------
Features:
  1. Camera 1: Object Tracking (ByteTrack) + Vertical Line Crossing Analytics
     - Left-to-Right crossing: Count IN (+1 Net Present)
     - Right-to-Left crossing: Count OUT (-1 Net Present)
  2. Camera 2: Real-time Frame Occupancy / Person Presence Count
  3. OpenVINO Engine Export & SIMD CPU Acceleration for High FPS
  4. Multi-threaded Camera Streams to prevent frame buffer I/O blocking
  5. Side-by-Side Unified Dashboard Visualization Canvas
"""

import argparse
import os
import pathlib
import platform
import sys
import time

import cv2
import torch
from dotenv import load_dotenv
from ultralytics import YOLO

from src.person_detection.camera import ThreadedCameraStream, get_backend, parse_source
from src.person_detection.line_counter import LineCrossingTracker

load_dotenv()

# ============================================================
# Environment / Default Configurations
# ============================================================
MODEL_NAME = os.getenv("MODEL_NAME", "yolo26n.pt")

RAW_CAM1 = (
    os.getenv("CAMERA1_SOURCE")
    or os.getenv("CAMERA_PATH")
    or os.getenv("CAMERA_SOURCE")
    or os.getenv("CAMERA_INDEX")
    or "0"
)

RAW_CAM2 = (
    os.getenv("CAMERA2_SOURCE")
    or "1"
)

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "320"))
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.40"))
LINE_POSITION = float(os.getenv("LINE_POSITION", "0.5"))

CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))

PERSON_CLASS = 0  # COCO "person"
WINDOW_NAME = "Dual Camera Monitor - Tracking (Cam 1) & Occupancy (Cam 2)"
FPS_SMOOTHING = 0.10


def list_cameras(max_index: int = 6) -> None:
    """Utility function to list available local hardware cameras."""
    print("\nScanning available camera indices 0.." + str(max_index - 1))
    for i in range(max_index):
        backend = get_backend(i)
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            status = "OK, frame read" if ok else "Opened, no frame"
            print(f"  [{i}] Available - {w}x{h} ({status})")
            cap.release()
        else:
            print(f"  [{i}] Not available")
    print("\nSet CAMERA1_SOURCE and CAMERA2_SOURCE in your .env or CLI arguments.")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_or_export_openvino_model(model_name: str, imgsz: int, device: str, force_torch: bool = False):
    """
    Loads model. If CPU device and PyTorch weights specified, exports to OpenVINO
    to leverage SIMD CPU optimization for 2x-4x speedup.
    """
    if device == "cuda" or force_torch:
        print(f"Loading PyTorch model on {device.upper()}...")
        model = YOLO(model_name)
        model.to(device)
        model.fuse()
        if device == "cuda":
            model.half()
        return model, f"PyTorch ({device.upper()})"

    # OpenVINO CPU Acceleration
    stem = pathlib.Path(model_name).stem
    openvino_dir = pathlib.Path(f"{stem}_openvino_model")

    if not openvino_dir.exists():
        print(f"Exporting {model_name} to OpenVINO format for CPU acceleration (imgsz={imgsz})...")
        base_model = YOLO(model_name)
        base_model.export(format="openvino", dynamic=True, imgsz=imgsz)
        print("OpenVINO export completed successfully.")

    print(f"Loading OpenVINO CPU model from: {openvino_dir}")
    model = YOLO(str(openvino_dir), task="detect")
    return model, "OpenVINO (CPU)"


def main():
    parser = argparse.ArgumentParser(description="Dual-Camera YOLO Analytics & Tracking Pipeline")
    parser.add_argument("--list-cameras", action="store_true", help="List connected camera indices")
    parser.add_argument("--source1", type=str, default=None, help="Camera 1 index or stream URL (Line Tracking)")
    parser.add_argument("--source2", type=str, default=None, help="Camera 2 index or stream URL (Occupancy Count)")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE, help="Inference resolution size (default: 320)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE, help="Detection confidence threshold")
    parser.add_argument("--line-x", type=float, default=LINE_POSITION, help="Vertical line X ratio (0.0 to 1.0)")
    parser.add_argument("--no-openvino", action="store_true", help="Disable OpenVINO CPU export and use PyTorch")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    cam1_src = parse_source(args.source1 if args.source1 is not None else RAW_CAM1)
    cam2_src = parse_source(args.source2 if args.source2 is not None else RAW_CAM2)

    device = get_device()

    print("=" * 65)
    print("Dual Camera Analytics Pipeline")
    print("=" * 65)
    print(f"Camera 1 (Line Tracker) : {cam1_src}")
    print(f"Camera 2 (Occupancy)    : {cam2_src}")
    print(f"Inference Resolution    : {args.imgsz}x{args.imgsz}")
    print(f"Confidence Threshold    : {args.conf}")

    # Load Model
    t0 = time.perf_counter()
    model, engine_info = load_or_export_openvino_model(
        MODEL_NAME, args.imgsz, device, force_torch=args.no_openvino
    )
    print(f"Engine Ready: {engine_info} (Loaded in {time.perf_counter() - t0:.2f}s)")

    # Open Camera Streams
    print("\nInitializing camera streams...")
    try:
        stream1 = ThreadedCameraStream(cam1_src, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS).start()
    except Exception as e:
        print(f"ERROR starting Camera 1 ({cam1_src}): {e}")
        print("Run 'python main.py --list-cameras' to inspect hardware camera indices.")
        return

    try:
        stream2 = ThreadedCameraStream(cam2_src, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS).start()
    except Exception as e:
        print(f"ERROR starting Camera 2 ({cam2_src}): {e}")
        print("Stopping Camera 1 stream.")
        stream1.stop()
        return

    print(f"Cam 1 Stream: {stream1.actual_width}x{stream1.actual_height} @ {stream1.actual_fps:.1f} FPS")
    print(f"Cam 2 Stream: {stream2.actual_width}x{stream2.actual_height} @ {stream2.actual_fps:.1f} FPS")

    # Warm-up model
    ret1, f1 = stream1.read()
    ret2, f2 = stream2.read()
    if ret1 and f1 is not None:
        print("Warming up inference engine...")
        model.predict(f1, imgsz=args.imgsz, conf=args.conf, classes=[PERSON_CLASS], verbose=False)
        print("Warm-up complete.")

    line_tracker = LineCrossingTracker(line_ratio=args.line_x)

    print("\nAnalytics active. Press Q to exit visual dashboard.\n")

    previous_time = time.perf_counter()
    fps = 0.0
    frame_count = 0

    try:
        while True:
            ret1, frame1 = stream1.read()
            ret2, frame2 = stream2.read()

            if not ret1 or frame1 is None or not ret2 or frame2 is None:
                if stream1.stopped or stream2.stopped:
                    print("Camera stream disconnected or stopped.")
                    break
                time.sleep(0.005)
                continue

            infer_start = time.perf_counter()

            # --- Camera 1: ByteTrack + Line Crossing Analytics ---
            results1 = model.track(
                source=frame1,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=args.imgsz,
                conf=args.conf,
                classes=[PERSON_CLASS],
                verbose=False,
            )

            boxes1 = None
            track_ids1 = None
            if len(results1) > 0 and results1[0].boxes is not None:
                boxes_tensor = results1[0].boxes.xyxy.cpu().numpy()
                if results1[0].boxes.id is not None:
                    ids_tensor = results1[0].boxes.id.int().cpu().numpy()
                    boxes1 = boxes_tensor
                    track_ids1 = ids_tensor

            annotated1 = line_tracker.update_and_draw(frame1, boxes1, track_ids1)

            # Draw Cam 1 Header Banner
            cv2.putText(
                annotated1,
                f"CAM 1: LINE CROSSING [{cam1_src}]",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # --- Camera 2: Person Occupancy Count ---
            results2 = model.predict(
                source=frame2,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=[PERSON_CLASS],
                verbose=False,
            )

            person_count_cam2 = len(results2[0].boxes) if len(results2) > 0 else 0
            annotated2 = results2[0].plot()

            # Overlay Cam 2 Banner & Occupancy Metrics
            h2, w2 = annotated2.shape[:2]
            cv2.putText(
                annotated2,
                f"CAM 2: OCCUPANCY [{cam2_src}]",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            overlay2 = annotated2.copy()
            cv2.rectangle(overlay2, (10, h2 - 70), (240, h2 - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.6, annotated2, 0.4, 0, annotated2)

            cv2.putText(
                annotated2,
                f"PERSONS VISIBLE: {person_count_cam2}",
                (20, h2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            infer_duration = time.perf_counter() - infer_start

            # Calculate FPS
            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                cur_fps = 1.0 / elapsed
                fps = cur_fps if fps == 0 else (FPS_SMOOTHING * cur_fps + (1 - FPS_SMOOTHING) * fps)

            # Resize frame2 to match frame1 height if different
            h1, w1 = annotated1.shape[:2]
            if h2 != h1:
                new_w2 = int(w2 * (h1 / h2))
                annotated2 = cv2.resize(annotated2, (new_w2, h1))

            # Combine Side-by-Side Canvas
            canvas = cv2.hconcat([annotated1, annotated2])

            # Draw Global Top FPS & System Status Bar
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 25), (30, 30, 30), -1)
            status_str = f"FPS: {fps:.1f} | Latency: {infer_duration * 1000:.0f} ms | Engine: {engine_info}"
            cv2.putText(
                canvas,
                status_str,
                (15, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, canvas)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1
            if frame_count % 100 == 0:
                print(
                    f"FPS: {fps:.1f} | Latency: {infer_duration * 1000:.1f}ms | "
                    f"Cam 1 Net: {line_tracker.net_count} | Cam 2 Occupancy: {person_count_cam2}"
                )

    finally:
        print("\nStopping camera streams...")
        stream1.stop()
        stream2.stop()
        cv2.destroyAllWindows()
        print("System shutdown complete.")


if __name__ == "__main__":
    main()