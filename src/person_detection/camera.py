"""
Threaded Camera Stream Module
-----------------------------
Runs camera acquisition on a background thread so the main execution loop
always reads the latest captured frame without I/O blocking or frame buffering latency.
"""

import platform
import threading
import time
import cv2


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


class ThreadedCameraStream:
    def __init__(self, source, width: int = 640, height: int = 480, fps: int = 30):
        self.source = parse_source(source)
        backend = get_backend(self.source)
        self.cap = cv2.VideoCapture(self.source, backend)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        if isinstance(self.source, int):
            if width and height:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps:
                self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.stopped = False
        self.last_frame_time = time.monotonic()
        self._STALL_TIMEOUT = 5.0  # seconds before declaring camera stalled

        # Prime first frame
        self.ret, self.frame = self.cap.read()
        if self.ret:
            self.last_frame_time = time.monotonic()
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        consecutive_failures = 0
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    # Attempt to reopen the camera once before giving up
                    print(f"[Camera {self.source}] Too many read failures, attempting reopen...")
                    self.cap.release()
                    time.sleep(0.5)
                    backend = get_backend(self.source)
                    self.cap = cv2.VideoCapture(self.source, backend)
                    if not self.cap.isOpened():
                        print(f"[Camera {self.source}] Reopen failed. Stopping.")
                        self.stopped = True
                        break
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    consecutive_failures = 0
                    continue
                time.sleep(0.005)
                continue
            consecutive_failures = 0
            with self.lock:
                self.ret, self.frame = ret, frame
                self.last_frame_time = time.monotonic()
            time.sleep(0.001)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    @property
    def is_healthy(self) -> bool:
        """Returns False if no new frame has arrived within the stall timeout."""
        with self.lock:
            return (time.monotonic() - self.last_frame_time) < self._STALL_TIMEOUT

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
