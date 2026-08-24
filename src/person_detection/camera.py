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

        # Prime first frame
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
            time.sleep(0.001)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
