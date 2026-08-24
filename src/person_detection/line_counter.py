"""
Line Crossing Analytics Module
-------------------------------
Tracks object centroids across a vertical boundary line and counts:
- Count IN: Person crosses line from Left to Right (+1 Net Present)
- Count OUT: Person crosses line from Right to Left (-1 Net Present)
"""

from collections import deque
import cv2
import numpy as np


class LineCrossingTracker:
    def __init__(self, line_ratio: float = 0.5, history_len: int = 30):
        self.line_ratio = line_ratio
        self.history_len = history_len

        self.count_in = 0
        self.count_out = 0

        # track_id -> deque of (x, y) centroids
        self.track_history = {}
        # track_id -> side relative to line: 'LEFT' or 'RIGHT'
        self.track_state = {}

    @property
    def net_count(self) -> int:
        return self.count_in - self.count_out

    def update_and_draw(self, frame, boxes, track_ids):
        """
        Updates line crossing state using detected boxes and track IDs,
        and renders tracking annotations onto the frame.

        Parameters:
            frame: OpenCV image frame
            boxes: Bounding boxes [[x1, y1, x2, y2], ...]
            track_ids: List of integer track IDs corresponding to boxes
        """
        h, w = frame.shape[:2]
        line_x = int(w * self.line_ratio)

        # Draw main boundary line
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 255, 255), 2, cv2.LINE_AA)

        # Draw directional indicators
        cv2.putText(
            frame,
            "IN ->",
            (line_x + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "<- OUT",
            (line_x - 85, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        current_active_ids = set()

        if boxes is not None and track_ids is not None:
            for box, track_id in zip(boxes, track_ids):
                track_id = int(track_id)
                current_active_ids.add(track_id)

                x1, y1, x2, y2 = map(int, box[:4])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if track_id not in self.track_history:
                    self.track_history[track_id] = deque(maxlen=self.history_len)

                self.track_history[track_id].append((cx, cy))
                history = self.track_history[track_id]

                # Determine current side
                current_side = "LEFT" if cx < line_x else "RIGHT"

                if track_id in self.track_state:
                    prev_side = self.track_state[track_id]
                    # Left to Right crossing -> IN
                    if prev_side == "LEFT" and current_side == "RIGHT":
                        self.count_in += 1
                        cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)
                    # Right to Left crossing -> OUT
                    elif prev_side == "RIGHT" and current_side == "LEFT":
                        self.count_out += 1
                        cv2.circle(frame, (cx, cy), 12, (0, 0, 255), -1)

                self.track_state[track_id] = current_side

                # Draw track trail line
                points = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(255, 200, 0), thickness=2)

                # Draw box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        # Cleanup stale tracks
        stale_ids = [tid for tid in list(self.track_history.keys()) if tid not in current_active_ids]
        for tid in stale_ids:
            # retain last state for a few frames or clean up
            if len(self.track_history[tid]) > 0:
                self.track_history[tid].popleft()
            if len(self.track_history[tid]) == 0:
                del self.track_history[tid]
                if tid in self.track_state:
                    del self.track_state[tid]

        # Draw summary overlay box on Camera 1 frame
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 110), (220, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(
            frame,
            f"IN  (L->R): {self.count_in}",
            (20, h - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"OUT (R->L): {self.count_out}",
            (20, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 100, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"NET PRESENT: {self.net_count}",
            (20, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return frame
