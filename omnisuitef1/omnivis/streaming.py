"""MJPEG streaming with per-stream state and detection overlay.

Usage:
    from omnisee.streaming import generate_detection_stream, StreamStateRegistry

    # In FastAPI:
    @app.get("/stream/{source}")
    async def stream(source: str):
        return StreamingResponse(
            generate_detection_stream(source),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
"""

from __future__ import annotations

import logging
import time
from collections import deque
from threading import Lock
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from omnivis._types import Detection, StreamState

logger = logging.getLogger(__name__)

# ── Colors for drawing ────────────────────────────────────────────────────
CLASS_COLORS = {}
DEFAULT_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _get_color(class_name: str) -> Tuple[int, int, int]:
    if class_name not in CLASS_COLORS:
        h = hash(class_name) % 180
        color_bgr = cv2.cvtColor(np.array([[[h, 255, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
        CLASS_COLORS[class_name] = tuple(int(c) for c in color_bgr)
    return CLASS_COLORS[class_name]


# ── Stream State Registry ─────────────────────────────────────────────────

class StreamStateRegistry:
    """Thread-safe registry of per-stream state."""

    def __init__(self):
        self._states: Dict[str, StreamState] = {}
        self._lock = Lock()

    def get_or_create(self, stream_id: str) -> StreamState:
        with self._lock:
            if stream_id not in self._states:
                self._states[stream_id] = StreamState(stream_id=stream_id)
            state = self._states[stream_id]
            state.last_access = time.time()
            return state

    def get(self, stream_id: str) -> Optional[StreamState]:
        with self._lock:
            return self._states.get(stream_id)

    def remove(self, stream_id: str):
        with self._lock:
            self._states.pop(stream_id, None)

    def cleanup_idle(self, max_idle: float = 300):
        now = time.time()
        with self._lock:
            stale = [k for k, v in self._states.items() if now - v.last_access > max_idle]
            for k in stale:
                del self._states[k]

    def active_streams(self) -> List[str]:
        with self._lock:
            return list(self._states.keys())


_registry = StreamStateRegistry()


def get_stream_registry() -> StreamStateRegistry:
    return _registry


# ── Drawing ───────────────────────────────────────────────────────────────

def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    *,
    show_labels: bool = True,
    line_width: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels on a frame."""
    output = frame.copy()

    for d in detections:
        x1, y1, x2, y2 = [int(c) for c in d.bbox]
        color = _get_color(d.class_name)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, line_width)

        if show_labels:
            label = f"{d.class_name} {d.confidence:.2f}"
            if d.track_id is not None:
                label = f"#{d.track_id} {label}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(output, label, (x1, y1 - 4), FONT, 0.5, (255, 255, 255), 1)

    return output


# ── Video Capture ─────────────────────────────────────────────────────────

def open_capture(source: str) -> cv2.VideoCapture:
    """Open a video source (URL, RTSP, file, or webcam index)."""
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


# ── MJPEG Stream Generator ───────────────────────────────────────────────

def generate_detection_stream(
    source: str,
    *,
    model_type: str = "Detection",
    confidence: float = 0.25,
    tracker: Optional[str] = None,
    imgsz: int = 768,
    skip_frames: int = 0,
    jpeg_quality: int = 85,
) -> Generator[bytes, None, None]:
    """Generate an MJPEG stream with detection overlay.

    Args:
        source: Video source (URL, RTSP, file path, or webcam index like "0").
        model_type: YOLO model type for detection.
        confidence: Detection confidence threshold.
        tracker: Optional tracker ("bytetrack", "botsort"). None = no tracking.
        imgsz: YOLO input resolution.
        skip_frames: Process every Nth frame (0 = every frame).
        jpeg_quality: JPEG encoding quality (0-100).

    Yields:
        MJPEG frame bytes for StreamingResponse.
    """
    from omnivis.detectors import detect
    from omnivis.tracking import track

    cap = open_capture(source)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", source)
        return

    state = _registry.get_or_create(source)
    frame_idx = 0

    try:
        while not state.stop_requested:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            state.frame_count = frame_idx

            # Skip frames for performance
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                continue

            # Detect or track
            if tracker:
                dets = track(frame, tracker=tracker, confidence=confidence, imgsz=imgsz)
            else:
                dets = detect(frame, model_type=model_type, confidence=confidence, imgsz=imgsz)

            # Record detections
            for d in dets:
                state.detection_history.append(d.to_dict())

            # Draw and encode
            annotated = draw_detections(frame, dets)
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )

    finally:
        cap.release()
        logger.info("Stream ended: %s (%d frames)", source, frame_idx)
