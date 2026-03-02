"""Full drone detection pipeline — heuristics + audio + multi-modal fusion.

Adapted from MediaSense:
- drone_heuristic_boost.py (behavioral signals + Kalman filter)
- drone_audio_classifier.py (AST + FFT frequency analysis)
- multimodal_drone_fusion.py (video + audio fusion)

Usage:
    from omnisee.drone_pipeline import detect_drone, DroneDetectionSession

    # Quick detection
    dets = detect_drone(frame)

    # Full session with audio
    with DroneDetectionSession() as session:
        session.start_audio()
        for frame in stream:
            dets = session.process_frame(yolo_dets, frame_num)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from omnivis._types import Detection, DroneDetection, get_priority

logger = logging.getLogger(__name__)

# ── Heuristic configuration ──────────────────────────────────────────────

MAX_BOOST = 0.30
MAX_PENALTY = 0.50
HISTORY_LEN = 60
MIN_HISTORY_FRAMES = 3
ASSUMED_FPS = 30.0

BOOST_WEIGHTS = {
    "hover": 0.15, "rigid_frame": 0.12, "altitude_hold": 0.08,
    "blink": 0.08, "aspect_ratio": 0.05, "sharp_turn": 0.02,
    "sky_region": 0.15, "consistent_motion": 0.10, "small_object": 0.08,
}

PENALTY_WEIGHTS = {
    "flapping": -0.20, "wrong_aspect": -0.12, "erratic": -0.10,
    "linear": -0.05, "ground_region": -0.25, "too_large": -0.15,
    "bbox_jitter": -0.35,
}

# Fusion weights
VIDEO_WEIGHT = 0.7
AUDIO_WEIGHT = 0.3
CONFIRMATION_BOOST = 0.15

# Drone audio classes with relevance weights
DRONE_AUDIO_CLASSES = {
    "Helicopter": 1.00, "Propeller, airscrew": 1.00, "Light aircraft": 0.95,
    "Buzzing": 0.95, "Whir": 0.95, "Whirring": 0.95, "Humming": 0.85,
    "Electric motor": 0.85, "Aircraft engine": 0.90, "Aircraft": 0.85,
    "Small engine": 0.80, "Mechanical fan": 0.50,
}


# ── Kalman Tracker ────────────────────────────────────────────────────────

class KalmanTracker:
    """Kalman filter for position+velocity tracking. State: [x, y, vx, vy]."""

    def __init__(self, initial_pos: Tuple[float, float], dt: float = 1.0 / ASSUMED_FPS):
        self.dt = dt
        self.state = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=np.float64)
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        q = 1.0
        self.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0], [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0], [0, q*dt**3/2, 0, q*dt**2],
        ], dtype=np.float64)
        self.R = np.eye(2, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self.frames_tracked = 0
        self.prediction_errors: deque = deque(maxlen=30)

    def predict(self) -> Tuple[float, float]:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (self.state[0], self.state[1])

    def update(self, measurement: Tuple[float, float]) -> float:
        z = np.array(measurement, dtype=np.float64)
        predicted = np.array([self.state[0], self.state[1]])
        error = float(np.linalg.norm(z - predicted))
        self.prediction_errors.append(error)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ (z - self.H @ self.state)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.frames_tracked += 1
        return error

    def get_velocity(self) -> Tuple[float, float]:
        return (self.state[2], self.state[3])

    def get_speed(self) -> float:
        return float(np.sqrt(self.state[2] ** 2 + self.state[3] ** 2))

    def get_track_quality(self) -> float:
        if self.frames_tracked < 3 or len(self.prediction_errors) < 3:
            return 0.0
        avg = float(np.mean(self.prediction_errors))
        if avg < 5:
            return 1.0
        elif avg < 50:
            return 1.0 - (avg - 5) / 45
        return 0.0


# ── Heuristic Boost ───────────────────────────────────────────────────────

@dataclass
class _TrackHistory:
    positions: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    bboxes: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    brightness: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    directions: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    kalman: Optional[KalmanTracker] = None
    last_update: float = 0.0


@dataclass
class BoostResult:
    boosted_confidence: float
    adjustment: float
    priority: str
    signatures: List[str] = field(default_factory=list)
    rejections: List[str] = field(default_factory=list)
    drone_likely: bool = True


class DroneHeuristicBoost:
    """Behavioral heuristic analysis for drone vs bird/plane discrimination.

    Positive signals: hover, rigid frame, altitude hold, LED blink, aspect ratio, sharp turns, sky region.
    Negative signals: flapping, wrong aspect, erratic motion, linear path, ground region, too large, jitter.
    """

    def __init__(self):
        self._tracks: Dict[int, _TrackHistory] = {}
        self._lock = Lock()

    def update(self, track_id: int, bbox: List[float], frame_gray: Optional[np.ndarray] = None):
        """Update track history with new detection."""
        with self._lock:
            if track_id not in self._tracks:
                self._tracks[track_id] = _TrackHistory()

            h = self._tracks[track_id]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            h.positions.append((cx, cy))
            h.bboxes.append(bbox)
            h.last_update = time.time()

            # Kalman init/update
            if h.kalman is None:
                h.kalman = KalmanTracker((cx, cy))
            else:
                h.kalman.predict()
                h.kalman.update((cx, cy))

            # Direction tracking
            if len(h.positions) >= 2:
                prev = h.positions[-2]
                dx, dy = cx - prev[0], cy - prev[1]
                angle = float(np.degrees(np.arctan2(dy, dx)))
                h.directions.append(angle)

            # Brightness in bbox region
            if frame_gray is not None:
                x1, y1, x2, y2 = [int(c) for c in bbox]
                roi_h, roi_w = frame_gray.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(roi_w, x2), min(roi_h, y2)
                if x2 > x1 and y2 > y1:
                    roi = frame_gray[y1:y2, x1:x2]
                    h.brightness.append(float(np.mean(roi)))

    def get_boost(self, track_id: int, raw_confidence: float, *, frame_height: int = 0) -> BoostResult:
        """Compute confidence adjustment based on behavioral analysis."""
        with self._lock:
            h = self._tracks.get(track_id)
            if h is None or len(h.positions) < MIN_HISTORY_FRAMES:
                return BoostResult(boosted_confidence=raw_confidence, adjustment=0.0,
                                   priority=get_priority(raw_confidence))

        sigs: List[str] = []
        rejs: List[str] = []
        boost = 0.0
        penalty = 0.0

        positions = list(h.positions)
        bboxes = list(h.bboxes)

        # ── Positive signals ──────────────────────────────────────────
        # Hover detection
        if len(positions) >= 10:
            recent = positions[-10:]
            displacements = [np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2) for a, b in zip(recent, recent[1:])]
            if np.mean(displacements) < 3.0:
                boost += BOOST_WEIGHTS["hover"]
                sigs.append("hover")

        # Rigid frame (stable bbox shape)
        if len(bboxes) >= 5:
            recent_bboxes = bboxes[-5:]
            widths = [b[2] - b[0] for b in recent_bboxes]
            heights = [b[3] - b[1] for b in recent_bboxes]
            w_var = np.std(widths) / (np.mean(widths) + 1e-6)
            h_var = np.std(heights) / (np.mean(heights) + 1e-6)
            if w_var < 0.08 and h_var < 0.08:
                boost += BOOST_WEIGHTS["rigid_frame"]
                sigs.append("rigid_frame")

        # Altitude hold
        if len(positions) >= 10:
            y_vals = [p[1] for p in positions[-10:]]
            y_var = np.std(y_vals)
            if y_var < 5.0:
                boost += BOOST_WEIGHTS["altitude_hold"]
                sigs.append("altitude_hold")

        # LED blink
        if len(h.brightness) >= 20:
            bright = list(h.brightness)[-20:]
            diffs = [abs(b - a) for a, b in zip(bright, bright[1:])]
            peaks = sum(1 for d in diffs if d > 15)
            if 3 <= peaks <= 10:
                boost += BOOST_WEIGHTS["blink"]
                sigs.append("blink")

        # Aspect ratio (drone-like: 0.7-1.4)
        if bboxes:
            last = bboxes[-1]
            w = last[2] - last[0]
            h_box = last[3] - last[1]
            aspect = w / (h_box + 1e-6)
            if 0.7 <= aspect <= 1.4:
                boost += BOOST_WEIGHTS["aspect_ratio"]
                sigs.append("aspect_ratio")

        # Sharp turns
        if len(h.directions) >= 5:
            dirs = list(h.directions)[-5:]
            for a, b in zip(dirs, dirs[1:]):
                diff = abs(a - b)
                if diff > 180:
                    diff = 360 - diff
                if diff > 60:
                    boost += BOOST_WEIGHTS["sharp_turn"]
                    sigs.append("sharp_turn")
                    break

        # Sky region boost
        if frame_height > 0 and bboxes:
            cy = (bboxes[-1][1] + bboxes[-1][3]) / 2
            if cy < frame_height * 0.6:
                boost += BOOST_WEIGHTS["sky_region"]
                sigs.append("sky_region")

        # Small object boost
        if bboxes:
            last = bboxes[-1]
            size = max(last[2] - last[0], last[3] - last[1])
            if size < 80:
                boost += BOOST_WEIGHTS["small_object"]
                sigs.append("small_object")

        # Kalman track quality
        if h.kalman and h.kalman.get_track_quality() > 0.7:
            boost += BOOST_WEIGHTS["consistent_motion"]
            sigs.append("consistent_motion")

        # ── Negative signals ──────────────────────────────────────────
        # Flapping (varying bbox shape)
        if len(bboxes) >= 10:
            recent_bboxes = bboxes[-10:]
            widths = [b[2] - b[0] for b in recent_bboxes]
            w_var = np.std(widths) / (np.mean(widths) + 1e-6)
            if w_var > 0.20:
                penalty += abs(PENALTY_WEIGHTS["flapping"])
                rejs.append("flapping")

        # Wrong aspect
        if bboxes:
            last = bboxes[-1]
            w = last[2] - last[0]
            h_box = last[3] - last[1]
            aspect = w / (h_box + 1e-6)
            if aspect > 3.0 or aspect < 0.3:
                penalty += abs(PENALTY_WEIGHTS["wrong_aspect"])
                rejs.append("wrong_aspect")

        # Ground region
        if frame_height > 0 and bboxes:
            cy = (bboxes[-1][1] + bboxes[-1][3]) / 2
            if cy > frame_height * 0.85:
                penalty += abs(PENALTY_WEIGHTS["ground_region"])
                rejs.append("ground_region")

        # Too large
        if bboxes:
            last = bboxes[-1]
            size = max(last[2] - last[0], last[3] - last[1])
            if size > 400:
                penalty += abs(PENALTY_WEIGHTS["too_large"])
                rejs.append("too_large")

        # Bbox jitter (random walk)
        if len(positions) >= 15:
            recent = positions[-15:]
            disps = [np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2) for a, b in zip(recent, recent[1:])]
            if np.std(disps) > 3 * np.mean(disps) + 1e-6:
                penalty += abs(PENALTY_WEIGHTS["bbox_jitter"])
                rejs.append("bbox_jitter")

        # Clamp
        adjustment = min(boost, MAX_BOOST) - min(penalty, MAX_PENALTY)
        final = max(0.0, min(1.0, raw_confidence + adjustment))
        drone_likely = adjustment >= -0.05

        return BoostResult(
            boosted_confidence=round(final, 3),
            adjustment=round(adjustment, 3),
            priority=get_priority(final),
            signatures=sigs,
            rejections=rejs,
            drone_likely=drone_likely,
        )

    def cleanup(self, active_ids: set):
        """Remove stale tracks not in active_ids."""
        with self._lock:
            stale = [k for k in self._tracks if k not in active_ids]
            for k in stale:
                del self._tracks[k]


# ── Audio Classifier ──────────────────────────────────────────────────────

def compute_drone_frequency_score(audio: np.ndarray, sr: int = 16000) -> float:
    """FFT-based drone frequency signature detection."""
    try:
        try:
            from scipy.fft import rfft, rfftfreq
        except ImportError:
            from numpy.fft import rfft, rfftfreq

        fft_vals = np.abs(rfft(audio))
        freqs = rfftfreq(len(audio), 1 / sr)
        max_val = np.max(fft_vals)
        if max_val < 1e-10:
            return 0.0
        fft_vals = fft_vals / max_val
        median_val = np.median(fft_vals) + 1e-9

        motor_mask = (freqs >= 80) & (freqs <= 250)
        motor_energy = float(np.mean(fft_vals[motor_mask])) if np.any(motor_mask) else 0.0

        blade_mask = (freqs >= 18) & (freqs <= 50)
        blade_energy = float(np.mean(fft_vals[blade_mask])) if np.any(blade_mask) else 0.0

        harmonic_mask = (freqs >= 250) & (freqs <= 500)
        harmonic_energy = float(np.mean(fft_vals[harmonic_mask])) if np.any(harmonic_mask) else 0.0

        peak_sharpness = float(np.max(fft_vals) / median_val)
        if motor_energy < 0.05 or peak_sharpness < 2.0:
            return 0.0

        score = motor_energy * 0.6 + blade_energy * 0.2 + harmonic_energy * 0.2
        return float(min(score * 2.0, 1.0))
    except Exception:
        return 0.0


class DroneAudioClassifier:
    """Drone audio classification using specialized AST model + FFT.

    Primary: preszzz/drone-audio-detection-05-17-trial-0 (99.6% accuracy)
    Fallback: MIT/ast-finetuned-audioset-10-10-0.4593
    """

    SPECIALIZED_MODEL = "preszzz/drone-audio-detection-05-17-trial-0"
    FALLBACK_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(self, *, device: str = "auto", use_specialized: bool = True):
        self.device = self._detect_device(device)
        self.sample_rate = 16000
        self.use_specialized = use_specialized
        self._specialized = None
        self._fallback = None
        self.history: deque = deque(maxlen=5)
        self._ema_conf = 0.0

    def _detect_device(self, device: str) -> str:
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def _load_specialized(self):
        if self._specialized is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            import torch
            device_id = 0 if self.device == "cuda" else -1
            self._specialized = hf_pipeline("audio-classification", model=self.SPECIALIZED_MODEL,
                                            device=device_id, model_kwargs={"torch_dtype": torch.float32})
            logger.info("Specialized drone audio model loaded")
        except Exception as e:
            logger.warning("Specialized model failed: %s", e)

    def _load_fallback(self):
        if self._fallback is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            import torch
            device_id = 0 if self.device == "cuda" else -1
            dtype = torch.float16 if device_id == 0 else torch.float32
            self._fallback = hf_pipeline("audio-classification", model=self.FALLBACK_MODEL,
                                         device=device_id, model_kwargs={"torch_dtype": dtype})
        except Exception as e:
            logger.error("AST fallback failed: %s", e)

    def classify(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """Classify audio for drone presence."""
        audio_array = np.asarray(audio_array, dtype=np.float32)
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)

        if sample_rate != self.sample_rate:
            try:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=self.sample_rate)
            except ImportError:
                pass

        rms = float(np.sqrt(np.mean(audio_array ** 2)) + 1e-9)
        if rms < 1e-4:
            return self._empty()

        min_samples = self.sample_rate
        if len(audio_array) < min_samples:
            audio_array = np.pad(audio_array, (0, min_samples - len(audio_array)))

        # Try specialized first
        if self.use_specialized:
            self._load_specialized()
            if self._specialized:
                return self._classify_specialized(audio_array)

        # Fallback to AST + FFT
        self._load_fallback()
        if self._fallback is None:
            return self._empty()
        return self._classify_generic(audio_array)

    def _classify_specialized(self, audio: np.ndarray) -> Dict[str, Any]:
        results = self._specialized({"raw": audio, "sampling_rate": self.sample_rate})
        drone_score = 0.0
        for r in results:
            label = r["label"].lower()
            if "drone" in label and "non" not in label:
                drone_score = max(drone_score, r["score"])
            elif label in ("positive", "1", "yes", "detected"):
                drone_score = max(drone_score, r["score"])

        self._ema_conf = 0.7 * self._ema_conf + 0.3 * drone_score
        is_drone = self._ema_conf > 0.03
        self.history.append(is_drone)
        smoothed = is_drone and (self._ema_conf >= 0.7 or
                                  (len(self.history) >= 2 and all(list(self.history)[-2:])))
        return {"is_drone": is_drone, "confidence": self._ema_conf, "drone_class": "drone",
                "smoothed": smoothed, "model": self.SPECIALIZED_MODEL}

    def _classify_generic(self, audio: np.ndarray) -> Dict[str, Any]:
        results = self._fallback({"raw": audio, "sampling_rate": self.sample_rate}, top_k=10)

        ast_max = 0.0
        drone_class = None
        for r in results:
            if r["label"] in DRONE_AUDIO_CLASSES:
                w = DRONE_AUDIO_CLASSES[r["label"]]
                ws = r["score"] * w
                if ws > ast_max:
                    ast_max = ws
                    drone_class = r["label"]

        freq_score = compute_drone_frequency_score(audio, self.sample_rate)
        combined = max(ast_max, freq_score * 0.6)
        self._ema_conf = 0.7 * self._ema_conf + 0.3 * combined
        is_drone = self._ema_conf > 0.03
        self.history.append(is_drone)
        smoothed = is_drone and (self._ema_conf >= 0.7 or
                                  (len(self.history) >= 2 and all(list(self.history)[-2:])))
        return {"is_drone": is_drone, "confidence": self._ema_conf, "drone_class": drone_class,
                "smoothed": smoothed, "frequency_score": freq_score, "model": self.FALLBACK_MODEL}

    def classify_file(self, audio_path: str) -> Dict[str, Any]:
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except ImportError:
            from scipy.io import wavfile
            sr, audio = wavfile.read(audio_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
        return self.classify(audio, sr)

    def reset(self):
        self.history.clear()
        self._ema_conf = 0.0

    def _empty(self) -> Dict[str, Any]:
        return {"is_drone": False, "confidence": 0.0, "drone_class": None, "smoothed": False}


# ── Multi-Modal Fusion ────────────────────────────────────────────────────

class MultiModalDroneFusion:
    """Fuses YOLO video detections with audio classification."""

    def __init__(self, *, video_weight: float = VIDEO_WEIGHT, audio_weight: float = AUDIO_WEIGHT,
                 confirmation_boost: float = CONFIRMATION_BOOST):
        self.video_weight = video_weight
        self.audio_weight = audio_weight
        self.confirmation_boost = confirmation_boost
        self.detection_history: deque = deque(maxlen=10)

    def fuse(
        self,
        video_detections: List[Dict[str, Any]],
        audio_result: Optional[Dict[str, Any]],
        frame_number: Optional[int] = None,
    ) -> List[DroneDetection]:
        fused = []
        ts = time.time()

        audio_drone = audio_result.get("is_drone", False) if audio_result else False
        audio_smoothed = audio_result.get("smoothed", False) if audio_result else False
        audio_conf = audio_result.get("confidence", 0) if audio_result else 0
        audio_class = audio_result.get("drone_class") if audio_result else None

        if video_detections:
            for det in video_detections:
                v_conf = det.get("confidence", 0)
                bbox = det.get("bbox") or det.get("xyxy")
                if bbox is not None and hasattr(bbox, "tolist"):
                    bbox = bbox.tolist()

                if audio_drone and audio_smoothed:
                    f_conf = min(1.0, self.video_weight * v_conf + self.audio_weight * audio_conf + self.confirmation_boost)
                    priority = get_priority(f_conf, require_fused_for_critical=True, is_fused=True)
                    source = "fused"
                else:
                    f_conf = v_conf
                    priority = get_priority(f_conf, require_fused_for_critical=True, is_fused=False)
                    source = "video"

                d = DroneDetection(
                    source=source, confidence=round(f_conf, 3), priority=priority,
                    bbox=bbox, audio_class=audio_class if audio_drone else None,
                    detected_class=det.get("class", "drone"), timestamp=ts,
                    frame_number=frame_number,
                    heuristic_signatures=det.get("heuristic_signatures", []),
                    metadata={"video_conf": v_conf, "audio_conf": audio_conf if audio_drone else 0},
                )
                fused.append(d)

        elif audio_drone and audio_smoothed:
            d = DroneDetection(
                source="audio", confidence=round(audio_conf * 0.5, 3), priority="MEDIUM",
                audio_class=audio_class, detected_class="drone", timestamp=ts,
                frame_number=frame_number,
                metadata={"audio_conf": audio_conf, "note": "audio-only (no visual)"},
            )
            fused.append(d)

        self.detection_history.extend(fused)
        return fused


# ── Session Manager ───────────────────────────────────────────────────────

class DroneDetectionSession:
    """Complete drone detection session with video + audio."""

    def __init__(self, *, enable_audio: bool = True):
        self.enable_audio = enable_audio
        self.heuristic = DroneHeuristicBoost()
        self.fusion = MultiModalDroneFusion()
        self.audio_classifier: Optional[DroneAudioClassifier] = None
        self._audio_monitor = None
        self._last_audio: Optional[Dict] = None

    def start_audio(self) -> bool:
        if not self.enable_audio:
            return False
        try:
            import sounddevice as sd
            import queue

            self.audio_classifier = DroneAudioClassifier()
            self._audio_queue: queue.Queue = queue.Queue(maxsize=20)
            self._audio_running = True

            chunk_samples = int(16000 * 3.0)
            hop_samples = int(chunk_samples * 0.5)
            self._audio_buffer = np.array([], dtype=np.float32)
            self._chunk_samples = chunk_samples
            self._hop_samples = hop_samples

            def callback(indata, frames, time_info, status):
                self._audio_buffer = np.concatenate([self._audio_buffer, indata.flatten()])
                while len(self._audio_buffer) >= self._chunk_samples:
                    chunk = self._audio_buffer[:self._chunk_samples]
                    self._audio_buffer = self._audio_buffer[self._hop_samples:]
                    result = self.audio_classifier.classify(chunk)
                    try:
                        self._audio_queue.put_nowait(result)
                    except queue.Full:
                        try:
                            self._audio_queue.get_nowait()
                            self._audio_queue.put_nowait(result)
                        except:
                            pass

            self._stream = sd.InputStream(samplerate=16000, channels=1, dtype=np.float32,
                                          callback=callback, blocksize=1600)
            self._stream.start()
            logger.info("Audio monitoring started")
            return True
        except Exception as e:
            logger.warning("Audio start failed: %s", e)
            return False

    def stop_audio(self):
        if hasattr(self, "_stream") and self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._audio_running = False

    def process_frame(
        self,
        video_detections: List[Dict[str, Any]],
        frame_number: Optional[int] = None,
    ) -> List[DroneDetection]:
        # Get latest audio
        if hasattr(self, "_audio_queue"):
            import queue
            try:
                result = self._audio_queue.get_nowait()
                self._last_audio = result
            except queue.Empty:
                pass

        return self.fusion.fuse(video_detections, self._last_audio, frame_number)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop_audio()


# ── Convenience function ──────────────────────────────────────────────────

def detect_drone(
    frame: np.ndarray,
    *,
    audio_array: Optional[np.ndarray] = None,
    preprocess: bool = True,
    sahi: bool = False,
    confidence: float = 0.25,
) -> List[DroneDetection]:
    """Quick drone detection on a single frame.

    Args:
        frame: BGR numpy array.
        audio_array: Optional audio samples for multi-modal fusion.
        preprocess: Apply CLAHE + bilateral preprocessing.
        sahi: Use SAHI sliced inference for small objects.
        confidence: Detection confidence threshold.
    """
    from omnivis.detectors import detect

    if preprocess:
        from omnivis.preprocessing import preprocess_frame
        frame = preprocess_frame(frame)

    dets = detect(frame, model_type="Drone", confidence=confidence, sahi=sahi)

    # Convert to fusion format
    video_dets = [{"bbox": d.bbox, "confidence": d.confidence, "class": d.class_name} for d in dets]

    audio_result = None
    if audio_array is not None:
        classifier = DroneAudioClassifier()
        audio_result = classifier.classify(audio_array)

    fusion = MultiModalDroneFusion()
    return fusion.fuse(video_dets, audio_result)
