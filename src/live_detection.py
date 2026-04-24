"""
Live / Streaming Inference Engine for PMSM FDD
===============================================

Architecture
------------
  Sensor Source → SensorBuffer → Preprocessor → ModelInferencer
       → PostProcessor (smoothing + thresholding) → AlertRouter

Two sensor sources are supported:
  1. SimulatedSource  — streams rows from a CSV file at configurable speed
                        (use this for demo / testing without hardware)
  2. RealSensorSource — stub for real hardware integration (e.g. serial port,
                        NI-DAQ, OPC-UA); plug in your driver here

Usage
-----
  python live_run.py                        # simulated stream
  python live_run.py --source csv           # explicit simulated
  python live_run.py --duration 60          # run for 60 s then stop
  python live_run.py --no-smooth            # disable temporal smoothing
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import threading
import queue
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Generator

from utils.io_utils import load_artifact, load_config
from src.alert_system import AlertRouter, FAULT_META


# ──────────────────────────────────────────────────────────────────────────────
# 1. Sensor sources
# ──────────────────────────────────────────────────────────────────────────────

class BaseSensorSource(ABC):
    """Yields one dict of sensor readings per call to next_sample()."""

    FEATURES = ["Ia", "Ib", "VDC", "IDC", "T1", "T2", "T3", "VD"]

    @abstractmethod
    def stream(self) -> Generator[dict, None, None]:
        """Infinite generator of sensor reading dicts."""


class SimulatedSource(BaseSensorSource):
    """
    Streams rows from a CSV file, simulating a real-time sensor feed.
    Optionally injects a synthetic fault at a specified row index for demo.
    """

    def __init__(self, cfg: dict):
        lc = cfg["live_detection"]
        sc = lc.get("simulation", {})
        self.csv_path    = sc.get("source_csv", cfg["data"]["raw_path"])
        self.speed       = sc.get("speed_multiplier", 1.0)
        self.inject_at   = sc.get("inject_fault_at", None)
        self.inject_cls  = sc.get("inject_fault_class", "F3")
        self.sample_rate = lc.get("sample_rate_hz", 100)
        self.sleep_s     = (1.0 / self.sample_rate) / max(self.speed, 0.01)
        self._df         = None

    def _load(self):
        df = pd.read_csv(self.csv_path)
        # Keep only the 8 features (+ FDD if present for ground truth)
        keep = self.FEATURES + (["FDD"] if "FDD" in df.columns else [])
        return df[keep].reset_index(drop=True)

    def stream(self) -> Generator[dict, None, None]:
        if self._df is None:
            self._df = self._load()
        print(f"[SimSource] Streaming {len(self._df):,} rows from {self.csv_path}")
        print(f"[SimSource] Sample rate: {self.sample_rate} Hz  |  "
              f"Speed: {self.speed}×  |  sleep/sample: {self.sleep_s*1000:.1f} ms")
        i = 0
        while True:                        # loop forever (real sensor never ends)
            row = self._df.iloc[i % len(self._df)]
            sample = {f: float(row[f]) for f in self.FEATURES}
            sample["ground_truth"] = str(row["FDD"]) if "FDD" in row else "?"

            # Synthetic fault injection for demo
            if self.inject_at and i == self.inject_at:
                print(f"\n[SimSource] 💉 Injecting {self.inject_cls} fault at sample {i}\n")
                sample = self._inject_fault(sample, self.inject_cls)

            time.sleep(self.sleep_s)
            yield sample
            i += 1

    @staticmethod
    def _inject_fault(sample: dict, fault_cls: str) -> dict:
        """Perturb a sample to simulate a specific fault signature."""
        if fault_cls in ("F1", "F3", "F5"):
            sample["Ia"] *= 0.4      # phase-A current drop
        elif fault_cls in ("F2", "F4", "F6"):
            sample["Ib"] *= 0.4
        elif fault_cls == "F7":
            sample["Ia"] *= 0.3; sample["Ib"] *= 0.3
        elif fault_cls == "F8":
            sample["Ia"] *= 2.2; sample["VDC"] *= 0.93; sample["T1"] += 20
        return sample


class RealSensorSource(BaseSensorSource):
    """
    Stub for real hardware integration.
    Replace the body of stream() with your driver code:
      - Serial / UART: use pyserial
      - NI-DAQ:        use nidaqmx
      - OPC-UA:        use opcua / asyncua
      - Modbus:        use pymodbus
    """

    def stream(self) -> Generator[dict, None, None]:
        raise NotImplementedError(
            "RealSensorSource.stream() must be implemented for your hardware.\n"
            "Replace this stub with your sensor driver code."
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Streaming buffer
# ──────────────────────────────────────────────────────────────────────────────

class SensorBuffer:
    """
    Thread-safe circular buffer.
    Accumulates raw sensor readings until a full window is available,
    then emits a (window_size × n_features) numpy array.
    """

    def __init__(self, window_size: int, n_features: int):
        self.window_size = window_size
        self.n_features  = n_features
        self._buf        = deque(maxlen=window_size)  # holds scaled rows

    def push(self, scaled_row: np.ndarray):
        """Push one scaled sensor reading (1D array of shape (n_features,))."""
        self._buf.append(scaled_row)

    @property
    def ready(self) -> bool:
        return len(self._buf) == self.window_size

    def get_window(self) -> np.ndarray:
        """Return (window_size, n_features) array — copy so buffer can keep filling."""
        return np.array(self._buf, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Real-time preprocessor
# ──────────────────────────────────────────────────────────────────────────────

class RealTimePreprocessor:
    """
    Applies the SAME scaler fitted during training.
    CRITICAL: never refit the scaler on live data — that would cause data leakage
    and destroy the calibration established during training.
    """

    def __init__(self, scaler, features: list):
        self.scaler   = scaler
        self.features = features

    def transform(self, sample: dict) -> np.ndarray:
        """Dict of sensor readings → scaled 1D numpy array."""
        raw = np.array([[sample[f] for f in self.features]], dtype=np.float64)
        scaled = self.scaler.transform(raw)          # (1, n_features)
        return scaled[0].astype(np.float32)          # (n_features,)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Model inferencer
# ──────────────────────────────────────────────────────────────────────────────

class ModelInferencer:
    """
    Wraps the trained Keras model.
    Handles multi-branch input splitting identically to training.
    """

    def __init__(self, model, encoder, cfg: dict):
        self.model    = model
        self.encoder  = encoder
        self.features = cfg["data"]["features"]
        self.branches = cfg["model"].get("branches", {})
        self._feat_idx = {f: i for i, f in enumerate(self.features)}

    def predict(self, window: np.ndarray) -> tuple[str, float, np.ndarray]:
        """
        window : (window_size, n_features) float32 array
        Returns (class_label, confidence, all_probs)
        """
        X = window[np.newaxis]  # (1, W, F)

        if self.branches:
            inputs = [X[:, :, [self._feat_idx[f] for f in feats
                               if f in self._feat_idx]]
                      for feats in self.branches.values()
                      if any(f in self._feat_idx for f in feats)]
        else:
            inputs = X

        probs      = self.model.predict(inputs, verbose=0)[0]  # (9,)
        cls_idx    = int(np.argmax(probs))
        cls_label  = self.encoder.inverse_transform([cls_idx])[0]
        confidence = float(probs[cls_idx])
        return cls_label, confidence, probs


# ──────────────────────────────────────────────────────────────────────────────
# 5. Post-processor (smoothing + thresholding + persistence)
# ──────────────────────────────────────────────────────────────────────────────

class PostProcessor:
    """
    Applies three layers of robustness on top of raw model predictions:

    Layer 1 — Confidence threshold
      If confidence < threshold → mark as UNKNOWN (don't commit to any class)

    Layer 2 — Temporal smoothing (majority vote)
      Take the mode of the last N predictions to suppress flickering

    Layer 3 — Fault persistence gate
      Only fire an alert if the smoothed class has been FAULT for ≥ K consecutive
      windows.  Healthy (F0) resets the counter immediately.
    """

    def __init__(self, cfg: dict):
        lc = cfg["live_detection"]
        self.conf_thresh    = lc.get("confidence_threshold", 0.80)
        self.smooth_n       = lc.get("smoothing_window", 5)
        self.persist_k      = lc.get("fault_persistence", 3)
        self.use_unknown    = lc.get("unknown_fallback", True)
        self._history       = deque(maxlen=self.smooth_n)
        self._fault_streak  = 0
        self._last_alert    = None

    def process(self, raw_class: str, confidence: float) -> dict:
        # Layer 1: threshold
        if confidence < self.conf_thresh and raw_class != "F0":
            effective = "UNKNOWN" if self.use_unknown else raw_class
        else:
            effective = raw_class

        # Layer 2: smoothing
        self._history.append(effective)
        smoothed = max(set(self._history), key=self._history.count)

        # Layer 3: persistence gate
        is_fault = smoothed != "F0"
        if is_fault:
            self._fault_streak += 1
        else:
            self._fault_streak = 0

        alert_fired = is_fault and (self._fault_streak >= self.persist_k)

        return {
            "raw_class":     raw_class,
            "smoothed_class": smoothed,
            "confidence":    confidence,
            "fault_streak":  self._fault_streak,
            "alert_fired":   alert_fired,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 6. Live detection engine (orchestrator)
# ──────────────────────────────────────────────────────────────────────────────

class LiveDetectionEngine:
    """
    Main streaming pipeline:
      Source → Buffer → Preprocessor → Inferencer → PostProcessor → AlertRouter

    Call run() to start.  Pass duration_s to stop automatically.
    """

    def __init__(self, cfg: dict, source: BaseSensorSource = None):
        self.cfg     = cfg
        self.lc      = cfg["live_detection"]
        self.wsize   = self.lc["window_size"]
        self.feats   = cfg["data"]["features"]

        # Load artefacts
        print("[Live] Loading model and preprocessing artefacts …")
        import tensorflow as tf
        self.model   = tf.keras.models.load_model(cfg["paths"]["model_path"])
        self.scaler  = load_artifact(cfg["paths"]["scaler_path"])
        self.encoder = load_artifact(cfg["paths"]["encoder_path"])

        # Components
        self.source        = source or SimulatedSource(cfg)
        self.buffer        = SensorBuffer(self.wsize, len(self.feats))
        self.preprocessor  = RealTimePreprocessor(self.scaler, self.feats)
        self.inferencer    = ModelInferencer(self.model, self.encoder, cfg)
        self.postprocessor = PostProcessor(cfg)
        self.alert_router  = AlertRouter(cfg)

        self._stop_event   = threading.Event()
        self._window_id    = 0
        self._stats        = {"total_windows": 0, "faults_detected": 0,
                               "alerts_fired": 0, "start_time": None}

    def run(self, duration_s: float = None, stride: int = None):
        """
        Start the live detection loop.

        Parameters
        ----------
        duration_s : float, optional
            Stop after this many seconds (None = run forever).
        stride : int, optional
            Emit a prediction every `stride` new samples (default = window_size // 2).
            Lower stride = higher prediction frequency but more CPU.
        """
        if stride is None:
            stride = self.cfg["windowing"].get("stride", self.wsize // 2)

        self._stats["start_time"] = time.time()
        deadline = (time.time() + duration_s) if duration_s else None

        print(f"\n{'═'*60}")
        print(f"  PMSM Live Fault Detection Engine")
        print(f"  Window: {self.wsize} samples  |  Stride: {stride}")
        print(f"  Conf threshold: {self.lc['confidence_threshold']:.0%}  |  "
              f"Smoothing: {self.lc['smoothing_window']}  |  "
              f"Persistence: {self.lc['fault_persistence']}")
        print(f"{'═'*60}\n")

        sample_count = 0
        try:
            for sample in self.source.stream():
                if self._stop_event.is_set():
                    break
                if deadline and time.time() > deadline:
                    print(f"\n[Live] Duration {duration_s}s reached — stopping.")
                    break

                # Preprocess & push to buffer
                scaled = self.preprocessor.transform(sample)
                self.buffer.push(scaled)
                sample_count += 1

                # Only predict when we have a full window AND on stride boundary
                if self.buffer.ready and (sample_count % stride == 0):
                    self._window_id += 1
                    self._stats["total_windows"] += 1
                    window = self.buffer.get_window()

                    # Inference
                    raw_cls, conf, probs = self.inferencer.predict(window)

                    # Post-processing
                    result = self.postprocessor.process(raw_cls, conf)

                    if result["smoothed_class"] != "F0":
                        self._stats["faults_detected"] += 1
                    if result["alert_fired"]:
                        self._stats["alerts_fired"] += 1

                    # Build alert event
                    event = {
                        "timestamp":       datetime.now().isoformat(timespec="milliseconds"),
                        "window_id":       self._window_id,
                        "predicted_class": result["smoothed_class"],
                        "raw_class":       result["raw_class"],
                        "smoothed_class":  result["smoothed_class"],
                        "confidence":      result["confidence"],
                        "fault_streak":    result["fault_streak"],
                        "alert_fired":     result["alert_fired"],
                        "ground_truth":    sample.get("ground_truth", "?"),
                        "all_probs":       {
                            self.encoder.inverse_transform([i])[0]: round(float(p), 4)
                            for i, p in enumerate(probs)
                        },
                    }

                    self.alert_router.dispatch(event)

        except KeyboardInterrupt:
            print("\n[Live] Interrupted by user.")

        self._print_summary()

    def stop(self):
        self._stop_event.set()

    def _print_summary(self):
        elapsed = time.time() - self._stats["start_time"]
        s = self._stats
        print(f"\n{'═'*60}")
        print(f"  Session Summary")
        print(f"  Runtime          : {elapsed:.1f} s")
        print(f"  Windows processed: {s['total_windows']}")
        print(f"  Fault windows    : {s['faults_detected']}  "
              f"({100*s['faults_detected']/max(s['total_windows'],1):.1f}%)")
        print(f"  Alerts fired     : {s['alerts_fired']}")
        if self.lc.get("log_to_file"):
            print(f"  Log saved to     : {self.lc['log_path']}")
        print(f"{'═'*60}\n")
