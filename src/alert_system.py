"""
Alert System for PMSM Live Fault Detection
===========================================
Supports three alert modes, configurable via config.yaml:
  console  — coloured terminal output (always active)
  file     — append to CSV log
  mqtt     — publish JSON payload to broker (optional, needs paho-mqtt)

Add new modes by subclassing BaseAlert and registering in AlertRouter.
"""

import csv
import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path


# ── Fault metadata ─────────────────────────────────────────────────────────────
FAULT_META = {
    "F0": dict(name="Healthy",            severity="OK",       color="\033[92m"),   # green
    "F1": dict(name="S1 Open-Circuit",    severity="WARNING",  color="\033[93m"),   # yellow
    "F2": dict(name="S2 Open-Circuit",    severity="WARNING",  color="\033[93m"),
    "F3": dict(name="S3 Open-Circuit",    severity="WARNING",  color="\033[93m"),
    "F4": dict(name="S4 Open-Circuit",    severity="WARNING",  color="\033[93m"),
    "F5": dict(name="S5 Open-Circuit",    severity="WARNING",  color="\033[93m"),
    "F6": dict(name="S6 Open-Circuit",    severity="WARNING",  color="\033[93m"),
    "F7": dict(name="Two-Switch OC",      severity="CRITICAL", color="\033[91m"),   # red
    "F8": dict(name="Short-Circuit (DC)", severity="CRITICAL", color="\033[91m"),
    "UNKNOWN": dict(name="Unknown Anomaly", severity="CAUTION", color="\033[95m"),  # magenta
}
RESET = "\033[0m"
BOLD  = "\033[1m"


# ── Base alert class ───────────────────────────────────────────────────────────
class BaseAlert(ABC):
    @abstractmethod
    def send(self, event: dict):
        """Send a fault alert event."""


# ── Console alert ──────────────────────────────────────────────────────────────
class ConsoleAlert(BaseAlert):
    """Rich coloured terminal output."""

    def send(self, event: dict):
        cls   = event["predicted_class"]
        meta  = FAULT_META.get(cls, FAULT_META["UNKNOWN"])
        ts    = event.get("timestamp", datetime.now().isoformat())
        conf  = event.get("confidence", 0.0)
        sev   = meta["severity"]
        col   = meta["color"]
        name  = meta["name"]
        smooth = event.get("smoothed_class", cls)

        if cls == "F0" and smooth == "F0":
            print(f"{col}[{ts}] ✅  NORMAL  ({name})  conf={conf:.2%}{RESET}")
        else:
            icon = "🔴" if sev == "CRITICAL" else "⚠️ "
            print(
                f"{col}{BOLD}[{ts}] {icon} FAULT DETECTED{RESET}\n"
                f"  Raw prediction  : {cls} — {name}\n"
                f"  Smoothed class  : {smooth}\n"
                f"  Severity        : {sev}\n"
                f"  Confidence      : {conf:.2%}\n"
                f"  Window #{event.get('window_id','?'):>5}"
                + RESET
            )


# ── File (CSV) alert ───────────────────────────────────────────────────────────
class FileAlert(BaseAlert):
    """Append every event to a CSV log."""

    FIELDS = ["timestamp", "window_id", "predicted_class", "smoothed_class",
              "confidence", "severity", "alert_fired"]

    def __init__(self, log_path: str):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(log_path).exists():
            with open(log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def send(self, event: dict):
        meta = FAULT_META.get(event.get("predicted_class", "UNKNOWN"), FAULT_META["UNKNOWN"])
        row  = {
            "timestamp":       event.get("timestamp", datetime.now().isoformat()),
            "window_id":       event.get("window_id", ""),
            "predicted_class": event.get("predicted_class", ""),
            "smoothed_class":  event.get("smoothed_class", ""),
            "confidence":      f"{event.get('confidence', 0):.4f}",
            "severity":        meta["severity"],
            "alert_fired":     event.get("alert_fired", False),
        }
        with open(self.log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)


# ── MQTT alert ─────────────────────────────────────────────────────────────────
class MQTTAlert(BaseAlert):
    """Publish JSON alert to an MQTT broker (requires paho-mqtt)."""

    def __init__(self, broker: str, port: int, topic: str):
        try:
            import paho.mqtt.client as mqtt
            self.client = mqtt.Client()
            self.client.connect(broker, port, keepalive=60)
            self.client.loop_start()
            self.topic = topic
            self._ok = True
            print(f"[MQTT] Connected to {broker}:{port}  topic={topic}")
        except Exception as e:
            print(f"[MQTT] ⚠ Could not connect: {e}  — MQTT alerts disabled.")
            self._ok = False

    def send(self, event: dict):
        if not self._ok:
            return
        payload = json.dumps(event)
        self.client.publish(self.topic, payload)


# ── Alert router ───────────────────────────────────────────────────────────────
class AlertRouter:
    """Fan-out a single event to all registered alert channels."""

    def __init__(self, cfg: dict):
        lc = cfg.get("live_detection", {})
        self.handlers: list[BaseAlert] = []

        modes = lc.get("alert_modes", ["console"])

        if "console" in modes:
            self.handlers.append(ConsoleAlert())

        if "file" in modes and lc.get("log_to_file", False):
            self.handlers.append(FileAlert(lc["log_path"]))

        if "mqtt" in modes:
            mc = lc.get("mqtt", {})
            self.handlers.append(
                MQTTAlert(mc.get("broker", "localhost"),
                          mc.get("port", 1883),
                          mc.get("topic", "pmsm/faults")))

    def dispatch(self, event: dict):
        for h in self.handlers:
            try:
                h.send(event)
            except Exception as e:
                print(f"[AlertRouter] Handler error: {e}")
