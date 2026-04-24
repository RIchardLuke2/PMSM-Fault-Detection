"""
live_run.py — Live / Streaming Fault Detection Entry Point
===========================================================

Usage examples
--------------
  # Stream from your converted_dataset.csv (simulated real-time)
  python live_run.py

  # Run for exactly 30 seconds then print summary
  python live_run.py --duration 30

  # Run at 5× real-time speed for quick testing
  python live_run.py --speed 5

  # Disable temporal smoothing (raw predictions only)
  python live_run.py --no-smooth

  # Use a different config or different CSV source
  python live_run.py --config configs/config.yaml --source-csv data/converted_dataset.csv

  # Inject a fault at sample 200 for demo purposes
  python live_run.py --inject-at 200 --inject-class F7

How it maps to your project
----------------------------
  run.py        → batch training + evaluation pipeline  (train once)
  live_run.py   → streaming inference on trained model  (deploy continuously)

The model, scaler, and encoder loaded here are the artefacts saved by run.py.
Make sure you have run `python run.py` at least once before launching live_run.py.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.io_utils import load_config, save_config
from src.live_detection import LiveDetectionEngine, SimulatedSource


def parse_args():
    p = argparse.ArgumentParser(
        description="PMSM Live Fault Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config",       default="configs/config.yaml",
                   help="Path to config.yaml")
    p.add_argument("--source-csv",   default=None,
                   help="CSV file to stream from (overrides config)")
    p.add_argument("--duration",     type=float, default=None,
                   help="Stop after N seconds (default: run forever)")
    p.add_argument("--speed",        type=float, default=None,
                   help="Simulation speed multiplier (1.0=real-time, 10=10x)")
    p.add_argument("--stride",       type=int,   default=None,
                   help="Predict every N new samples (default: window_size//2)")
    p.add_argument("--no-smooth",    action="store_true",
                   help="Disable temporal smoothing (raw predictions)")
    p.add_argument("--conf",         type=float, default=None,
                   help="Override confidence threshold (0.0–1.0)")
    p.add_argument("--inject-at",    type=int,   default=None,
                   help="Inject a synthetic fault at this sample index (demo)")
    p.add_argument("--inject-class", default="F3",
                   help="Fault class to inject (F0–F8)")
    p.add_argument("--no-log",       action="store_true",
                   help="Disable CSV logging")
    p.add_argument("--mqtt",         action="store_true",
                   help="Enable MQTT alerts (requires paho-mqtt + broker)")
    return p.parse_args()


def apply_overrides(cfg: dict, args) -> dict:
    """Apply CLI overrides onto the config dict."""
    lc = cfg["live_detection"]
    sc = lc.setdefault("simulation", {})

    if args.source_csv:
        sc["source_csv"] = args.source_csv
    if args.speed is not None:
        sc["speed_multiplier"] = args.speed
    if args.inject_at is not None:
        sc["inject_fault_at"]    = args.inject_at
        sc["inject_fault_class"] = args.inject_class
    if args.no_smooth:
        lc["smoothing_window"] = 1
    if args.conf is not None:
        lc["confidence_threshold"] = args.conf
    if args.no_log:
        lc["log_to_file"] = False
    if args.mqtt:
        modes = lc.get("alert_modes", ["console"])
        if "mqtt" not in modes:
            modes.append("mqtt")
        lc["alert_modes"] = modes

    return cfg


def check_artefacts(cfg: dict):
    """Make sure the model and preprocessing artefacts exist."""
    missing = []
    for key in ["model_path", "scaler_path", "encoder_path"]:
        p = cfg["paths"][key]
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        print("\n[live_run] ❌ Missing artefacts — run `python run.py` first:\n")
        for m in missing:
            print(f"   • {m}")
        sys.exit(1)


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = apply_overrides(cfg, args)

    check_artefacts(cfg)

    source = SimulatedSource(cfg)
    engine = LiveDetectionEngine(cfg, source=source)
    engine.run(duration_s=args.duration, stride=args.stride)


if __name__ == "__main__":
    main()
