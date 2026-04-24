"""
Inference script: single sample (or batch) → predicted fault class + confidence.

Usage (CLI):
    python src/inference.py --Ia 0.85 --Ib 0.88 --VDC 558 --IDC 10.2 \
                             --T1 41 --T2 40 --T3 40 --VD 0.71

Usage (API):
    from src.inference import PMSMInference
    inf = PMSMInference.load()
    result = inf.predict_single({"Ia": 0.85, "Ib": 0.88, "VDC": 558,
                                  "IDC": 10.2, "T1": 41, "T2": 40, "T3": 40, "VD": 0.71})
    print(result)
"""
import argparse
import numpy as np
from pathlib import Path
from utils.io_utils import load_artifact, load_config
import tensorflow as tf


FAULT_DESCRIPTIONS = {
    "F0": "Healthy — no fault detected",
    "F1": "Switch S1 open-circuit fault",
    "F2": "Switch S2 open-circuit fault",
    "F3": "Switch S3 open-circuit fault",
    "F4": "Switch S4 open-circuit fault",
    "F5": "Switch S5 open-circuit fault",
    "F6": "Switch S6 open-circuit fault",
    "F7": "Two-switch open-circuit fault",
    "F8": "Short-circuit (DC bus) fault",
}


class PMSMInference:
    def __init__(self, model, scaler, encoder, cfg):
        self.model   = model
        self.scaler  = scaler
        self.encoder = encoder
        self.cfg     = cfg
        self.features    = cfg["data"]["features"]
        self.window_size = cfg["windowing"]["window_size"]
        self.branch_cfg  = cfg["model"].get("branches", {})

    @classmethod
    def load(cls, config_path: str = "configs/config.yaml") -> "PMSMInference":
        cfg     = load_config(config_path)
        model   = tf.keras.models.load_model(cfg["paths"]["model_path"])
        scaler  = load_artifact(cfg["paths"]["scaler_path"])
        encoder = load_artifact(cfg["paths"]["encoder_path"])
        return cls(model, scaler, encoder, cfg)

    def _to_window(self, sample_dict: dict) -> np.ndarray:
        """
        Expand a single sensor reading into a window by repeating + tiny noise,
        then normalise.  For real deployment, supply a proper time-series buffer.
        """
        raw = np.array([sample_dict[f] for f in self.features], dtype=np.float32)
        # Tile + jitter to fill window
        window_raw = np.tile(raw, (self.window_size, 1))
        window_raw += np.random.normal(0, 0.001, window_raw.shape).astype(np.float32)
        window_scaled = self.scaler.transform(window_raw)  # (W, F)
        return window_scaled

    def _prepare_branches(self, window: np.ndarray):
        """Split window into per-branch inputs."""
        feat_idx = {f: i for i, f in enumerate(self.features)}
        inputs = []
        for feats in self.branch_cfg.values():
            idxs = [feat_idx[f] for f in feats if f in feat_idx]
            inputs.append(window[np.newaxis, :, idxs].astype(np.float32))
        return inputs if inputs else window[np.newaxis].astype(np.float32)

    def predict_single(self, sample_dict: dict) -> dict:
        window    = self._to_window(sample_dict)
        inp       = self._prepare_branches(window)
        probs     = self.model.predict(inp, verbose=0)[0]  # (9,)
        cls_idx   = int(np.argmax(probs))
        cls_label = self.encoder.inverse_transform([cls_idx])[0]
        confidence = float(probs[cls_idx])

        return {
            "predicted_class":   cls_label,
            "description":       FAULT_DESCRIPTIONS.get(cls_label, ""),
            "confidence":        round(confidence, 4),
            "all_probabilities": {
                self.encoder.inverse_transform([i])[0]: round(float(p), 4)
                for i, p in enumerate(probs)
            }
        }

    def predict_window(self, window: np.ndarray) -> dict:
        """
        Direct inference on a pre-processed, scaled window (W, F).
        """
        inp     = self._prepare_branches(window)
        probs   = self.model.predict(inp, verbose=0)[0]
        cls_idx = int(np.argmax(probs))
        cls_label = self.encoder.inverse_transform([cls_idx])[0]
        return {
            "predicted_class": cls_label,
            "description":     FAULT_DESCRIPTIONS.get(cls_label, ""),
            "confidence":      round(float(probs[cls_idx]), 4),
        }


def _cli():
    parser = argparse.ArgumentParser(description="PMSM FDD — single-sample inference")
    for f in ["Ia", "Ib", "VDC", "IDC", "T1", "T2", "T3", "VD"]:
        parser.add_argument(f"--{f}", type=float, required=True)
    args = parser.parse_args()
    sample = vars(args)

    inf    = PMSMInference.load()
    result = inf.predict_single(sample)

    print("\n" + "="*50)
    print(f"  Prediction : {result['predicted_class']}")
    print(f"  Description: {result['description']}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print("\n  Class probabilities:")
    for k, v in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(v * 30)
        print(f"    {k}: {v:.4f} {bar}")
    print("="*50 + "\n")


if __name__ == "__main__":
    _cli()
