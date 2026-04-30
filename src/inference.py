"""
Single-sample inference.

CLI:
    python src/inference.py \
        --Ia 0.85 --Ib 0.88 --VDC 558 --IDC 10.2 \
        --T1 41   --T2 40   --T3 40   --VD 0.71

API:
    from src.inference import PMSMInference
    inf = PMSMInference()
    print(inf.predict({"Ia":0.85,"Ib":0.88,"VDC":558,"IDC":10.2,
                        "T1":41,"T2":40,"T3":40,"VD":0.71}))
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F

FAULT_DESC = {
    "F0": "Healthy — no fault detected",
    "F1": "Switch S1 open-circuit fault",
    "F2": "Switch S2 open-circuit fault",
    "F3": "Switch S3 open-circuit fault",
    "F4": "Switch S4 open-circuit fault",
    "F5": "Switch S5 open-circuit fault",
    "F6": "Switch S6 open-circuit fault",
    "F7": "Two-switch open-circuit fault",
    "F8": "Short-circuit fault",
}


class PMSMInference:
    def __init__(self, config_path="configs/config.yaml"):
        from utils.io_utils import load_config, load_artifact
        from src.model import build_cnn_bilstm
        cfg  = load_config(config_path)
        dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = build_cnn_bilstm(cfg)
        ckpt = torch.load(cfg["paths"]["model_path"], map_location=dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(dev).eval()
        self.model    = model
        self.scaler   = load_artifact(cfg["paths"]["scaler_path"])
        self.encoder  = load_artifact(cfg["paths"]["encoder_path"])
        self.cfg      = cfg
        self.device   = dev
        self.features = cfg["data"]["features"]
        self.wsize    = cfg["windowing"]["window_size"]
        self.branches = cfg["model"].get("branches", {})
        self.feat_idx = {f: i for i, f in enumerate(self.features)}

    def _window(self, sample):
        raw = np.array([[sample[f] for f in self.features]], dtype=np.float64)
        row = self.scaler.transform(raw)[0].astype(np.float32)
        return np.tile(row, (self.wsize, 1))

    def _inputs(self, window):
        t = torch.tensor(window[np.newaxis], dtype=torch.float32).to(self.device)
        if not self.branches:
            return t
        return [t[:, :, [self.feat_idx[f] for f in feats if f in self.feat_idx]]
                for feats in self.branches.values()
                if any(f in self.feat_idx for f in feats)]

    @torch.no_grad()
    def predict(self, sample: dict) -> dict:
        probs   = F.softmax(self.model(self._inputs(self._window(sample))), -1)[0].cpu().numpy()
        idx     = int(np.argmax(probs))
        lbl     = self.encoder.inverse_transform([idx])[0]
        all_p   = {self.encoder.inverse_transform([i])[0]: round(float(p), 4)
                   for i, p in enumerate(probs)}
        return {"predicted_class": lbl,
                "description":     FAULT_DESC.get(lbl, ""),
                "confidence":      round(float(probs[idx]), 4),
                "all_probabilities": all_p}

    @torch.no_grad()
    def predict_window(self, window: np.ndarray) -> dict:
        probs = F.softmax(self.model(self._inputs(window)), -1)[0].cpu().numpy()
        idx   = int(np.argmax(probs))
        lbl   = self.encoder.inverse_transform([idx])[0]
        return {"predicted_class": lbl, "description": FAULT_DESC.get(lbl, ""),
                "confidence": round(float(probs[idx]), 4)}


def _cli():
    parser = argparse.ArgumentParser()
    for f in ["Ia","Ib","VDC","IDC","T1","T2","T3","VD"]:
        parser.add_argument("--"+f, type=float, required=True)
    args = parser.parse_args()
    inf  = PMSMInference()
    res  = inf.predict(vars(args))
    print("\n" + "="*50)
    print("  Predicted : " + res["predicted_class"])
    print("  Desc      : " + res["description"])
    print("  Confidence: {:.2%}".format(res["confidence"]))
    print("\n  All probabilities:")
    for lbl, p in sorted(res["all_probabilities"].items(), key=lambda x: -x[1]):
        print("    {}: {:.4f}  {}".format(lbl, p, chr(9608)*int(p*30)))
    print("="*50)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _cli()