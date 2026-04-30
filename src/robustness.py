"""
Robustness testing:
  1. Noise injection at varying SNR levels
  2. Window-size / stride ablation
"""
"""Noise injection test and window-size/stride ablation."""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score
from src.evaluation import _predict_probs
from src.preprocessing import make_windows


def noise_robustness(model, branch_idxs, data, cfg) -> pd.DataFrame:
    from src.training import get_device
    device = get_device()
    pp = cfg["paths"]
    Path(pp["plots_dir"]).mkdir(parents=True, exist_ok=True)
    Path(pp["metrics_dir"]).mkdir(parents=True, exist_ok=True)

    X0    = data["X_test"].copy()
    yt    = data["y_test"]
    rows  = []
    rng   = np.random.default_rng(42)

    print("\n[Robustness] Noise injection ...")
    for sig in cfg["robustness"]["noise_levels"]:
        Xn = X0 + rng.normal(0, sig, X0.shape).astype(np.float32) if sig > 0 else X0
        yp = np.argmax(_predict_probs(model, Xn, branch_idxs, device), axis=1)
        acc = accuracy_score(yt, yp)
        f1  = f1_score(yt, yp, average="macro", zero_division=0)
        print("  sigma={:.2f}  acc={:.4f}  f1={:.4f}".format(sig, acc, f1))
        rows.append({"noise_sigma": sig, "accuracy": acc, "f1_macro": f1})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(pp["metrics_dir"], "noise_robustness.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Robustness: Noise Injection", fontsize=13, fontweight="bold")
    for ax, col in zip(axes, ["accuracy","f1_macro"]):
        ax.plot(df["noise_sigma"], df[col], "o-", color="#0077b6", lw=2)
        ax.fill_between(df["noise_sigma"],
                        df[col]-0.02, df[col]+0.02, alpha=0.15, color="#0077b6")
        ax.set_xlabel("Gaussian Noise Sigma"); ax.set_ylabel(col)
        ax.set_title(col.replace("_"," ").title()); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pp["plots_dir"], "noise_robustness.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    return df


def window_ablation(model, branch_idxs, data, cfg) -> pd.DataFrame:
    if "X_test_raw_scaled" not in data:
        print("[Robustness] Skipped — raw test array missing.")
        return pd.DataFrame()

    from src.training import get_device
    device  = get_device()
    pp      = cfg["paths"]
    X_raw   = data["X_test_raw_scaled"]
    y_raw   = data["y_test_raw"]
    base_ws = cfg["windowing"]["window_size"]
    rows    = []

    print("\n[Robustness] Window ablation ...")
    for ws in cfg["robustness"]["window_sizes"]:
        for st in cfg["robustness"]["strides"]:
            if st >= ws:
                continue
            Xw, yw = make_windows(X_raw, y_raw, ws, st)
            if Xw.shape[0] == 0:
                continue
            if ws == base_ws:
                yp  = np.argmax(_predict_probs(model, Xw, branch_idxs, device), axis=1)
                acc = accuracy_score(yw, yp)
                f1  = f1_score(yw, yp, average="macro", zero_division=0)
            else:
                acc = f1 = float("nan")
            print("  ws={:3d}  stride={:2d}  n={:5d}  acc={:.4f}  f1={:.4f}".format(
                  ws, st, Xw.shape[0], acc, f1))
            rows.append({"window_size": ws, "stride": st,
                         "n_windows": Xw.shape[0],
                         "accuracy": acc, "f1_macro": f1})

    df = pd.DataFrame(rows)
    Path(pp["metrics_dir"]).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(pp["metrics_dir"], "window_ablation.csv"), index=False)
    return df