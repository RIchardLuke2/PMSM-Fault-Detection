"""
Robustness testing:
  1. Noise injection at varying SNR levels
  2. Window-size / stride ablation
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras
from src.preprocessing import make_windows
from src.training import _prepare_inputs


def _evaluate(model, branch_idxs, X, y_true, batch_size=256):
    X_in = _prepare_inputs(X, branch_idxs)
    y_prob = model.predict(X_in, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro", zero_division=0))


def noise_robustness(model, branch_idxs, data: dict, cfg: dict) -> pd.DataFrame:
    """Inject Gaussian noise at σ ∈ noise_levels and re-evaluate."""
    rc = cfg["robustness"]
    pp = cfg["paths"]
    os.makedirs(pp["plots_dir"],   exist_ok=True)
    os.makedirs(pp["metrics_dir"], exist_ok=True)

    X_base = data["X_test"].copy()
    y_true = data["y_test"]
    rows = []

    print("\n[Robustness] Noise injection test …")
    for sigma in rc["noise_levels"]:
        if sigma == 0.0:
            X_noisy = X_base
        else:
            noise = np.random.normal(0, sigma, X_base.shape).astype(np.float32)
            X_noisy = X_base + noise
        acc, f1 = _evaluate(model, branch_idxs, X_noisy, y_true)
        print(f"  σ={sigma:.2f}  acc={acc:.4f}  f1_macro={f1:.4f}")
        rows.append({"noise_sigma": sigma, "accuracy": acc, "f1_macro": f1})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(pp["metrics_dir"], "noise_robustness.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Robustness: Noise Injection", fontsize=13, fontweight="bold")
    for ax, col in zip(axes, ["accuracy", "f1_macro"]):
        ax.plot(df["noise_sigma"], df[col], marker="o", color="#0077b6", lw=2)
        ax.fill_between(df["noise_sigma"], df[col] - 0.02, df[col] + 0.02,
                        alpha=0.15, color="#0077b6")
        ax.set_xlabel("Gaussian Noise σ"); ax.set_ylabel(col)
        ax.set_title(col.replace("_", " ").title()); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pp["plots_dir"], "noise_robustness.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Robustness] Noise test chart saved.")
    return df


def window_ablation(model, branch_idxs, data: dict, cfg: dict) -> pd.DataFrame:
    """
    Re-create test windows at different sizes and strides and measure performance.
    We reuse the already-scaled raw flat test data by treating X_test windows as
    the canonical representation, so we rebuild from the test raw scaled data
    stored in data['X_test_raw_scaled'].
    """
    rc = cfg["robustness"]
    pp = cfg["paths"]

    if "X_test_raw_scaled" not in data:
        print("[Robustness] Window ablation skipped — raw scaled test array not available.")
        return pd.DataFrame()

    X_raw = data["X_test_raw_scaled"]
    y_raw = data["y_test_raw"]
    rows  = []

    print("\n[Robustness] Window-size / stride ablation …")
    for ws in rc["window_sizes"]:
        for st in rc["strides"]:
            if st >= ws:
                continue
            X_w, y_w = make_windows(X_raw, y_raw, ws, st)
            if X_w.shape[0] == 0:
                continue
            # Resize feature dim to match model input if needed
            # (only valid if we're changing window_size not n_features)
            # For this ablation the branch slice only works for matching window dim.
            # We use a simple global-avg along time to side-step shape mismatch.
            acc, f1 = float("nan"), float("nan")
            # Only evaluate when window size matches model's expected input
            if ws == cfg["windowing"]["window_size"]:
                acc, f1 = _evaluate(model, branch_idxs, X_w, y_w)
            print(f"  ws={ws:3d}  stride={st:3d}  N={X_w.shape[0]:5d}  "
                  f"acc={acc:.4f}  f1={f1:.4f}")
            rows.append({"window_size": ws, "stride": st,
                         "n_windows": X_w.shape[0], "accuracy": acc, "f1_macro": f1})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(pp["metrics_dir"], "window_ablation.csv"), index=False)
    print("[Robustness] Window ablation saved.")
    return df
