"""
Explainability module for PMSM FDD.

Methods
-------
1. Permutation Feature Importance (model-agnostic, works on windows)
2. Gradient-based Saliency (works for CNN / BiLSTM models)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from src.evaluation import _predict_probs

FAULT_NAMES = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]


def permutation_importance(model, branch_idxs, X_test, y_test,
                           feature_names, cfg, n_repeats=5) -> pd.DataFrame:
    from src.training import get_device

    device = get_device()
    pp = cfg["paths"]
    Path(pp["plots_dir"]).mkdir(parents=True, exist_ok=True)
    Path(pp["metrics_dir"]).mkdir(parents=True, exist_ok=True)

    base_probs = _predict_probs(model, X_test, branch_idxs, device)
    base = accuracy_score(y_test, np.argmax(base_probs, axis=1))

    rng = np.random.default_rng(42)
    rows = []

    print("\n[Explain] Permutation importance (base_acc={:.4f}) ...".format(base))
    for fi, fname in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            perm_idx = rng.permutation(len(Xp))
            Xp[:, :, fi] = Xp[perm_idx, :, fi]

            perm_probs = _predict_probs(model, Xp, branch_idxs, device)
            perm_acc = accuracy_score(y_test, np.argmax(perm_probs, axis=1))
            drops.append(base - perm_acc)

        m, s = float(np.mean(drops)), float(np.std(drops))
        rows.append({
            "feature": fname,
            "importance_mean": round(m, 5),
            "importance_std": round(s, 5),
        })
        print("  {:<8} drop={:+.4f} +/- {:.4f}".format(fname, m, s))

    df = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    df.to_csv(os.path.join(pp["metrics_dir"], "feature_importance.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e63946" if v > 0 else "#4cc9f0" for v in df["importance_mean"]]
    ax.barh(
        df["feature"],
        df["importance_mean"],
        xerr=df["importance_std"],
        color=colors,
        edgecolor="white",
        height=0.6,
        error_kw=dict(capsize=4, elinewidth=1),
    )
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Mean Accuracy Drop")
    ax.set_title("Permutation Feature Importance (CNN-BiLSTM)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pp["plots_dir"], "feature_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    return df


def compute_saliency(model, X_sample: np.ndarray, branch_idxs, true_label: int, cfg) -> np.ndarray:
    """
    Gradient saliency: |d logit[true_label] / d input|.

    Fixes:
    - disables CuDNN during backward pass to avoid:
      "cudnn RNN backward can only be called in training mode"
    - uses leaf tensors with requires_grad=True
    - restores the model's original training/eval state
    """
    from src.training import get_device

    device = get_device()
    pp = cfg["paths"]
    Path(pp["plots_dir"]).mkdir(parents=True, exist_ok=True)

    feat_names = cfg["data"]["features"]
    W, F_ = X_sample.shape

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    # Make a leaf tensor on the correct device
    X_t = torch.tensor(X_sample[np.newaxis], dtype=torch.float32, device=device)
    X_t.requires_grad_(True)

    try:
        with torch.backends.cudnn.flags(enabled=False):
            if branch_idxs is None:
                logits = model(X_t)
                label = int(true_label)
                score = logits[0, label]
                score.backward()

                saliency = X_t.grad.detach().abs().squeeze(0).cpu().numpy()

            else:
                inp_list = []
                for idxs in branch_idxs:
                    # Create leaf tensors for each branch input
                    b = X_t[:, :, idxs].detach().clone().requires_grad_(True)
                    inp_list.append((idxs, b))

                logits = model([b for _, b in inp_list])
                label = int(true_label)
                score = logits[0, label]
                score.backward()

                saliency = np.zeros((W, F_), dtype=np.float32)
                for idxs, b in inp_list:
                    if b.grad is not None:
                        saliency[:, idxs] = b.grad.detach().abs().squeeze(0).cpu().numpy()

    finally:
        if was_training:
            model.train()

    lname = FAULT_NAMES[true_label] if 0 <= int(true_label) < len(FAULT_NAMES) else str(true_label)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        saliency.T,
        ax=ax,
        cmap="rocket",
        xticklabels=False,
        yticklabels=feat_names,
        cbar_kws={"label": "Gradient magnitude"},
    )
    ax.set_xlabel("Time Steps")
    ax.set_title("Gradient Saliency — Class: {}".format(lname), fontweight="bold")
    plt.tight_layout()

    out = os.path.join(pp["plots_dir"], "saliency_map.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print("[Explain] Saliency map saved ->", out)
    return saliency