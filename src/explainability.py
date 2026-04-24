"""
Explainability module for PMSM FDD.

Methods
-------
1. Permutation Feature Importance (model-agnostic, works on windows)
2. Attention-weight extraction (temporal + channel attention)
3. Gradient-based Saliency (GradCAM-lite via tf.GradientTape)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score


FAULT_NAMES = ["F0","F1","F2","F3","F4","F5","F6","F7","F8"]


# ── 1. Permutation Feature Importance ─────────────────────────────────────────

def permutation_importance(model, branch_idxs, X_test: np.ndarray,
                           y_test: np.ndarray, feature_names: list,
                           cfg: dict, n_repeats: int = 5) -> pd.DataFrame:
    pp = cfg["paths"]
    os.makedirs(pp["plots_dir"],   exist_ok=True)
    os.makedirs(pp["metrics_dir"], exist_ok=True)

    def _predict(X):
        if branch_idxs is None:
            inp = X
        else:
            inp = [X[:, :, idxs] for idxs in branch_idxs]
        probs = model.predict(inp, batch_size=256, verbose=0)
        return np.argmax(probs, axis=1)

    base_acc = accuracy_score(y_test, _predict(X_test))
    importances = []
    print("\n[Explain] Computing permutation importance …")
    for fi, fname in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            np.random.shuffle(X_perm[:, :, fi])
            perm_acc = accuracy_score(y_test, _predict(X_perm))
            drops.append(base_acc - perm_acc)
        mean_drop = float(np.mean(drops))
        std_drop  = float(np.std(drops))
        importances.append({"feature": fname, "importance_mean": mean_drop,
                             "importance_std": std_drop})
        print(f"  {fname:<8}: drop={mean_drop:+.4f} ± {std_drop:.4f}")

    df = pd.DataFrame(importances).sort_values("importance_mean", ascending=False)
    df.to_csv(os.path.join(pp["metrics_dir"], "feature_importance.csv"), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e63946" if v > 0 else "#4cc9f0" for v in df["importance_mean"]]
    ax.barh(df["feature"], df["importance_mean"], xerr=df["importance_std"],
            color=colors, edgecolor="white", height=0.6, error_kw=dict(capsize=4))
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Mean Accuracy Drop (higher = more important)")
    ax.set_title("Permutation Feature Importance\n(CNN-BiLSTM on Test Set)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pp["plots_dir"], "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Explain] Importance chart saved.")
    return df


# ── 2. Gradient-based saliency ────────────────────────────────────────────────

def compute_saliency(model, X_sample: np.ndarray, branch_idxs,
                     true_label: int, cfg: dict):
    """
    GradCAM-lite: compute |∂loss/∂input| for a single window to visualise
    which time-steps / features are most influential.
    """
    pp = cfg["paths"]
    os.makedirs(pp["plots_dir"], exist_ok=True)

    X_in_np = X_sample[np.newaxis]  # (1, W, F)

    if branch_idxs is None:
        inputs_tf = [tf.Variable(X_in_np.astype(np.float32))]
        model_inputs = inputs_tf[0]
    else:
        inputs_tf = [tf.Variable(X_in_np[:, :, idxs].astype(np.float32))
                     for idxs in branch_idxs]
        model_inputs = inputs_tf

    with tf.GradientTape() as tape:
        tape.watch(inputs_tf)
        preds = model(model_inputs)              # (1, 9)
        loss  = preds[0, true_label]

    grads = tape.gradient(loss, inputs_tf)
    if isinstance(grads, list):
        # Reconstruct into full feature space
        n_features = X_sample.shape[-1]
        window_size = X_sample.shape[0]
        saliency = np.zeros((window_size, n_features), dtype=np.float32)
        for g, idxs in zip(grads, branch_idxs):
            saliency[:, idxs] = np.abs(g.numpy()[0])
    else:
        saliency = np.abs(grads.numpy()[0])

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(saliency.T, ax=ax, cmap="rocket",
                xticklabels=False,
                yticklabels=cfg["data"]["features"])
    ax.set_xlabel("Time Steps")
    ax.set_title(f"Saliency Map — True class: {FAULT_NAMES[true_label]}", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(pp["plots_dir"], "saliency_map.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Explain] Saliency map saved → {path}")
    return saliency
