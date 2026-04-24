"""
Evaluation module for PMSM FDD.

Produces:
  - Accuracy / Precision / Recall / F1 (macro & per-class)
  - Confusion matrix heatmap
  - ROC curves (one-vs-rest)
  - Per-class metrics table saved as CSV
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
from tensorflow import keras


FAULT_NAMES = {
    "F0": "Healthy",
    "F1": "S1 Open",
    "F2": "S2 Open",
    "F3": "S3 Open",
    "F4": "S4 Open",
    "F5": "S5 Open",
    "F6": "S6 Open",
    "F7": "Two-Switch OC",
    "F8": "Short-Circuit",
}


def _split_inputs(X: np.ndarray, branch_idxs) -> list | np.ndarray:
    if branch_idxs is None:
        return X
    return [X[:, :, idxs] for idxs in branch_idxs]


def evaluate_model(model, branch_idxs, data: dict, cfg: dict,
                   encoder, tag: str = "test") -> dict:
    pp = cfg["paths"]
    os.makedirs(pp["plots_dir"],   exist_ok=True)
    os.makedirs(pp["metrics_dir"], exist_ok=True)

    X = data[f"X_{tag}"]
    y_true = data[f"y_{tag}"]
    class_names = [f"{c} ({FAULT_NAMES.get(c, '')})" for c in encoder.classes_]
    short_names = list(encoder.classes_)

    # Predictions
    X_in = _split_inputs(X, branch_idxs)
    y_prob = model.predict(X_in, batch_size=256, verbose=0)  # (N, 9)
    y_pred = np.argmax(y_prob, axis=1)

    # ── Scalar metrics ────────────────────────────────────────────────────────
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ROC-AUC (OvR)
    n_classes = cfg["model"]["num_classes"]
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    try:
        roc_auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float("nan")

    print(f"\n{'='*60}")
    print(f"  Evaluation — {tag.upper()} SET")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f} (macro)")
    print(f"  Recall    : {rec:.4f} (macro)")
    print(f"  F1        : {f1:.4f} (macro)")
    print(f"  ROC-AUC   : {roc_auc:.4f} (OvR macro)")
    print(f"\n{classification_report(y_true, y_pred, target_names=short_names, zero_division=0)}")

    # ── Per-class DataFrame ───────────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=short_names,
                                   output_dict=True, zero_division=0)
    per_class = {k: v for k, v in report.items()
                 if k not in ["accuracy", "macro avg", "weighted avg"]}
    df_perclass = pd.DataFrame(per_class).T.reset_index().rename(columns={"index": "class"})
    out_csv = os.path.join(pp["metrics_dir"], f"per_class_{tag}.csv")
    df_perclass.to_csv(out_csv, index=False)
    print(f"[Eval] Per-class CSV → {out_csv}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=short_names, yticklabels=short_names,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True",      fontsize=13)
    ax.set_title(f"Confusion Matrix — {tag.upper()} Set\n"
                 f"Accuracy={acc:.3f}  F1={f1:.3f}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(pp["plots_dir"], f"confusion_matrix_{tag}.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Confusion matrix → {cm_path}")

    # ── ROC curves ────────────────────────────────────────────────────────────
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_i = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=colors[i], lw=1.8,
                label=f"{short_names[i]} (AUC={auc_i:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (One-vs-Rest) — {tag.upper()}\nmacro AUC={roc_auc:.4f}",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(pp["plots_dir"], f"roc_curves_{tag}.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] ROC curves → {roc_path}")

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=roc_auc)
