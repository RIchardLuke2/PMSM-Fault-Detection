"""Full evaluation suite: accuracy, F1, ROC, confusion matrix."""
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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from sklearn.preprocessing import label_binarize


def _predict_probs(model, X, branch_idxs, device, batch_size=256):
    model.eval()
    probs_all = []

    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            Xb = torch.tensor(X[s:s + batch_size], dtype=torch.float32)

            if branch_idxs is None:
                logits = model(Xb.to(device))
            else:
                logits = model([Xb[:, :, idx].to(device) for idx in branch_idxs])

            probs_all.append(F.softmax(logits, dim=-1).cpu().numpy())

    return np.concatenate(probs_all, axis=0)


def _safe_multiclass_auc(y_true, y_prob, n_cls):
    """
    Return macro OvR AUC if possible, otherwise NaN.
    This avoids crashes when y_true contains only one class.
    """
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")

        y_bin = label_binarize(y_true, classes=np.arange(n_cls))

        # If y_bin ends up with shape (n, 1) in some edge cases, skip AUC.
        if y_bin.ndim != 2 or y_bin.shape[1] != n_cls:
            return float("nan")

        return roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        return float("nan")


def evaluate_model(model, branch_idxs, data, cfg, encoder, device=None, tag="test") -> dict:
    from src.training import get_device

    if device is None:
        device = get_device()

    pp = cfg["paths"]
    Path(pp["plots_dir"]).mkdir(parents=True, exist_ok=True)
    Path(pp["metrics_dir"]).mkdir(parents=True, exist_ok=True)

    X = data["X_" + tag]
    y_true = data["y_" + tag]
    names = list(encoder.classes_)
    n_cls = cfg["model"]["num_classes"]
    labels = np.arange(n_cls)

    y_prob = _predict_probs(model, X, branch_idxs, device)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    auc = _safe_multiclass_auc(y_true, y_prob, n_cls)

    print("\n" + "=" * 55)
    print("  {} | acc={:.4f} prec={:.4f} rec={:.4f} f1={:.4f} auc={:.4f}".format(
        tag.upper(), acc, prec, rec, f1, auc
    ))
    print("=" * 55)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        zero_division=0
    )
    print(report_text)

    # Per-class CSV
    rpt = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0
    )

    skip = {"accuracy", "macro avg", "weighted avg"}
    per_class = pd.DataFrame(
        {k: v for k, v in rpt.items() if k not in skip}
    ).T.reset_index().rename(columns={"index": "class"})

    per_class.to_csv(
        os.path.join(pp["metrics_dir"], f"per_class_{tag}.csv"),
        index=False
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=names,
        yticklabels=names,
        linewidths=0.5,
        linecolor="white"
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(
        "Confusion Matrix — {}\nAcc={:.3f}  F1={:.3f}".format(tag.upper(), acc, f1),
        fontsize=13,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(pp["plots_dir"], f"confusion_matrix_{tag}.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    # ROC curves
    y_bin = label_binarize(y_true, classes=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_cls))

    for i in range(n_cls):
        # Skip classes that are absent in y_true for this split
        if len(np.unique(y_bin[:, i])) < 2:
            continue

        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            ai = roc_auc_score(y_bin[:, i], y_prob[:, i])
            ax.plot(
                fpr,
                tpr,
                color=colors[i],
                lw=1.8,
                label="{} (AUC={:.3f})".format(names[i], ai)
            )
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(
        "ROC Curves (OvR) — {}\nmacro AUC={:.4f}".format(tag.upper(), auc),
        fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(pp["plots_dir"], f"roc_curves_{tag}.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    print("[Eval] Plots saved ->", pp["plots_dir"])
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }