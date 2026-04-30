"""
Baseline model comparison: SVM, Random Forest, Extra-Trees, Simple CNN, Simple LSTM.
Results saved as CSV + bar chart.
"""
"""Baseline comparison: SVM, RF, ExtraTrees, SimpleCNN, SimpleLSTM."""
import os, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.model     import build_simple_cnn, build_simple_lstm
from src.training  import get_device, make_loader, _run_epoch, EarlyStopping
from src.evaluation import _predict_probs


def _sk_eval(clf, Xtr, ytr, Xte, yte, name):
    t0 = time.time()
    clf.fit(Xtr, ytr)
    elapsed = time.time() - t0
    yp  = clf.predict(Xte)
    acc = accuracy_score(yte, yp)
    f1  = f1_score(yte, yp, average="macro", zero_division=0)
    print("  {:<28} acc={:.4f}  f1={:.4f}  ({:.1f}s)".format(name, acc, f1, elapsed))
    return {"model": name, "accuracy": round(acc,4),
            "f1_macro": round(f1,4), "train_time_s": round(elapsed,1)}


def _dl_eval(build_fn, cfg, data, name):
    device  = get_device()
    model, bi = build_fn(cfg)
    model   = model.to(device)
    tc      = cfg["training"]
    crit    = nn.CrossEntropyLoss()
    opt     = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])
    tr_ld   = make_loader(data["X_train"], data["y_train"],
                          bi, tc["batch_size"], True,  device)
    vl_ld   = make_loader(data["X_val"],   data["y_val"],
                          bi, tc["batch_size"], False, device)
    es      = EarlyStopping(patience=6)
    t0      = time.time()
    for _ in range(40):
        _run_epoch(model, tr_ld, crit, opt,  device, bi, True)
        vl, _ = _run_epoch(model, vl_ld, crit, None, device, bi, False)
        if es.step(vl, model):
            break
    es.restore(model)
    elapsed = time.time() - t0
    yp  = np.argmax(_predict_probs(model, data["X_test"], bi, device), axis=1)
    acc = accuracy_score(data["y_test"], yp)
    f1  = f1_score(data["y_test"], yp, average="macro", zero_division=0)
    print("  {:<28} acc={:.4f}  f1={:.4f}  ({:.1f}s)".format(name, acc, f1, elapsed))
    return {"model": name, "accuracy": round(acc,4),
            "f1_macro": round(f1,4), "train_time_s": round(elapsed,1)}


def run_baselines(data: dict, cfg: dict) -> pd.DataFrame:
    pp = cfg["paths"]
    Path(pp["metrics_dir"]).mkdir(parents=True, exist_ok=True)
    Path(pp["plots_dir"]).mkdir(parents=True, exist_ok=True)

    Xtr, ytr = data["X_train_flat"], data["y_train"]
    Xte, yte = data["X_test_flat"],  data["y_test"]

    print("\n[Baselines] Running comparison...")
    results = [
        _sk_eval(SVC(kernel="rbf", C=10, gamma="scale",
                     class_weight="balanced", random_state=42),
                 Xtr, ytr, Xte, yte, "SVM (RBF)"),
        _sk_eval(RandomForestClassifier(n_estimators=200,
                     class_weight="balanced", n_jobs=-1, random_state=42),
                 Xtr, ytr, Xte, yte, "Random Forest"),
        _sk_eval(ExtraTreesClassifier(n_estimators=200,
                     class_weight="balanced", n_jobs=-1, random_state=42),
                 Xtr, ytr, Xte, yte, "Extra-Trees"),
        _dl_eval(build_simple_cnn,  cfg, data, "Simple CNN  (PyTorch)"),
        _dl_eval(build_simple_lstm, cfg, data, "Simple LSTM (PyTorch)"),
    ]
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(pp["metrics_dir"], "baseline_comparison.csv"), index=False)
    print("[Baselines] Saved.")

    colors = ["#0077b6","#2d9057","#e63946","#f4a261","#8338ec"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Baseline Model Comparison", fontsize=13, fontweight="bold")
    for ax, metric in zip(axes, ["accuracy","f1_macro"]):
        bars = ax.bar(df["model"], df[metric], color=colors,
                      edgecolor="white", width=0.55)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    "{:.3f}".format(b.get_height()),
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_title(metric.replace("_"," ").title())
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pp["plots_dir"], "baseline_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    return df