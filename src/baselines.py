"""
Baseline model comparison: SVM, Random Forest, Extra-Trees, Simple CNN, Simple LSTM.
Results saved as CSV + bar chart.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras
from src.model import build_simple_cnn, build_simple_lstm
from src.training import _prepare_inputs


def _train_predict_sklearn(clf, X_tr, y_tr, X_te, y_te, name: str):
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"  {name:<25} acc={acc:.4f}  f1={f1:.4f}  ({elapsed:.1f}s)")
    return {"model": name, "accuracy": acc, "f1_macro": f1, "train_time_s": round(elapsed, 1)}


def _train_predict_dl(model_fn, cfg, data, name: str):
    model, bidxs = model_fn(cfg)
    tc = cfg["training"]
    nc = cfg["model"]["num_classes"]
    X_tr = _prepare_inputs(data["X_train"], bidxs)
    X_te = _prepare_inputs(data["X_test"],  bidxs)
    y_tr = keras.utils.to_categorical(data["y_train"], nc)
    y_te_cat = keras.utils.to_categorical(data["y_test"],  nc)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                         restore_best_weights=True, verbose=0)]
    t0 = time.time()
    model.fit(X_tr, y_tr, validation_split=0.15, epochs=40,
              batch_size=tc["batch_size"], callbacks=cb, verbose=0)
    elapsed = time.time() - t0

    y_prob = model.predict(X_te, batch_size=256, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(data["y_test"], y_pred)
    f1  = f1_score(data["y_test"], y_pred, average="macro", zero_division=0)
    print(f"  {name:<25} acc={acc:.4f}  f1={f1:.4f}  ({elapsed:.1f}s)")
    keras.backend.clear_session()
    return {"model": name, "accuracy": acc, "f1_macro": f1, "train_time_s": round(elapsed, 1)}


def run_baselines(data: dict, cfg: dict) -> pd.DataFrame:
    pp = cfg["paths"]
    os.makedirs(pp["metrics_dir"], exist_ok=True)
    os.makedirs(pp["plots_dir"],   exist_ok=True)

    X_tr_f = data["X_train_flat"]
    X_te_f = data["X_test_flat"]
    y_tr   = data["y_train"]
    y_te   = data["y_test"]

    print("\n[Baselines] Training comparison models …")
    results = []

    results.append(_train_predict_sklearn(
        SVC(kernel="rbf", C=10, gamma="scale", decision_function_shape="ovr",
            class_weight="balanced", random_state=42),
        X_tr_f, y_tr, X_te_f, y_te, "SVM (RBF)"))

    results.append(_train_predict_sklearn(
        RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               n_jobs=-1, random_state=42),
        X_tr_f, y_tr, X_te_f, y_te, "Random Forest"))

    results.append(_train_predict_sklearn(
        ExtraTreesClassifier(n_estimators=200, class_weight="balanced",
                             n_jobs=-1, random_state=42),
        X_tr_f, y_tr, X_te_f, y_te, "Extra-Trees"))

    results.append(_train_predict_dl(build_simple_cnn,  cfg, data, "Simple CNN"))
    results.append(_train_predict_dl(build_simple_lstm, cfg, data, "Simple LSTM"))

    df = pd.DataFrame(results)
    csv_path = os.path.join(pp["metrics_dir"], "baseline_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Baselines] Results saved → {csv_path}")

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Baseline Model Comparison", fontsize=14, fontweight="bold")
    colors = ["#0077b6", "#2d9057", "#e63946", "#f4a261", "#8338ec"]
    for ax, metric in zip(axes, ["accuracy", "f1_macro"]):
        bars = ax.bar(df["model"], df[metric], color=colors, edgecolor="white", width=0.6)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(pp["plots_dir"], "baseline_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Baselines] Chart → {chart_path}")
    return df
