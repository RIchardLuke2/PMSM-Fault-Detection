"""
Training pipeline for the PMSM FDD deep model.
Includes:
  - class-weight computation
  - EarlyStopping / ReduceLROnPlateau / ModelCheckpoint callbacks
  - training-curve plots saved to outputs/plots/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, CSVLogger)


def compute_weights(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, cw)}


def _prepare_inputs(X: np.ndarray, branch_idxs: list | None) -> list | np.ndarray:
    """Split window array into branch sub-arrays if multi-branch."""
    if branch_idxs is None:
        return X
    return [X[:, :, idxs] for idxs in branch_idxs]


def build_callbacks(cfg: dict) -> list:
    tc = cfg["training"]
    pp = cfg["paths"]
    Path(pp["plots_dir"]).mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss",
                      patience=tc["early_stopping_patience"],
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss",
                          factor=tc["reduce_lr_factor"],
                          patience=tc["reduce_lr_patience"],
                          min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=pp["model_path"],
                        monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        CSVLogger(os.path.join(pp["metrics_dir"], "training_log.csv"), append=False),
    ]
    return callbacks


def plot_training_curves(history, plots_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves — PMSM FDD CNN-BiLSTM", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(history.history["loss"],     label="Train Loss", color="#0077b6")
    ax.plot(history.history["val_loss"], label="Val Loss",   color="#e63946", linestyle="--")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Categorical Cross-Entropy")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history.history["accuracy"],     label="Train Acc", color="#2d9057")
    ax.plot(history.history["val_accuracy"], label="Val Acc",   color="#e63946", linestyle="--")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(plots_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Train] Saved training curves → {out}")


def train_model(model: keras.Model, branch_idxs,
                data: dict, cfg: dict) -> keras.callbacks.History:
    tc = cfg["training"]
    pp = cfg["paths"]
    Path(pp["metrics_dir"]).mkdir(parents=True, exist_ok=True)
    Path(pp["model_path"]).parent.mkdir(parents=True, exist_ok=True)

    X_tr = _prepare_inputs(data["X_train"], branch_idxs)
    X_v  = _prepare_inputs(data["X_val"],   branch_idxs)

    y_tr = keras.utils.to_categorical(data["y_train"], tc.get("num_classes",
                                       cfg["model"]["num_classes"]))
    y_v  = keras.utils.to_categorical(data["y_val"],   tc.get("num_classes",
                                       cfg["model"]["num_classes"]))

    class_weights = (compute_weights(data["y_train"])
                     if tc.get("class_weights", True) else None)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tc["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_v, y_v),
        epochs=tc["epochs"],
        batch_size=tc["batch_size"],
        class_weight=class_weights,
        callbacks=build_callbacks(cfg),
        verbose=1,
    )

    plot_training_curves(history, pp["plots_dir"])

    # Overfitting check
    final_train = history.history["accuracy"][-1]
    final_val   = history.history["val_accuracy"][-1]
    gap = final_train - final_val
    print(f"\n[Overfitting Check]  train_acc={final_train:.4f}  val_acc={final_val:.4f}  gap={gap:.4f}")
    if gap > 0.05:
        print("  ⚠ Gap > 5% — possible overfitting. Consider more dropout or regularization.")
    else:
        print("  ✓ Generalisation gap is acceptable.")

    return history
