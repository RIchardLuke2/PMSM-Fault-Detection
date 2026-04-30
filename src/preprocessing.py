"""
Leak-free preprocessing pipeline.

Order:
  1. Split raw rows: train / val / test  (NO windowing yet)
  2. Fit scaler on X_train ONLY
  3. Scale val + test
  4. Apply sliding window per split

Main fix:
  - window labels use the CENTER sample by default instead of majority vote
  - raw and windowed class distributions are printed so imbalance is visible
  - one shared split helper keeps scaler fitting and pipeline data aligned
"""

import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ----------------------------
# Small utilities
# ----------------------------

def _get_paths(cfg: dict) -> Tuple[str, str]:
    pp = cfg.get("paths", {})
    plots_dir = pp.get("plots_dir", "outputs/plots")
    metrics_dir = pp.get("metrics_dir", "outputs/metrics")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    return plots_dir, metrics_dir


def show_class_distribution_df(df: pd.DataFrame, target_col: str, title: str = "Class distribution") -> None:
    counts = df[target_col].value_counts().sort_index()
    perc = df[target_col].value_counts(normalize=True).sort_index() * 100

    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    for cls in counts.index:
        print(f"{cls}: {counts[cls]} samples ({perc[cls]:.2f}%)")
    print("=" * 60 + "\n")


def show_encoded_distribution(y: np.ndarray, encoder: LabelEncoder, title: str = "Window distribution") -> None:
    if len(y) == 0:
        print(f"[Preprocess] {title}: empty")
        return

    unique, counts = np.unique(y, return_counts=True)
    names = encoder.inverse_transform(unique)

    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    for name, c in zip(names, counts):
        print(f"{name}: {int(c)} samples")
    print("=" * 60 + "\n")


def plot_class_distribution_df(df: pd.DataFrame, target_col: str, out_path: str, title: str = "Class distribution") -> None:
    counts = df[target_col].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------
# Load / clean
# ----------------------------

def load_and_clean(cfg: dict) -> pd.DataFrame:
    raw = cfg["data"]["raw_path"]
    feat = cfg["data"]["features"]
    tgt = cfg["data"]["target"]
    plots_dir, _ = _get_paths(cfg)

    if not os.path.exists(raw):
        print("[Preprocess] CSV not found — generating synthetic data...")
        from utils.data_gen import generate
        df = generate(n_per_class=1000, seed=cfg.get("seed", 42))
        Path(raw).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw, index=False)
    else:
        df = pd.read_csv(raw)

    missing = [c for c in feat + [tgt] if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + str(missing))

    n_nan = int(df[feat].isnull().sum().sum())
    if n_nan:
        print(f"[Preprocess] Filling {n_nan} NaN with median")
        df[feat] = df[feat].fillna(df[feat].median())

    for col in feat:
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        iqr = p99 - p1
        df[col] = df[col].clip(p1 - 3 * iqr, p99 + 3 * iqr)

    print(f"[Preprocess] {len(df)} rows, {df[tgt].nunique()} classes")
    show_class_distribution_df(df, tgt, title="Raw class distribution")
    plot_class_distribution_df(
        df,
        tgt,
        os.path.join(plots_dir, "class_distribution_raw.png"),
        title="Raw class distribution",
    )
    return df


# ----------------------------
# Labels
# ----------------------------

def encode_labels(df: pd.DataFrame, cfg: dict) -> LabelEncoder:
    enc = LabelEncoder()
    enc.fit(sorted(df[cfg["data"]["target"]].unique()))
    print("[Preprocess] Classes:", list(enc.classes_))
    return enc


# ----------------------------
# Shared split helper
# ----------------------------

def _split_train_val_test(X: np.ndarray, y: np.ndarray, cfg: dict):
    seed = cfg.get("seed", 42)
    ts = cfg["data"]["test_size"]
    vs = cfg["data"]["val_size"]

    # First split off test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y,
        test_size=ts,
        random_state=seed,
        stratify=y
    )

    # Then split tmp into train and val
    val_frac = vs / (1.0 - ts)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=val_frac,
        random_state=seed,
        stratify=y_tmp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------
# Scaler
# ----------------------------

def fit_scaler(df: pd.DataFrame, cfg: dict) -> StandardScaler:
    feat = cfg["data"]["features"]
    tgt = cfg["data"]["target"]

    X = df[feat].values
    y = df[tgt].values

    Xtr, _, _, _, _, _ = _split_train_val_test(X, y, cfg)

    sc = StandardScaler()
    sc.fit(Xtr)
    print(f"[Preprocess] Scaler fitted on {len(Xtr)} rows")
    return sc


# ----------------------------
# Windowing
# ----------------------------

def _window_label(y_window: np.ndarray, label_strategy: str = "center") -> int:
    """
    Choose the label for one window.

    Strategies:
      - center: label at the center timestep (default, best starting point)
      - last:   label at the last timestep
      - majority: majority vote across the window
    """
    label_strategy = (label_strategy or "center").lower()

    if len(y_window) == 0:
        raise ValueError("Empty window received in _window_label")

    if label_strategy == "center":
        return int(y_window[len(y_window) // 2])
    if label_strategy == "last":
        return int(y_window[-1])
    if label_strategy == "majority":
        y_int = y_window.astype(int)
        return int(np.bincount(y_int).argmax())

    raise ValueError(f"Unknown label_strategy: {label_strategy}")


def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    stride: int,
    label_strategy: str = "center",
):
    if len(X) < window_size:
        return (
            np.empty((0, window_size, X.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    wins, labs = [], []
    for s in range(0, len(X) - window_size + 1, stride):
        xw = X[s:s + window_size]
        yw = y[s:s + window_size]
        wins.append(xw)
        labs.append(_window_label(yw, label_strategy=label_strategy))

    return np.array(wins, dtype=np.float32), np.array(labs, dtype=np.int64)


# ----------------------------
# Main pipeline
# ----------------------------

def build_pipeline_data(df, encoder, scaler, cfg) -> dict:
    feat = cfg["data"]["features"]
    tgt = cfg["data"]["target"]
    ws = cfg["windowing"]["window_size"]
    st = cfg["windowing"]["stride"]
    label_strategy = cfg.get("windowing", {}).get("label_strategy", "center")

    X = df[feat].values.astype(np.float64)
    y = encoder.transform(df[tgt].values).astype(np.int64)

    Xtr, Xv, Xte, ytr, yv, yte = _split_train_val_test(X, y, cfg)

    # Show class balance before windowing
    show_encoded_distribution(ytr, encoder, title="Train split before windowing")
    show_encoded_distribution(yv, encoder, title="Validation split before windowing")
    show_encoded_distribution(yte, encoder, title="Test split before windowing")

    # Scale using train-only scaler
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xv_s = scaler.transform(Xv).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    # Window each split
    Xtrw, ytrw = make_windows(Xtr_s, ytr, ws, st, label_strategy=label_strategy)
    Xvw, yvw = make_windows(Xv_s, yv, ws, st, label_strategy=label_strategy)
    Xtew, ytew = make_windows(Xte_s, yte, ws, st, label_strategy=label_strategy)

    print(f"[Preprocess] train={Xtrw.shape}  val={Xvw.shape}  test={Xtew.shape}")

    # Show class balance after windowing
    show_encoded_distribution(ytrw, encoder, title=f"Train windows ({label_strategy})")
    show_encoded_distribution(yvw, encoder, title=f"Validation windows ({label_strategy})")
    show_encoded_distribution(ytew, encoder, title=f"Test windows ({label_strategy})")

    # Optional warning if a split ends up empty
    if len(Xtrw) == 0 or len(Xvw) == 0 or len(Xtew) == 0:
        raise ValueError(
            f"One of the splits became empty after windowing. "
            f"Check window_size={ws}, stride={st}, and label_strategy={label_strategy}."
        )

    return {
        "X_train": Xtrw, "y_train": ytrw,
        "X_val": Xvw, "y_val": yvw,
        "X_test": Xtew, "y_test": ytew,
        "X_train_flat": Xtrw.reshape(len(Xtrw), -1),
        "X_val_flat": Xvw.reshape(len(Xvw), -1),
        "X_test_flat": Xtew.reshape(len(Xtew), -1),
        "X_test_raw_scaled": Xte_s,
        "y_test_raw": yte,
    }