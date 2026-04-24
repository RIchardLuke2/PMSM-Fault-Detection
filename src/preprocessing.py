"""
Leak-free preprocessing pipeline for PMSM FDD.

Key design decisions
--------------------
1. Train/val/test split happens BEFORE any fitting (scaler, encoder).
2. Sliding windows are created AFTER the split so no window straddles
   the boundary, eliminating temporal leakage.
3. Scaler and LabelEncoder are fit on training data only, then applied
   to val/test.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from utils.io_utils import save_artifact, load_config


_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


# ── Outlier removal ──────────────────────────────────────────────────────────

def remove_outliers(df: pd.DataFrame, features: list, method: str = "iqr",
                    threshold: float = 3.0) -> pd.DataFrame:
    """Replace outliers with NaN; they are later imputed by forward-fill."""
    df = df.copy()
    if method == "iqr":
        for col in features:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - threshold * IQR, Q3 + threshold * IQR
            mask = (df[col] < lo) | (df[col] > hi)
            df.loc[mask, col] = np.nan
    elif method == "zscore":
        for col in features:
            z = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
            df.loc[z.abs() > threshold, col] = np.nan
    return df


def impute(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Forward-fill then backward-fill; remaining NaNs → column median."""
    df = df.copy()
    df[features] = df[features].ffill().bfill()
    for col in features:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df


# ── Splitting ─────────────────────────────────────────────────────────────────

def split_by_class(df: pd.DataFrame, target: str, test_size: float,
                   val_size: float, seed: int):
    """
    Stratified split that preserves class balance across all three sets.
    Split order: full → trainval + test → train + val.
    """
    trainval, test = train_test_split(df, test_size=test_size,
                                      stratify=df[target], random_state=seed)
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(trainval, test_size=relative_val,
                                  stratify=trainval[target], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ── Windowing ─────────────────────────────────────────────────────────────────

def make_windows(X: np.ndarray, y: np.ndarray, window_size: int,
                 stride: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows.  A window's label is the MAJORITY class
    within that window (handles boundary ambiguity cleanly).

    Returns
    -------
    X_win : (N, window_size, n_features)
    y_win : (N,)  integer labels
    """
    n_samples, n_features = X.shape
    starts = range(0, n_samples - window_size + 1, stride)
    X_win, y_win = [], []
    for s in starts:
        e = s + window_size
        X_win.append(X[s:e])
        # majority-vote label inside window
        labels, counts = np.unique(y[s:e], return_counts=True)
        y_win.append(labels[counts.argmax()])
    return np.array(X_win, dtype=np.float32), np.array(y_win, dtype=np.int32)


def flatten_windows(X_win: np.ndarray) -> np.ndarray:
    """Flatten (N, W, F) → (N, W*F) for sklearn baseline models."""
    return X_win.reshape(X_win.shape[0], -1)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing(cfg: dict) -> dict:
    dc = cfg["data"]
    pc = cfg["preprocessing"]
    wc = cfg["windowing"]

    # 1. Load raw data
    df = pd.read_csv(dc["raw_path"])
    print(f"[Preprocess] Loaded {len(df):,} rows, {df.shape[1]} cols")
    print(f"[Preprocess] Missing before clean: {df[dc['features']].isna().sum().sum()}")

    # 2. Outlier → NaN
    df = remove_outliers(df, dc["features"], method=pc["outlier_method"],
                          threshold=pc["outlier_threshold"])

    # 3. Impute
    df = impute(df, dc["features"])
    print(f"[Preprocess] Missing after impute: {df[dc['features']].isna().sum().sum()}")

    # 4. Label encode (fit only after we have the full label set)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df[dc["target"]])
    print(f"[Preprocess] Classes: {list(le.classes_)}")

    # 5. Stratified split (raw rows, BEFORE windowing)
    train_df, val_df, test_df = split_by_class(
        df, "label", dc["test_size"], dc["val_size"], dc["random_seed"])
    print(f"[Preprocess] Split → train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # 6. Fit scaler on TRAIN only
    scaler_cls = _SCALERS[pc["scaler"]]
    scaler = scaler_cls()
    X_train_raw = scaler.fit_transform(train_df[dc["features"]].values)
    X_val_raw   = scaler.transform(val_df[dc["features"]].values)
    X_test_raw  = scaler.transform(test_df[dc["features"]].values)

    y_train_raw = train_df["label"].values
    y_val_raw   = val_df["label"].values
    y_test_raw  = test_df["label"].values

    # 7. Sliding windows (per-split, no leakage across split boundaries)
    ws, st = wc["window_size"], wc["stride"]
    X_train, y_train = make_windows(X_train_raw, y_train_raw, ws, st)
    X_val,   y_val   = make_windows(X_val_raw,   y_val_raw,   ws, st)
    X_test,  y_test  = make_windows(X_test_raw,  y_test_raw,  ws, st)

    print(f"[Preprocess] Window shapes  → train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    # 8. Save artefacts
    out = cfg["paths"]
    save_artifact(scaler, out["scaler_path"])
    save_artifact(le,     out["encoder_path"])

    processed = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler,  "encoder": le,
        "features": dc["features"],
    }

    # Also save flattened versions for baseline models
    if wc.get("flatten_for_ml", True):
        processed["X_train_flat"] = flatten_windows(X_train)
        processed["X_val_flat"]   = flatten_windows(X_val)
        processed["X_test_flat"]  = flatten_windows(X_test)

    return processed


if __name__ == "__main__":
    cfg = load_config()
    run_preprocessing(cfg)
