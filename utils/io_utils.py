"""Config loading, artifact saving/loading helpers."""
import os, yaml, joblib
from pathlib import Path


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def save_artifact(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    print(f"[IO] Saved → {path}")


def load_artifact(path: str):
    obj = joblib.load(path)
    print(f"[IO] Loaded ← {path}")
    return obj
