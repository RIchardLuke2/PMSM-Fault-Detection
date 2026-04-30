"""Config loading, artifact saving/loading helpers."""
import joblib, yaml
from pathlib import Path

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)

def save_config(cfg: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False)
    print("[IO] Config saved ->", path)

def save_artifact(obj: object, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    print("[IO] Saved ->", path)

def load_artifact(path: str) -> object:
    obj = joblib.load(path)
    print("[IO] Loaded <-", path)
    return obj