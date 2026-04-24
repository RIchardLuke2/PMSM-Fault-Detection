"""
run.py — End-to-end PMSM Inverter Fault Detection Pipeline
===========================================================
Usage:
    python run.py                        # full pipeline
    python run.py --skip-baselines       # skip baseline comparison
    python run.py --skip-robustness      # skip robustness tests
    python run.py --skip-explain         # skip explainability
    python run.py --data-only            # only generate + preprocess data
"""
import argparse
import os
import sys
import time

# ── Make src/ and utils/ importable from project root ─────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from utils.seed     import set_global_seed
from utils.io_utils import load_config, save_config
from utils.data_gen import generate_dataset


def parse_args():
    p = argparse.ArgumentParser(description="PMSM FDD Pipeline")
    p.add_argument("--config",           default="configs/config.yaml")
    p.add_argument("--skip-baselines",   action="store_true")
    p.add_argument("--skip-robustness",  action="store_true")
    p.add_argument("--skip-explain",     action="store_true")
    p.add_argument("--data-only",        action="store_true")
    p.add_argument("--generate-data",    action="store_true",
                   help="Force re-generate synthetic dataset")
    return p.parse_args()


def banner(msg: str):
    w = 70
    print("\n" + "═" * w)
    print(f"  {msg}")
    print("═" * w)


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # ── Reproducibility ───────────────────────────────────────────────────────
    set_global_seed(cfg["data"]["random_seed"])

    t_total = time.time()

    # ── 0. Data generation ────────────────────────────────────────────────────
    banner("STEP 0 — Data Generation")
    raw_path = cfg["data"]["raw_path"]
    if args.generate_data or not os.path.exists(raw_path):
        generate_dataset(out_path=raw_path, seed=cfg["data"]["random_seed"])
    else:
        print(f"[Run] Found existing dataset at {raw_path} — skipping generation.")
        print("      Pass --generate-data to force regeneration.")

    if args.data_only:
        print("[Run] --data-only flag set. Exiting after data generation.")
        return

    # ── 1. Preprocessing ──────────────────────────────────────────────────────
    banner("STEP 1 — Preprocessing (leak-free)")
    from src.preprocessing import run_preprocessing
    data = run_preprocessing(cfg)

    # Stash raw-scaled test split for window ablation
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from utils.io_utils import load_artifact
    le: LabelEncoder = data["encoder"]

    # ── 2. Model building ─────────────────────────────────────────────────────
    banner("STEP 2 — Building CNN-BiLSTM Model")
    import tensorflow as tf
    from src.model import build_cnn_bilstm
    set_global_seed(cfg["training"]["seed"])  # re-seed before model init
    model, branch_idxs = build_cnn_bilstm(cfg)
    model.summary()

    # ── 3. Training ───────────────────────────────────────────────────────────
    banner("STEP 3 — Training")
    from src.training import train_model
    t0 = time.time()
    history = train_model(model, branch_idxs, data, cfg)
    print(f"[Run] Training finished in {(time.time()-t0)/60:.1f} min")

    # Reload best checkpoint
    best_model_path = cfg["paths"]["model_path"]
    if os.path.exists(best_model_path):
        print(f"[Run] Loading best checkpoint from {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)

    # ── 4. Evaluation ─────────────────────────────────────────────────────────
    banner("STEP 4 — Evaluation on Held-Out Test Set")
    from src.evaluation import evaluate_model
    metrics = evaluate_model(model, branch_idxs, data, cfg, le, tag="test")

    # Save scalar metrics summary
    import json, pathlib
    pathlib.Path(cfg["paths"]["metrics_dir"]).mkdir(parents=True, exist_ok=True)
    summary_path = os.path.join(cfg["paths"]["metrics_dir"], "test_metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Run] Metrics summary → {summary_path}")

    # ── 5. Baselines ──────────────────────────────────────────────────────────
    if not args.skip_baselines:
        banner("STEP 5 — Baseline Model Comparison")
        from src.baselines import run_baselines
        baseline_df = run_baselines(data, cfg)
        # Append main model result for fair comparison
        import pandas as pd
        main_row = pd.DataFrame([{
            "model":        "CNN-BiLSTM (ours)",
            "accuracy":     metrics["accuracy"],
            "f1_macro":     metrics["f1"],
            "train_time_s": round(time.time() - t0, 1),
        }])
        full_df = pd.concat([baseline_df, main_row], ignore_index=True)
        full_df.to_csv(
            os.path.join(cfg["paths"]["metrics_dir"], "full_comparison.csv"),
            index=False)
        print("\n[Run] Full model comparison:")
        print(full_df.to_string(index=False))
    else:
        print("[Run] Skipping baselines (--skip-baselines).")

    # ── 6. Robustness ─────────────────────────────────────────────────────────
    if not args.skip_robustness:
        banner("STEP 6 — Robustness Testing")
        from src.robustness import noise_robustness, window_ablation
        noise_robustness(model, branch_idxs, data, cfg)
        window_ablation(model, branch_idxs, data, cfg)
    else:
        print("[Run] Skipping robustness (--skip-robustness).")

    # ── 7. Explainability ─────────────────────────────────────────────────────
    if not args.skip_explain:
        banner("STEP 7 — Explainability")
        from src.explainability import permutation_importance, compute_saliency
        perm_df = permutation_importance(
            model, branch_idxs,
            data["X_test"], data["y_test"],
            cfg["data"]["features"], cfg, n_repeats=3
        )
        # Saliency on first test sample
        compute_saliency(model, data["X_test"][0], branch_idxs,
                         int(data["y_test"][0]), cfg)
    else:
        print("[Run] Skipping explainability (--skip-explain).")

    # ── 8. Save run config ────────────────────────────────────────────────────
    save_config(cfg, cfg["paths"]["config_path"])

    banner(f"PIPELINE COMPLETE — Total time: {(time.time()-t_total)/60:.1f} min")
    print(f"  Model   → {cfg['paths']['model_path']}")
    print(f"  Metrics → {cfg['paths']['metrics_dir']}/")
    print(f"  Plots   → {cfg['paths']['plots_dir']}/\n")


if __name__ == "__main__":
    main()
