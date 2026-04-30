"""
End-to-end pipeline:
  python run.py                        # full pipeline
  python run.py --skip-baselines       # skip slow baselines
  python run.py --skip-robustness      # skip robustness tests
  python run.py --inference-only       # load saved model and run inference
"""
import argparse
import os
import sys

# Ensure project root is in path (works when called from any cwd)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from utils.seed     import set_all_seeds
from utils.io_utils import load_config, save_config, save_artifact
from src.preprocessing import load_and_clean, encode_labels, fit_scaler, build_pipeline_data
from src.model         import build_cnn_bilstm
from src.training      import train_model, load_checkpoint, get_device
from src.evaluation    import evaluate_model
from src.baselines     import run_baselines
from src.robustness    import noise_robustness, window_ablation
from src.explainability import permutation_importance, compute_saliency


def parse_args():
    p = argparse.ArgumentParser(description="PMSM FDD Pipeline")
    p.add_argument("--config",            default="configs/config.yaml")
    p.add_argument("--skip-baselines",    action="store_true")
    p.add_argument("--skip-robustness",   action="store_true")
    p.add_argument("--skip-explainability", action="store_true")
    p.add_argument("--inference-only",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    set_all_seeds(cfg.get("seed", 42))

    # ── 1. Preprocessing ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 1: Preprocessing")
    print("="*60)
    df      = load_and_clean(cfg)
    encoder = encode_labels(df, cfg)
    scaler  = fit_scaler(df, cfg)
    data    = build_pipeline_data(df, encoder, scaler, cfg)

    # Save artifacts
    save_artifact(scaler,  cfg["paths"]["scaler_path"])
    save_artifact(encoder, cfg["paths"]["encoder_path"])
    save_config(cfg,       cfg["paths"]["config_save"])

    if args.inference_only:
        print("\n[Run] Inference-only mode — loading saved model ...")
        model, branch_idxs = build_cnn_bilstm(cfg)
        device = get_device()
        model  = load_checkpoint(model, cfg["paths"]["model_path"], device)
    else:
        # ── 2. Training ───────────────────────────────────────────────────────
        print("\n" + "="*60)
        print("  STEP 2: Training CNN-BiLSTM")
        print("="*60)
        model, branch_idxs = build_cnn_bilstm(cfg)
        history = train_model(model, branch_idxs, data, cfg)
        device  = get_device()
        model   = load_checkpoint(model, cfg["paths"]["model_path"], device)

    # ── 3. Evaluation ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3: Evaluation")
    print("="*60)
    val_metrics  = evaluate_model(model, branch_idxs, data, cfg, encoder,
                                  device=None, tag="val")
    test_metrics = evaluate_model(model, branch_idxs, data, cfg, encoder,
                                  device=None, tag="test")

    import pandas as pd
    combined = pd.DataFrame([
        {"split": "val",  **val_metrics},
        {"split": "test", **test_metrics},
    ])
    combined.to_csv(
        os.path.join(cfg["paths"]["metrics_dir"], "final_metrics.csv"), index=False)
    print("\n[Run] Final metrics:")
    print(combined.to_string(index=False))

    # ── 4. Baselines ──────────────────────────────────────────────────────────
    if not args.skip_baselines:
        print("\n" + "="*60)
        print("  STEP 4: Baselines")
        print("="*60)
        bl_df = run_baselines(data, cfg)
        main_row = pd.DataFrame([{
            "model": "CNN-BiLSTM (ours)",
            "accuracy":    round(test_metrics["accuracy"], 4),
            "f1_macro":    round(test_metrics["f1"], 4),
            "train_time_s": "—",
        }])
        full_table = pd.concat([bl_df, main_row], ignore_index=True)
        full_table.to_csv(
            os.path.join(cfg["paths"]["metrics_dir"], "all_models_comparison.csv"),
            index=False)
        print("\n[Baselines] Full comparison:")
        print(full_table.to_string(index=False))

    # ── 5. Robustness ─────────────────────────────────────────────────────────
    if not args.skip_robustness:
        print("\n" + "="*60)
        print("  STEP 5: Robustness")
        print("="*60)
        noise_robustness(model, branch_idxs, data, cfg)
        window_ablation(model,  branch_idxs, data, cfg)

    # ── 6. Explainability ─────────────────────────────────────────────────────
    if not args.skip_explainability:
        print("\n" + "="*60)
        print("  STEP 6: Explainability")
        print("="*60)
        permutation_importance(
            model, branch_idxs,
            data["X_test"], data["y_test"],
            cfg["data"]["features"], cfg)


        for cls_idx in range(cfg["model"]["num_classes"]):
            mask = (data["y_test"] == cls_idx)
            if not mask.any():
                continue
            sample = data["X_test"][np.where(mask)[0][0]]
            compute_saliency(model, sample, branch_idxs, cls_idx, cfg)
            break

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("  Outputs in: outputs/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()