"""
PyTorch training loop.
- nn.CrossEntropyLoss (raw logits + class weights)
- Adam with weight_decay (L2 regularization)
- ReduceLROnPlateau scheduler
- Manual EarlyStopping with best-weight restore
- torch.save checkpoint (ModelCheckpoint equivalent)
"""
import os, time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print("[Train] GPU:", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        d = torch.device("mps")
        print("[Train] Apple MPS")
    else:
        d = torch.device("cpu")
        print("[Train] CPU")
    return d


def make_loader(X: np.ndarray, y: np.ndarray,
                branch_idxs: Optional[list],
                batch_size: int, shuffle: bool,
                device: torch.device) -> DataLoader:
    y_t = torch.tensor(y, dtype=torch.long)
    if branch_idxs is None:
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), y_t)
    else:
        bs = [torch.tensor(X[:, :, idx], dtype=torch.float32)
              for idx in branch_idxs]
        ds = TensorDataset(*bs, y_t)
    pin = (device.type == "cuda")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=pin)


def get_class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    cls = np.unique(y)
    w   = compute_class_weight("balanced", classes=cls, y=y)
    return torch.tensor(w, dtype=torch.float32).to(device)


class EarlyStopping:
    def __init__(self, patience: int = 12, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print("[Train] Best weights restored (val_loss={:.4f})".format(
                  self.best_loss))


def _run_epoch(model, loader, criterion, optimizer,
               device, branch_idxs, train: bool):
    model.train() if train else model.eval()
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            yb = batch[-1].to(device)
            if branch_idxs is None:
                logits = model(batch[0].to(device))
            else:
                logits = model([b.to(device) for b in batch[:-1]])
            loss = criterion(logits, yb)
            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * yb.size(0)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += yb.size(0)
    return total_loss / total, correct / total


def _plot_curves(history: dict, plots_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves — PMSM CNN-BiLSTM", fontsize=13, fontweight="bold")
    for ax, (tk, vk, title), (ct, cv) in zip(
            axes,
            [("train_loss","val_loss","Loss"),("train_acc","val_acc","Accuracy")],
            [("#0077b6","#e63946"),("#2d9057","#e63946")]):
        ax.plot(history[tk], label="Train", color=ct, lw=2)
        ax.plot(history[vk], label="Val",   color=cv, lw=2, linestyle="--")
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(plots_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print("[Train] Curves saved ->", out)


def train_model(model: nn.Module, branch_idxs: Optional[list],
                data: dict, cfg: dict) -> dict:
    tc = cfg["training"]
    pp = cfg["paths"]
    for p in [pp["plots_dir"], pp["metrics_dir"]]:
        Path(p).mkdir(parents=True, exist_ok=True)
    Path(pp["model_path"]).parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model  = model.to(device)
    cw     = get_class_weights(data["y_train"], device) if tc.get("class_weights") else None
    crit   = nn.CrossEntropyLoss(weight=cw)
    opt    = torch.optim.Adam(model.parameters(),
                              lr=tc["learning_rate"],
                              weight_decay=float(cfg["model"].get("l2_reg", 1e-4)))
    sched  = ReduceLROnPlateau(opt, mode="min",
                               factor=tc["reduce_lr_factor"],
                               patience=tc["reduce_lr_patience"],
                               min_lr=1e-6)
    es     = EarlyStopping(patience=tc["early_stopping_patience"])
    bs     = tc["batch_size"]
    tr_ld  = make_loader(data["X_train"], data["y_train"],
                         branch_idxs, bs, True,  device)
    vl_ld  = make_loader(data["X_val"],   data["y_val"],
                         branch_idxs, bs, False, device)

    hist   = {"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}
    best   = 0.0
    t0     = time.time()

    print("[Train] train={:,}  val={:,}  epochs={}  batch={}  lr={}".format(
          len(data["y_train"]), len(data["y_val"]),
          tc["epochs"], bs, tc["learning_rate"]))

    for ep in range(1, tc["epochs"] + 1):
        tl, ta = _run_epoch(model, tr_ld, crit, opt,   device, branch_idxs, True)
        vl, va = _run_epoch(model, vl_ld, crit, None,  device, branch_idxs, False)
        sched.step(vl)
        hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        hist["train_acc"].append(ta);  hist["val_acc"].append(va)

        if va > best:
            best = va
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "val_acc": va, "val_loss": vl, "cfg": cfg},
                       pp["model_path"])

        lr_now = opt.param_groups[0]["lr"]
        print("  Ep {:3d}/{} | loss={:.4f}  val_loss={:.4f} | "
              "acc={:.4f}  val_acc={:.4f} | lr={:.2e}".format(
              ep, tc["epochs"], tl, vl, ta, va, lr_now))

        if es.step(vl, model):
            print("[Train] Early stop at epoch", ep); break

    es.restore(model)
    elapsed = time.time() - t0
    gap = hist["train_acc"][-1] - hist["val_acc"][-1]
    flag = "WARNING: gap>5% — increase dropout/weight_decay" if gap > 0.05 else "OK"
    print("[Train] Done in {:.1f} min | best_val_acc={:.4f}".format(elapsed/60, best))
    print("[Overfit] gap={:.4f}  {}".format(gap, flag))

    _plot_curves(hist, pp["plots_dir"])
    pd.DataFrame(hist).to_csv(
        os.path.join(pp["metrics_dir"], "training_log.csv"), index_label="epoch")
    return hist


def load_checkpoint(model: nn.Module, path: str,
                    device: Optional[torch.device] = None) -> nn.Module:
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print("[Train] Loaded ckpt  ep={}  val_acc={:.4f}".format(
          ckpt["epoch"], ckpt["val_acc"]))
    return model