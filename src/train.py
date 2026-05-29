"""
Bird Language ML - Training Pipeline
Full training loop with:
  - Mixed precision (AMP)
  - Cosine annealing with linear warmup
  - Differential learning rates (backbone vs heads)
  - Early stopping + best model checkpointing
  - TensorBoard / CSV logging
  - Gradient clipping
"""

import os
import time
import json
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TrainingConfig, ModelConfig, SPECIES_LIST, CALL_TYPES, audio_cfg, aug_cfg
from src.model import BirdLanguageModel, BirdLanguageLoss, build_model
from src.dataset import build_dataloaders


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler with Warmup
# ──────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """
    Linear warmup for warmup_epochs, then cosine annealing.
    Compatible with any optimizer.
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self._step = 0

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr * pg.get("lr_scale", 1.0)
        return lr


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._losses = []
        self._sp_preds, self._sp_labels = [], []
        self._ct_preds, self._ct_labels = [], []

    def update(self, loss: float, sp_preds, sp_labels, ct_preds, ct_labels):
        self._losses.append(loss)
        self._sp_preds.extend(sp_preds.cpu().numpy())
        self._sp_labels.extend(sp_labels.cpu().numpy())
        self._ct_preds.extend(ct_preds.cpu().numpy())
        self._ct_labels.extend(ct_labels.cpu().numpy())

    def compute(self) -> Dict:
        sp_acc = accuracy_score(self._sp_labels, self._sp_preds)
        ct_acc = accuracy_score(self._ct_labels, self._ct_preds)
        sp_f1 = f1_score(self._sp_labels, self._sp_preds, average="macro", zero_division=0)
        ct_f1 = f1_score(self._ct_labels, self._ct_preds, average="macro", zero_division=0)
        return {
            "loss": np.mean(self._losses),
            "species_acc": sp_acc,
            "call_acc": ct_acc,
            "species_f1": sp_f1,
            "call_f1": ct_f1,
            "mean_acc": (sp_acc + ct_acc) / 2,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Training Step
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: BirdLanguageModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: BirdLanguageLoss,
    scaler: GradScaler,
    device: torch.device,
    cfg: TrainingConfig,
) -> Dict:
    model.train()
    tracker = MetricsTracker()
    total_steps = len(loader)

    for step, batch in enumerate(loader):
        specs = batch["spectrogram"].to(device, non_blocking=True)

        with autocast(enabled=cfg.use_amp):
            outputs = model(specs)
            losses = criterion(outputs, batch)

        scaler.scale(losses["total_loss"]).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Compute predictions for metrics (skip if mixup)
        if not batch.get("mixed", False):
            sp_preds = outputs["species_logits"].argmax(dim=1)
            ct_preds = outputs["call_logits"].argmax(dim=1)
            tracker.update(
                losses["total_loss"].item(),
                sp_preds.detach(), batch["species_label"].to(device),
                ct_preds.detach(), batch["call_type_label"].to(device),
            )

        if (step + 1) % 20 == 0:
            metrics = tracker.compute()
            print(f"  [{step+1}/{total_steps}] "
                  f"loss={metrics['loss']:.4f} | "
                  f"sp_acc={metrics['species_acc']:.3f} | "
                  f"ct_acc={metrics['call_acc']:.3f}")

    return tracker.compute()


@torch.no_grad()
def eval_epoch(
    model: BirdLanguageModel,
    loader: DataLoader,
    criterion: BirdLanguageLoss,
    device: torch.device,
) -> Dict:
    model.eval()
    tracker = MetricsTracker()

    for batch in loader:
        specs = batch["spectrogram"].to(device, non_blocking=True)
        sp_labels = batch["species_label"].to(device)
        ct_labels = batch["call_type_label"].to(device)

        outputs = model(specs)
        losses = criterion(outputs, batch)

        sp_preds = outputs["species_logits"].argmax(dim=1)
        ct_preds = outputs["call_logits"].argmax(dim=1)
        tracker.update(losses["total_loss"].item(), sp_preds, sp_labels, ct_preds, ct_labels)

    return tracker.compute()


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint Utilities
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: BirdLanguageModel,
    optimizer,
    epoch: int,
    metrics: Dict,
    cfg: TrainingConfig,
    filename: str = "best_model.pth",
):
    Path(cfg.checkpoint_dir).mkdir(exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": metrics["mean_acc"],
        "val_species_f1": metrics["species_f1"],
        "val_call_f1": metrics["call_f1"],
        "config": {
            "species_classes": len(SPECIES_LIST),
            "call_type_classes": len(CALL_TYPES),
        },
    }, os.path.join(cfg.checkpoint_dir, filename))


# ──────────────────────────────────────────────────────────────────────────────
# Plot Training Curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_curves(history: Dict, save_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Bird Language ML - Training Curves", fontsize=14)

    metrics = [
        ("loss", "Loss"),
        ("species_acc", "Species Accuracy"),
        ("call_acc", "Call Type Accuracy"),
        ("species_f1", "Species Macro F1"),
    ]
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.plot(history["train_" + key], label="Train", linewidth=2)
        ax.plot(history["val_" + key], label="Val", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainingConfig = TrainingConfig()):
    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg.metadata_csv, audio_cfg, aug_cfg, cfg, SPECIES_LIST, CALL_TYPES
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        backbone="efficientnet_b0",
        pretrained=True,
        species_classes=len(SPECIES_LIST),
        call_type_classes=len(CALL_TYPES),
    ).to(device)

    # ── Optimizer: different LR for backbone vs heads ─────────────────────────
    optimizer = AdamW([
        {"params": model.get_backbone_params(), "lr": cfg.learning_rate * 0.1, "lr_scale": 0.1},
        {"params": model.get_head_params(),    "lr": cfg.learning_rate,       "lr_scale": 1.0},
    ], weight_decay=cfg.weight_decay)

    scheduler = WarmupCosineScheduler(optimizer, cfg.warmup_epochs, cfg.epochs, cfg.learning_rate)
    criterion = BirdLanguageLoss(
        species_weight=cfg.species_loss_weight,
        call_weight=cfg.call_type_loss_weight,
        label_smoothing=cfg.label_smoothing,
        num_species=len(SPECIES_LIST),
        num_call_types=len(CALL_TYPES),
    )
    scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    # ── Logging ───────────────────────────────────────────────────────────────
    Path(cfg.log_dir).mkdir(exist_ok=True)
    Path(cfg.checkpoint_dir).mkdir(exist_ok=True)
    log_path = os.path.join(cfg.log_dir, "training_log.csv")
    log_fields = ["epoch", "lr", "train_loss", "train_species_acc", "train_call_acc",
                  "train_species_f1", "val_loss", "val_species_acc", "val_call_acc", "val_species_f1"]

    history = {f"train_{m}": [] for m in ["loss", "species_acc", "call_acc", "species_f1"]}
    history.update({f"val_{m}": [] for m in ["loss", "species_acc", "call_acc", "species_f1"]})

    best_val_acc = 0.0
    patience_counter = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(cfg.epochs):
        lr = scheduler.step(epoch)
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{cfg.epochs}  |  LR: {lr:.6f}")
        print(f"{'='*60}")

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler, device, cfg)
        val_metrics = eval_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch+1} Summary ({elapsed:.1f}s):")
        print(f"  Train → loss={train_metrics['loss']:.4f} | "
              f"sp_acc={train_metrics['species_acc']:.3f} | ct_acc={train_metrics['call_acc']:.3f}")
        print(f"  Val   → loss={val_metrics['loss']:.4f} | "
              f"sp_acc={val_metrics['species_acc']:.3f} | ct_acc={val_metrics['call_acc']:.3f} | "
              f"sp_f1={val_metrics['species_f1']:.3f}")

        # Log
        for m in ["loss", "species_acc", "call_acc", "species_f1"]:
            history[f"train_{m}"].append(train_metrics[m])
            history[f"val_{m}"].append(val_metrics[m])

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow({
                "epoch": epoch + 1, "lr": lr,
                "train_loss": train_metrics["loss"],
                "train_species_acc": train_metrics["species_acc"],
                "train_call_acc": train_metrics["call_acc"],
                "train_species_f1": train_metrics["species_f1"],
                "val_loss": val_metrics["loss"],
                "val_species_acc": val_metrics["species_acc"],
                "val_call_acc": val_metrics["call_acc"],
                "val_species_f1": val_metrics["species_f1"],
            })

        # Save best checkpoint
        val_acc = val_metrics["mean_acc"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch + 1, val_metrics, cfg, "best_model.pth")
            print(f"  ✓ New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

        # Save latest checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch + 1, val_metrics, cfg,
                            f"checkpoint_epoch_{epoch+1}.pth")

    # ── Final evaluation on test set ──────────────────────────────────────────
    print("\n" + "="*60)
    print("Final Test Set Evaluation")
    print("="*60)
    from src.model import load_model
    best_model = load_model(
        os.path.join(cfg.checkpoint_dir, "best_model.pth"),
        device, len(SPECIES_LIST), len(CALL_TYPES)
    )
    test_metrics = eval_epoch(best_model, test_loader, criterion, device)
    print(f"Test species accuracy: {test_metrics['species_acc']:.4f}")
    print(f"Test call type accuracy: {test_metrics['call_acc']:.4f}")
    print(f"Test species macro-F1: {test_metrics['species_f1']:.4f}")
    print(f"Test call type macro-F1: {test_metrics['call_f1']:.4f}")

    with open(os.path.join(cfg.log_dir, "test_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    plot_curves(history, cfg.log_dir)
    print(f"\nTraining complete. Artifacts saved to: {cfg.checkpoint_dir}/ and {cfg.log_dir}/")
    return best_model


if __name__ == "__main__":
    train()
