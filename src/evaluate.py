"""
Bird Language ML - Model Evaluation
Generates detailed evaluation report:
  - Per-class accuracy and F1
  - Confusion matrices (species + call type)
  - ROC curves and AUC scores
  - Top-5 accuracy
  - Embedding space visualization (t-SNE)
"""

import os
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, top_k_accuracy_score
)
from sklearn.manifold import TSNE

from config import SPECIES_LIST, CALL_TYPES, audio_cfg, aug_cfg, train_cfg
from src.model import load_model
from src.dataset import build_dataloaders
from src.train import MetricsTracker

EVAL_DIR = Path("logs/evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Collect Predictions
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    sp_probs_all, ct_probs_all = [], []
    sp_labels_all, ct_labels_all = [], []
    embeddings_all = []

    for batch in loader:
        specs = batch["spectrogram"].to(device)
        out = model.predict(specs)

        sp_probs_all.append(out["species_probs"].cpu().numpy())
        ct_probs_all.append(out["call_probs"].cpu().numpy())
        sp_labels_all.extend(batch["species_label"].numpy())
        ct_labels_all.extend(batch["call_type_label"].numpy())
        embeddings_all.append(out["embedding"].cpu().numpy())

    return {
        "sp_probs": np.concatenate(sp_probs_all),
        "ct_probs": np.concatenate(ct_probs_all),
        "sp_labels": np.array(sp_labels_all),
        "ct_labels": np.array(ct_labels_all),
        "embeddings": np.concatenate(embeddings_all),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, title, save_path, figsize=(12, 10)):
    # Normalize
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm, annot=(len(class_names) <= 15), fmt=".2f",
        cmap="Blues", xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.3, vmin=0, vmax=1,
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tsne_embeddings(embeddings, labels, class_names, title, save_path):
    print("  Computing t-SNE (this takes ~1-2 min)...")
    n_samples = min(2000, len(embeddings))
    idx = np.random.choice(len(embeddings), n_samples, replace=False)
    emb_sub = embeddings[idx]
    lab_sub = labels[idx]

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    emb_2d = tsne.fit_transform(emb_sub)

    cmap = plt.cm.get_cmap("tab20", len(class_names))
    fig, ax = plt.subplots(figsize=(12, 10))
    for i, name in enumerate(class_names):
        mask = lab_sub == i
        if mask.sum() == 0:
            continue
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[cmap(i)], label=name[:20], alpha=0.6, s=15)

    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=7, ncol=3, markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_metrics(report_dict, class_names, title, save_path):
    f1_scores = [report_dict.get(c, {}).get("f1-score", 0) for c in class_names]
    accs = [report_dict.get(c, {}).get("precision", 0) for c in class_names]

    x = np.arange(len(class_names))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.4), 5))
    ax.bar(x - width/2, f1_scores, width, label="F1 Score", color="#4C72B0", alpha=0.85)
    ax.bar(x + width/2, accs, width, label="Precision", color="#DD8452", alpha=0.85)
    ax.axhline(np.mean(f1_scores), color="#4C72B0", linestyle="--", alpha=0.5, label=f"Mean F1={np.mean(f1_scores):.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([c[:15] for c in class_names], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(checkpoint_path: str = "checkpoints/best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # Load model
    model = load_model(checkpoint_path, device, len(SPECIES_LIST), len(CALL_TYPES))

    # Load test data
    _, _, test_loader = build_dataloaders(
        train_cfg.metadata_csv, audio_cfg, aug_cfg, train_cfg, SPECIES_LIST, CALL_TYPES
    )

    print("\nCollecting test predictions...")
    preds = collect_predictions(model, test_loader, device)

    sp_preds = preds["sp_probs"].argmax(axis=1)
    ct_preds = preds["ct_probs"].argmax(axis=1)

    # ── Species metrics ────────────────────────────────────────────────────────
    print("\n── Species Classification ─────────────────────")
    sp_report = classification_report(
        preds["sp_labels"], sp_preds, target_names=SPECIES_LIST,
        output_dict=True, zero_division=0
    )
    print(classification_report(preds["sp_labels"], sp_preds, target_names=SPECIES_LIST, zero_division=0))

    top5_sp = top_k_accuracy_score(preds["sp_labels"], preds["sp_probs"], k=5)
    print(f"Top-5 Species Accuracy: {top5_sp:.4f}")

    # ── Call type metrics ──────────────────────────────────────────────────────
    print("\n── Call Type Classification ───────────────────")
    ct_report = classification_report(
        preds["ct_labels"], ct_preds, target_names=CALL_TYPES,
        output_dict=True, zero_division=0
    )
    print(classification_report(preds["ct_labels"], ct_preds, target_names=CALL_TYPES, zero_division=0))

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating evaluation plots...")

    # Confusion matrices
    sp_cm = confusion_matrix(preds["sp_labels"], sp_preds)
    plot_confusion_matrix(sp_cm, SPECIES_LIST, "Species Confusion Matrix (normalized)",
                          EVAL_DIR / "species_confusion_matrix.png", figsize=(16, 14))

    ct_cm = confusion_matrix(preds["ct_labels"], ct_preds)
    plot_confusion_matrix(ct_cm, CALL_TYPES, "Call Type Confusion Matrix (normalized)",
                          EVAL_DIR / "call_type_confusion_matrix.png", figsize=(9, 8))

    # Per-class bar charts
    plot_per_class_metrics(sp_report, SPECIES_LIST, "Per-species Precision & F1",
                           EVAL_DIR / "species_per_class.png")
    plot_per_class_metrics(ct_report, CALL_TYPES, "Per-call-type Precision & F1",
                           EVAL_DIR / "call_type_per_class.png")

    # t-SNE embedding space
    plot_tsne_embeddings(preds["embeddings"], preds["ct_labels"], CALL_TYPES,
                         "Embedding Space (t-SNE) — Color by Call Type",
                         EVAL_DIR / "embeddings_tsne.png")

    # ── Save summary JSON ──────────────────────────────────────────────────────
    summary = {
        "species": {
            "accuracy": sp_report["accuracy"],
            "macro_f1": sp_report["macro avg"]["f1-score"],
            "top5_accuracy": top5_sp,
        },
        "call_type": {
            "accuracy": ct_report["accuracy"],
            "macro_f1": ct_report["macro avg"]["f1-score"],
        },
    }
    with open(EVAL_DIR / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Species accuracy:  {summary['species']['accuracy']:.4f}")
    print(f"Species macro-F1:  {summary['species']['macro_f1']:.4f}")
    print(f"Species top-5 acc: {summary['species']['top5_accuracy']:.4f}")
    print(f"Call type accuracy: {summary['call_type']['accuracy']:.4f}")
    print(f"Call type macro-F1: {summary['call_type']['macro_f1']:.4f}")
    print(f"\nAll evaluation artifacts saved to: {EVAL_DIR}/")


if __name__ == "__main__":
    evaluate()
