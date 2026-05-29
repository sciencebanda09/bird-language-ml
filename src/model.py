"""
Bird Language ML - Model Architecture
Multi-task CNN: EfficientNet-B0 backbone with dual classification heads
- Head 1: Bird species (50 classes)
- Head 2: Call type / communication meaning (9 classes)

Architecture overview:
  Input (3, 128, 313)        ← 3-channel mel-spectrogram
  ↓ EfficientNet-B0 backbone (pretrained ImageNet)
  ↓ Global Average Pooling   → (1280,)
  ↓ Dropout + BatchNorm
  ├─ Species Head            → softmax over N species
  └─ Call Type Head          → softmax over 9 call types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Dict, Optional


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over the time axis of the spectrogram.
    Learns which time frames are most discriminative.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        weights = self.attn(x)              # (batch, time, 1)
        weights = F.softmax(weights, dim=1)
        pooled = (x * weights).sum(dim=1)  # (batch, features)
        return pooled


class ClassificationHead(nn.Module):
    """
    Shared classification head structure used for both tasks.
    FC → BN → GELU → Dropout → FC → output logits
    """

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        hidden = in_dim // 2
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class BirdLanguageModel(nn.Module):
    """
    Production bird sound classification model.

    Args:
        backbone:         timm model name (efficientnet_b0, tf_efficientnetv2_s, etc.)
        pretrained:       load ImageNet weights
        species_classes:  number of bird species to classify
        call_type_classes: number of call/communication types
        dropout:          dropout probability in heads
        embedding_dim:    feature dimension from backbone

    Forward returns a dict with:
        species_logits:   (B, species_classes)
        call_logits:      (B, call_type_classes)
        embedding:        (B, embedding_dim)  ← for retrieval / similarity search
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        species_classes: int = 50,
        call_type_classes: int = 9,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,           # remove default classifier
            global_pool="avg",       # global average pooling
            in_chans=3,              # 3-channel spectrogram input
        )
        embedding_dim = self.backbone.num_features

        # ── Shared projection ─────────────────────────────────────────────────
        self.shared_proj = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout * 0.5),
        )

        # ── Task-specific heads ───────────────────────────────────────────────
        self.species_head = ClassificationHead(embedding_dim, species_classes, dropout)
        self.call_head = ClassificationHead(embedding_dim, call_type_classes, dropout)

        # ── L2-normalized embedding for similarity search ─────────────────────
        self.embedding_proj = nn.Linear(embedding_dim, 256, bias=False)

        self._init_heads()

    def _init_heads(self):
        """Initialize classification heads with kaiming uniform."""
        for module in [self.species_head, self.call_head, self.embedding_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: mel-spectrogram tensor (B, 3, n_mels, time_frames)
        Returns:
            dict with species_logits, call_logits, embedding
        """
        features = self.backbone(x)           # (B, embedding_dim)
        features = self.shared_proj(features)

        species_logits = self.species_head(features)
        call_logits = self.call_head(features)

        # Normalized embedding for retrieval tasks
        embedding = self.embedding_proj(features)
        embedding = F.normalize(embedding, p=2, dim=1)

        return {
            "species_logits": species_logits,
            "call_logits": call_logits,
            "embedding": embedding,
            "features": features,
        }

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience method returning probabilities (not logits)."""
        with torch.no_grad():
            out = self.forward(x)
        return {
            "species_probs": F.softmax(out["species_logits"], dim=-1),
            "call_probs": F.softmax(out["call_logits"], dim=-1),
            "embedding": out["embedding"],
        }

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_head_params(self):
        return list(self.species_head.parameters()) + \
               list(self.call_head.parameters()) + \
               list(self.shared_proj.parameters()) + \
               list(self.embedding_proj.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────────────────────

class BirdLanguageLoss(nn.Module):
    """
    Multi-task loss combining species and call-type cross-entropy.
    Supports Mixup (soft labels) and label smoothing.
    """

    def __init__(
        self,
        species_weight: float = 0.6,
        call_weight: float = 0.4,
        label_smoothing: float = 0.1,
        num_species: int = 50,
        num_call_types: int = 9,
    ):
        super().__init__()
        self.species_weight = species_weight
        self.call_weight = call_weight
        self.num_species = num_species
        self.num_call_types = num_call_types
        self.label_smoothing = label_smoothing

        self.ce_species = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.ce_call = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _mixup_loss(self, logits, labels_a, labels_b, lam, num_classes, ce_fn):
        """Compute mixup loss: λ·loss(a) + (1-λ)·loss(b)."""
        loss_a = ce_fn(logits, labels_a)
        loss_b = ce_fn(logits, labels_b)
        return lam * loss_a + (1 - lam) * loss_b

    def forward(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        mixed = batch.get("mixed", False)

        if mixed:
            sp_a, sp_b, lam = batch["species_label"]
            ct_a, ct_b, lam_ct = batch["call_type_label"]

            sp_loss = self._mixup_loss(
                outputs["species_logits"], sp_a, sp_b, lam, self.num_species, self.ce_species
            )
            ct_loss = self._mixup_loss(
                outputs["call_logits"], ct_a, ct_b, lam_ct, self.num_call_types, self.ce_call
            )
        else:
            sp_loss = self.ce_species(outputs["species_logits"], batch["species_label"])
            ct_loss = self.ce_call(outputs["call_logits"], batch["call_type_label"])

        total = self.species_weight * sp_loss + self.call_weight * ct_loss

        return {
            "total_loss": total,
            "species_loss": sp_loss,
            "call_loss": ct_loss,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Model Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    species_classes: int = 50,
    call_type_classes: int = 9,
    dropout: float = 0.3,
) -> BirdLanguageModel:
    model = BirdLanguageModel(
        backbone=backbone,
        pretrained=pretrained,
        species_classes=species_classes,
        call_type_classes=call_type_classes,
        dropout=dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {backbone} | Trainable params: {n_params/1e6:.2f}M")
    return model


def load_model(
    checkpoint_path: str,
    device: torch.device,
    species_classes: int = 50,
    call_type_classes: int = 9,
) -> BirdLanguageModel:
    """Load a trained model from checkpoint."""
    model = build_model(pretrained=False, species_classes=species_classes,
                        call_type_classes=call_type_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
          f"(val_acc: {checkpoint.get('val_acc', 0):.3f})")
    return model
