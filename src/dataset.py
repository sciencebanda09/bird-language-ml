"""
Bird Language ML - Dataset & Audio Preprocessing
Handles loading audio, converting to mel-spectrograms, augmentations,
and building PyTorch datasets for training and inference.
"""

import os
import random
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import torchaudio.transforms as T
import librosa
import librosa.effects

warnings.filterwarnings("ignore")

from config import AudioConfig, AugmentationConfig, TrainingConfig, CALL_TYPES, SPECIES_LIST


# ──────────────────────────────────────────────────────────────────────────────
# Core Audio Utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, cfg: AudioConfig) -> np.ndarray:
    """Load audio file and resample to target SR. Returns mono float32 array."""
    waveform, sr = librosa.load(path, sr=cfg.sample_rate, mono=True)
    return waveform.astype(np.float32)


def pad_or_trim(waveform: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Pad (with zeros) or trim waveform to exactly cfg.duration seconds."""
    target_len = int(cfg.sample_rate * cfg.duration)
    if len(waveform) >= target_len:
        start = random.randint(0, len(waveform) - target_len)
        return waveform[start: start + target_len]
    # Pad with wrap to preserve spectral texture
    repeats = (target_len // len(waveform)) + 1
    waveform = np.tile(waveform, repeats)
    return waveform[:target_len]


def waveform_to_melspec(waveform: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """
    Convert raw waveform to log-mel spectrogram.
    Output shape: (n_mels, time_frames)
    """
    spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.f_max,
        power=cfg.power,
    )
    # Convert to dB scale and normalize to [0, 1]
    log_spec = librosa.power_to_db(spec, ref=np.max, top_db=cfg.top_db)
    log_spec = (log_spec + cfg.top_db) / cfg.top_db  # → [0, 1]
    return log_spec.astype(np.float32)


def chunk_audio(waveform: np.ndarray, cfg: AudioConfig, overlap: float = 0.5) -> List[np.ndarray]:
    """
    Split long waveform into overlapping chunks for inference.
    Returns list of chunks, each exactly cfg.duration seconds.
    """
    chunk_len = int(cfg.sample_rate * cfg.duration)
    step = int(chunk_len * (1 - overlap))
    chunks = []
    for start in range(0, len(waveform) - chunk_len + 1, step):
        chunks.append(waveform[start: start + chunk_len])
    if not chunks:
        chunks.append(pad_or_trim(waveform, cfg))
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Augmentations (applied in waveform domain)
# ──────────────────────────────────────────────────────────────────────────────

class WaveformAugmenter:
    """Time-domain augmentations applied to raw waveform."""

    def __init__(self, cfg: AugmentationConfig, audio_cfg: AudioConfig):
        self.cfg = cfg
        self.audio_cfg = audio_cfg

    def time_shift(self, waveform: np.ndarray) -> np.ndarray:
        shift = int(np.random.uniform(-self.cfg.time_shift_limit,
                                      self.cfg.time_shift_limit) * len(waveform))
        return np.roll(waveform, shift)

    def gain(self, waveform: np.ndarray) -> np.ndarray:
        db = np.random.uniform(*self.cfg.gain_range)
        return waveform * (10 ** (db / 20))

    def add_gaussian_noise(self, waveform: np.ndarray) -> np.ndarray:
        snr_db = np.random.uniform(*self.cfg.noise_snr_range)
        signal_power = np.mean(waveform ** 2) + 1e-10
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(len(waveform)) * np.sqrt(noise_power)
        return (waveform + noise).astype(np.float32)

    def __call__(self, waveform: np.ndarray, training: bool = True) -> np.ndarray:
        if not training:
            return waveform
        if np.random.random() < self.cfg.time_shift_prob:
            waveform = self.time_shift(waveform)
        if np.random.random() < self.cfg.gain_prob:
            waveform = self.gain(waveform)
        if np.random.random() < self.cfg.noise_prob:
            waveform = self.add_gaussian_noise(waveform)
        # Clip to prevent clipping distortion
        return np.clip(waveform, -1.0, 1.0)


class SpecAugment:
    """
    Spectrogram-domain augmentations (SpecAugment - Park et al. 2019).
    Applied to mel-spectrogram tensor of shape (1, n_mels, time).
    """

    def __init__(self, cfg: AugmentationConfig):
        self.cfg = cfg

    def freq_mask(self, spec: torch.Tensor) -> torch.Tensor:
        n_mels = spec.shape[-2]
        size = random.randint(1, self.cfg.freq_mask_size)
        start = random.randint(0, n_mels - size)
        spec[..., start: start + size, :] = 0.0
        return spec

    def time_mask(self, spec: torch.Tensor) -> torch.Tensor:
        n_frames = spec.shape[-1]
        size = random.randint(1, min(self.cfg.time_mask_size, n_frames - 1))
        start = random.randint(0, n_frames - size)
        spec[..., start: start + size] = 0.0
        return spec

    def __call__(self, spec: torch.Tensor, training: bool = True) -> torch.Tensor:
        if not training:
            return spec
        if random.random() < self.cfg.freq_mask_prob:
            spec = self.freq_mask(spec)
        if random.random() < self.cfg.time_mask_prob:
            spec = self.time_mask(spec)
        return spec


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class BirdSoundDataset(Dataset):
    """
    PyTorch Dataset for bird audio classification.

    Expects a CSV with columns:
        filepath, species, call_type, [quality, duration, country, ...]

    Outputs:
        spectrogram tensor (3, n_mels, time_frames)  ← 3-channel for pretrained CNNs
        species label (int)
        call_type label (int)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_cfg: AudioConfig,
        aug_cfg: AugmentationConfig,
        species_list: List[str],
        call_types: List[str],
        training: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_cfg = audio_cfg
        self.aug_cfg = aug_cfg
        self.training = training

        self.species2idx = {s: i for i, s in enumerate(species_list)}
        self.call2idx = {c: i for i, c in enumerate(call_types)}

        self.waveform_aug = WaveformAugmenter(aug_cfg, audio_cfg)
        self.spec_aug = SpecAugment(aug_cfg)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        filepath = row["filepath"]

        # ── Load & augment waveform ──
        try:
            waveform = load_audio(filepath, self.audio_cfg)
        except Exception as e:
            waveform = np.zeros(int(self.audio_cfg.sample_rate * self.audio_cfg.duration),
                                dtype=np.float32)

        waveform = pad_or_trim(waveform, self.audio_cfg)
        waveform = self.waveform_aug(waveform, training=self.training)

        # ── Convert to mel-spectrogram ──
        spec = waveform_to_melspec(waveform, self.audio_cfg)   # (n_mels, time)
        spec = torch.from_numpy(spec).unsqueeze(0)              # (1, n_mels, time)

        # ── SpecAugment ──
        spec = self.spec_aug(spec, training=self.training)

        # ── Repeat to 3 channels for pretrained CNN backbone ──
        spec = spec.repeat(3, 1, 1)                             # (3, n_mels, time)

        # ── Labels ──
        species_idx = self.species2idx.get(row["species"], 0)
        call_idx = self.call2idx.get(row["call_type"], 0)

        return {
            "spectrogram": spec,
            "species_label": torch.tensor(species_idx, dtype=torch.long),
            "call_type_label": torch.tensor(call_idx, dtype=torch.long),
            "filepath": filepath,
        }

    def mixup_collate(self, batch):
        """Custom collate with Mixup augmentation."""
        specs = torch.stack([b["spectrogram"] for b in batch])
        sp_labels = torch.stack([b["species_label"] for b in batch])
        ct_labels = torch.stack([b["call_type_label"] for b in batch])

        if self.training and random.random() < self.aug_cfg.mixup_prob:
            alpha = self.aug_cfg.mixup_alpha
            lam = np.random.beta(alpha, alpha)
            idx = torch.randperm(len(specs))
            specs = lam * specs + (1 - lam) * specs[idx]
            # Return mixed labels as a tuple for soft-label loss
            return {
                "spectrogram": specs,
                "species_label": (sp_labels, sp_labels[idx], lam),
                "call_type_label": (ct_labels, ct_labels[idx], lam),
                "filepath": [b["filepath"] for b in batch],
                "mixed": True,
            }

        return {
            "spectrogram": specs,
            "species_label": sp_labels,
            "call_type_label": ct_labels,
            "filepath": [b["filepath"] for b in batch],
            "mixed": False,
        }


def build_dataloaders(
    metadata_csv: str,
    audio_cfg: AudioConfig,
    aug_cfg: AugmentationConfig,
    train_cfg: TrainingConfig,
    species_list: List[str],
    call_types: List[str],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split metadata CSV into train/val/test and build DataLoaders.
    Uses class-balanced WeightedRandomSampler for training.
    """
    df = pd.read_csv(metadata_csv)
    # Filter to known species & call types
    df = df[df["species"].isin(species_list) & df["call_type"].isin(call_types)]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    n_test = int(n * train_cfg.test_split)
    n_val = int(n * train_cfg.val_split)

    test_df = df.iloc[:n_test]
    val_df = df.iloc[n_test: n_test + n_val]
    train_df = df.iloc[n_test + n_val:]

    train_ds = BirdSoundDataset(train_df, audio_cfg, aug_cfg, species_list, call_types, training=True)
    val_ds = BirdSoundDataset(val_df, audio_cfg, aug_cfg, species_list, call_types, training=False)
    test_ds = BirdSoundDataset(test_df, audio_cfg, aug_cfg, species_list, call_types, training=False)

    # Class-balanced sampler to handle imbalanced species distributions
    species2idx = {s: i for i, s in enumerate(species_list)}
    labels = train_df["species"].map(species2idx).values
    class_counts = np.bincount(labels, minlength=len(species_list))
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, sampler=sampler,
        num_workers=train_cfg.num_workers, pin_memory=True,
        collate_fn=train_ds.mixup_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size * 2, shuffle=False,
        num_workers=train_cfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg.batch_size * 2, shuffle=False,
        num_workers=train_cfg.num_workers, pin_memory=True,
    )

    print(f"Dataset splits → Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
