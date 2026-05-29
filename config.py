"""
Bird Language ML - Central Configuration
All hyperparameters, paths, and constants in one place.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AudioConfig:
    sample_rate: int = 32000          # BirdNET standard SR
    duration: float = 5.0             # seconds per chunk
    n_fft: int = 1024
    hop_length: int = 320             # ~10ms hop at 32kHz
    n_mels: int = 128                 # mel filterbanks
    f_min: float = 50.0              # Hz - birds rarely below 50Hz
    f_max: float = 15000.0           # Hz - covers most bird calls
    power: float = 2.0               # spectrogram power
    top_db: float = 80.0             # dynamic range for log compression


@dataclass
class AugmentationConfig:
    # Time-domain augmentations
    time_shift_prob: float = 0.5
    time_shift_limit: float = 0.1    # ±10% of duration
    gain_prob: float = 0.5
    gain_range: tuple = (-6, 6)      # dB
    noise_prob: float = 0.4
    noise_snr_range: tuple = (10, 30) # dB

    # Spectrogram augmentations (SpecAugment)
    freq_mask_prob: float = 0.5
    freq_mask_size: int = 20         # max mel bins to mask
    time_mask_prob: float = 0.5
    time_mask_size: int = 30         # max time frames to mask
    mixup_prob: float = 0.3
    mixup_alpha: float = 0.4


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b0"  # from timm
    pretrained: bool = True
    num_classes: int = 50              # number of bird species/call types
    dropout: float = 0.3
    embedding_dim: int = 1280         # EfficientNet-B0 output dim

    # Multi-task heads
    species_classes: int = 50          # bird species
    call_type_classes: int = 9         # call type categories


CALL_TYPES = [
    "alarm_call",
    "mating_song",
    "contact_call",
    "territorial_call",
    "distress_call",
    "begging_call",
    "flight_call",
    "foraging_sound",
    "dawn_chorus",
]

CALL_TYPE_DESCRIPTIONS = {
    "alarm_call":       "Sharp, repeated chips signaling danger to nearby birds",
    "mating_song":      "Complex melodic song to attract mates, peak at dawn",
    "contact_call":     "Soft calls to maintain flock cohesion while moving",
    "territorial_call": "Loud, persistent song to defend territory boundaries",
    "distress_call":    "Urgent, high-pitched call when captured or threatened",
    "begging_call":     "Insistent repetitive calls by chicks demanding food",
    "flight_call":      "Brief chips emitted during nocturnal or active migration",
    "foraging_sound":   "Soft, exploratory sounds while searching for food",
    "dawn_chorus":      "Collective multi-species singing at sunrise",
}

# Top 50 common species (global + India focus)
SPECIES_LIST = [
    "House Sparrow", "Common Myna", "Rock Pigeon", "Rose-ringed Parakeet",
    "Red-vented Bulbul", "Asian Koel", "Indian Peafowl", "Jungle Crow",
    "Common Kingfisher", "White-throated Kingfisher", "Barn Swallow",
    "Wire-tailed Swallow", "Purple Sunbird", "Ashy Drongo", "Indian Robin",
    "Oriental Magpie-Robin", "Common Tailorbird", "Jungle Babbler",
    "Indian Pond Heron", "Little Egret", "Cattle Egret", "Black-winged Stilt",
    "Red-wattled Lapwing", "Spotted Dove", "Laughing Dove", "Common Swift",
    "Brown-headed Barbet", "Coppersmith Barbet", "Black-rumped Flameback",
    "Greater Coucal", "Shikra", "Black Kite", "Brahminy Kite", "Osprey",
    "Indian Grey Hornbill", "Common Hoopoe", "Indian Roller", "Bee-eater",
    "Pied Wagtail", "White Wagtail", "Common Sandpiper", "Green Bee-eater",
    "Eurasian Tree Sparrow", "Baya Weaver", "Scaly-breasted Munia",
    "White-rumped Munia", "Black-headed Munia", "Indian Silverbill",
    "Paddy field Pipit", "Richard's Pipit",
]


@dataclass
class TrainingConfig:
    # Data
    data_dir: str = "data/audio"
    metadata_csv: str = "data/metadata.csv"
    val_split: float = 0.15
    test_split: float = 0.10
    num_workers: int = 4

    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    label_smoothing: float = 0.1

    # Optimizer
    optimizer: str = "adamw"
    scheduler: str = "cosine"      # cosine annealing with warmup

    # Mixed precision
    use_amp: bool = True            # automatic mixed precision (faster)

    # Regularization
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    log_dir: str = "logs"

    # Multi-task loss weights
    species_loss_weight: float = 0.6
    call_type_loss_weight: float = 0.4


@dataclass
class InferenceConfig:
    model_path: str = "checkpoints/best_model.pth"
    confidence_threshold: float = 0.3
    top_k: int = 3
    chunk_overlap: float = 0.5      # 50% overlap for sliding window
    device: str = "auto"            # "auto", "cuda", "mps", "cpu"


# Singleton configs
audio_cfg = AudioConfig()
aug_cfg = AugmentationConfig()
model_cfg = ModelConfig()
train_cfg = TrainingConfig()
infer_cfg = InferenceConfig()
