"""
Bird Language ML - Inference Engine
Production-ready inference with:
  - Sliding window analysis for long recordings
  - Confidence thresholding & top-k results
  - Temporal aggregation across overlapping windows
  - Batch inference for speed
  - Full meaning/context lookup
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    AudioConfig, InferenceConfig, SPECIES_LIST, CALL_TYPES,
    CALL_TYPE_DESCRIPTIONS, audio_cfg, infer_cfg
)
from src.dataset import load_audio, pad_or_trim, chunk_audio, waveform_to_melspec
from src.model import BirdLanguageModel, load_model


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


class BirdLanguageInference:
    """
    End-to-end inference pipeline for bird sound analysis.

    Usage:
        engine = BirdLanguageInference("checkpoints/best_model.pth")
        results = engine.analyze("path/to/bird_recording.wav")
        print(results)
    """

    def __init__(
        self,
        model_path: str,
        cfg: InferenceConfig = infer_cfg,
        audio_config: AudioConfig = audio_cfg,
        device: Optional[str] = None,
    ):
        self.cfg = cfg
        self.audio_cfg = audio_config
        self.device = get_device(device or cfg.device)

        self.species_list = SPECIES_LIST
        self.call_types = CALL_TYPES

        print(f"Loading model on {self.device}...")
        self.model = load_model(
            model_path, self.device,
            species_classes=len(SPECIES_LIST),
            call_type_classes=len(CALL_TYPES),
        )
        self.model.eval()
        print("Inference engine ready.")

    def preprocess(self, waveform: np.ndarray) -> torch.Tensor:
        """Convert waveform chunk to model-ready tensor (1, 3, n_mels, time)."""
        spec = waveform_to_melspec(waveform, self.audio_cfg)
        spec = torch.from_numpy(spec).unsqueeze(0).repeat(3, 1, 1)  # (3, n_mels, time)
        return spec.unsqueeze(0)  # (1, 3, n_mels, time)

    @torch.no_grad()
    def predict_chunk(self, waveform: np.ndarray) -> Dict:
        """Run model on a single audio chunk."""
        spec = self.preprocess(waveform).to(self.device)
        out = self.model.predict(spec)
        return {
            "species_probs": out["species_probs"][0].cpu().numpy(),
            "call_probs": out["call_probs"][0].cpu().numpy(),
            "embedding": out["embedding"][0].cpu().numpy(),
        }

    @torch.no_grad()
    def predict_batch(self, chunks: List[np.ndarray]) -> Dict:
        """Batch inference for efficiency on long recordings."""
        batch = torch.stack([
            self.preprocess(c)[0] for c in chunks
        ]).to(self.device)                              # (B, 3, n_mels, time)

        sp_probs = []
        ct_probs = []
        embeddings = []

        # Process in mini-batches to avoid OOM
        mini_bs = 16
        for i in range(0, len(batch), mini_bs):
            sub = batch[i: i + mini_bs]
            out = self.model.predict(sub)
            sp_probs.append(out["species_probs"].cpu().numpy())
            ct_probs.append(out["call_probs"].cpu().numpy())
            embeddings.append(out["embedding"].cpu().numpy())

        return {
            "species_probs": np.concatenate(sp_probs, axis=0),
            "call_probs": np.concatenate(ct_probs, axis=0),
            "embeddings": np.concatenate(embeddings, axis=0),
        }

    def aggregate_predictions(self, probs: np.ndarray, method: str = "mean") -> np.ndarray:
        """
        Aggregate chunk-level predictions to file-level.
        method: "mean" | "max" | "geometric_mean"
        """
        if method == "mean":
            return probs.mean(axis=0)
        elif method == "max":
            return probs.max(axis=0)
        elif method == "geometric_mean":
            log_probs = np.log(probs + 1e-10)
            return np.exp(log_probs.mean(axis=0))
        return probs.mean(axis=0)

    def format_results(
        self,
        sp_probs: np.ndarray,
        ct_probs: np.ndarray,
        recording_duration: float,
        filepath: str = "",
    ) -> Dict:
        """Structure inference results into human-readable output."""

        # Top-k species
        sp_topk_idx = np.argsort(sp_probs)[::-1][: self.cfg.top_k]
        species_results = []
        for idx in sp_topk_idx:
            conf = float(sp_probs[idx])
            if conf >= self.cfg.confidence_threshold:
                species_results.append({
                    "species": self.species_list[idx],
                    "confidence": round(conf, 4),
                    "percentage": f"{conf*100:.1f}%",
                })

        # Top call type
        ct_topk_idx = np.argsort(ct_probs)[::-1][:3]
        call_results = []
        for idx in ct_topk_idx:
            call_type = self.call_types[idx]
            conf = float(ct_probs[idx])
            call_results.append({
                "call_type": call_type,
                "label": call_type.replace("_", " ").title(),
                "confidence": round(conf, 4),
                "percentage": f"{conf*100:.1f}%",
                "description": CALL_TYPE_DESCRIPTIONS.get(call_type, ""),
            })

        primary_call = call_results[0] if call_results else {}
        primary_species = species_results[0] if species_results else {}

        return {
            "filepath": filepath,
            "duration_seconds": round(recording_duration, 2),
            "primary_species": primary_species,
            "primary_call_type": primary_call,
            "top_species": species_results,
            "top_call_types": call_results,
            "communication_meaning": self._explain_communication(
                primary_species.get("species", "Unknown"),
                primary_call.get("call_type", "unknown"),
                primary_call.get("confidence", 0),
            ),
            "model_confidence": {
                "species": round(float(sp_probs.max()), 4),
                "call_type": round(float(ct_probs.max()), 4),
            }
        }

    def _explain_communication(self, species: str, call_type: str, confidence: float) -> str:
        """Generate a natural language explanation of what the bird is communicating."""
        if confidence < self.cfg.confidence_threshold:
            return "Low confidence — could not reliably determine communication type."

        explanations = {
            "alarm_call": f"The {species} is alerting nearby birds to a potential predator or danger. "
                          "This urgent signal triggers a coordinated response — others freeze, flee, or mob.",
            "mating_song": f"The {species} is advertising its fitness to potential mates. "
                           "Song complexity signals genetic quality and territory ownership.",
            "contact_call": f"The {species} is maintaining social cohesion with its flock using soft locator calls. "
                            "These ensure group members stay connected while foraging.",
            "territorial_call": f"The {species} is broadcasting ownership of its territory. "
                                "This warns rival males to keep their distance.",
            "distress_call": f"The {species} is emitting a distress signal — likely caught or cornered. "
                             "This can recruit helpers and startle predators.",
            "begging_call": f"Juvenile {species} is begging for food from parents. "
                            "High-pitched persistence signals hunger level.",
            "flight_call": f"The {species} is using a brief contact note during movement or migration. "
                           "These coordinate group flight and maintain contact in darkness.",
            "foraging_sound": f"The {species} is producing soft exploratory sounds while searching for food. "
                              "May also signal food discovery to nearby kin.",
            "dawn_chorus": f"The {species} is participating in the dawn chorus — peak singing intensity at sunrise. "
                           "Multiple functions: mate attraction, territory reinforcement, social bonding.",
        }
        return explanations.get(call_type, f"The {species} is vocalizing — context unclear.")

    def analyze(
        self,
        audio_path: str,
        overlap: float = 0.5,
        aggregate_method: str = "mean",
        return_chunk_details: bool = False,
    ) -> Dict:
        """
        Full analysis pipeline for an audio file.

        Args:
            audio_path:           path to .wav or .mp3 file
            overlap:              sliding window overlap (0–1)
            aggregate_method:     how to merge chunk predictions
            return_chunk_details: include per-chunk results

        Returns:
            Structured results dict
        """
        print(f"Analyzing: {audio_path}")

        # Load full recording
        waveform = load_audio(audio_path, self.audio_cfg)
        duration = len(waveform) / self.audio_cfg.sample_rate
        print(f"  Duration: {duration:.1f}s | SR: {self.audio_cfg.sample_rate}Hz")

        # Chunk into overlapping windows
        chunks = chunk_audio(waveform, self.audio_cfg, overlap=overlap)
        print(f"  Windows: {len(chunks)} × {self.audio_cfg.duration}s (overlap={overlap*100:.0f}%)")

        # Batch inference
        preds = self.predict_batch(chunks)

        # Aggregate
        sp_agg = self.aggregate_predictions(preds["species_probs"], aggregate_method)
        ct_agg = self.aggregate_predictions(preds["call_probs"], aggregate_method)

        results = self.format_results(sp_agg, ct_agg, duration, audio_path)

        # Optional: per-chunk timeline
        if return_chunk_details:
            chunk_step = self.audio_cfg.duration * (1 - overlap)
            timeline = []
            for i, (sp_p, ct_p) in enumerate(zip(preds["species_probs"], preds["call_probs"])):
                t_start = i * chunk_step
                sp_idx = np.argmax(sp_p)
                ct_idx = np.argmax(ct_p)
                timeline.append({
                    "time_start": round(t_start, 2),
                    "time_end": round(t_start + self.audio_cfg.duration, 2),
                    "species": self.species_list[sp_idx],
                    "call_type": self.call_types[ct_idx],
                    "sp_confidence": round(float(sp_p[sp_idx]), 4),
                    "ct_confidence": round(float(ct_p[ct_idx]), 4),
                })
            results["timeline"] = timeline

        return results

    def analyze_directory(self, directory: str, extensions: Tuple = (".wav", ".mp3", ".flac", ".ogg")) -> List[Dict]:
        """Batch analyze all audio files in a directory."""
        files = [
            str(p) for p in Path(directory).rglob("*")
            if p.suffix.lower() in extensions
        ]
        print(f"Found {len(files)} audio files in {directory}")
        results = []
        for i, f in enumerate(files):
            print(f"\n[{i+1}/{len(files)}]", end=" ")
            try:
                r = self.analyze(f)
                results.append(r)
            except Exception as e:
                print(f"Error: {e}")
                results.append({"filepath": f, "error": str(e)})
        return results


def demo_inference():
    """
    Quick demo — generates a synthetic chirp and runs inference.
    Use this to verify the pipeline works before real audio.
    """
    import soundfile as sf
    import tempfile

    print("=== Bird Language ML - Inference Demo ===\n")

    # Generate a synthetic bird-like chirp (for pipeline testing)
    sr = 32000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))

    # Simulate a frequency-modulated bird call
    freq_mod = 4000 + 2000 * np.sin(2 * np.pi * 8 * t)
    chirp = 0.3 * np.sin(2 * np.pi * freq_mod * t / sr)
    # Add harmonics
    chirp += 0.1 * np.sin(2 * np.pi * freq_mod * 2 * t / sr)
    chirp = chirp.astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, chirp, sr)
        tmp_path = f.name

    print(f"Demo audio written to: {tmp_path}")
    print("To run inference on a real trained model:")
    print("  engine = BirdLanguageInference('checkpoints/best_model.pth')")
    print("  results = engine.analyze('path/to/bird_sound.wav')")
    print("  print(json.dumps(results, indent=2))")

    return tmp_path


if __name__ == "__main__":
    demo_inference()
