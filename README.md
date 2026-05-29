```
██████╗ ██╗██████╗ ██████╗     ██╗      █████╗ ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗  ██████╗ ███████╗
██╔══██╗██║██╔══██╗██╔══██╗    ██║     ██╔══██╗████╗  ██║██╔════╝ ██║   ██║██╔══██╗██╔════╝ ██╔════╝
██████╔╝██║██████╔╝██║  ██║    ██║     ███████║██╔██╗ ██║██║  ███╗██║   ██║███████║██║  ███╗█████╗  
██╔══██╗██║██╔══██╗██║  ██║    ██║     ██╔══██║██║╚██╗██║██║   ██║██║   ██║██╔══██║██║   ██║██╔══╝  
██████╔╝██║██║  ██║██████╔╝    ███████╗██║  ██║██║ ╚████║╚██████╔╝╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═════╝ ╚═╝╚═╝  ╚═╝╚═════╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
                                                                                                      
███╗   ███╗██╗                                                                                        
████╗ ████║██║                                                                                        
██╔████╔██║██║                                                                                        
██║╚██╔╝██║██║                                                                                        
██║ ╚═╝ ██║███████╗                                                                                   
╚═╝     ╚═╝╚══════╝                                                                                   
```

<div align="center">

🐦 **A production-grade deep learning system for classifying bird vocalizations**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Decode bird communication meaning — built with PyTorch, EfficientNet & xeno-canto data*

</div>

---

## 🧠 What It Does

Bird Language ML listens to bird audio recordings and classifies both the **species** and the **communication type** — telling you not just *which* bird is singing, but *why*.

---

## 🏗️ Architecture

```
Audio File (.wav / .mp3)
        │
        ▼
   Sliding Window (5s chunks, 50% overlap)
        │
        ▼
   Mel-Spectrogram (128 mel × 313 time frames)
   + SpecAugment (training only)
        │
        ▼
   EfficientNet-B0 Backbone (ImageNet pretrained)
   Global Average Pooling → 1280-dim embedding
        │
        ├──► Species Head     → 50 species classes
        └──► Call Type Head   → 9 communication categories
```

---

## 🔊 Call Type Categories

| Label | Meaning |
|----|---------|
| `alarm_call` | Predator warning, recruits mob response |
| `mating_song` | Mate attraction, fitness advertisement |
| `contact_call` | Flock cohesion while moving/foraging |
| `territorial_call` | Boundary defense against rivals |
| `distress_call` | Captured/cornered, recruits helpers |
| `begging_call` | Juveniles demanding food |
| `flight_call` | Migration contact, nocturnal navigation |
| `foraging_sound` | Food search, food discovery signal |
| `dawn_chorus` | Multi-function sunrise singing |

---

## 📊 Expected Performance

With 150 recordings/species across 20 species:

| Metric | Expected |
|--------|----------|
| Species Top-1 Acc | 75–85% |
| Species Top-5 Acc | 90–95% |
| Call Type Acc | 80–90% |
| Species Macro-F1 | 0.70–0.82 |

> Performance improves significantly with more data. BirdCLEF 2024 winners achieve >90% Top-1 on 200+ species.

---

## 🚀 Setup

```bash
git clone https://github.com/sciencebanda09/bird_language_ml
cd bird_language_ml
pip install -r requirements.txt
```

---

## 📥 Step 1 — Download Data

```bash
# Download top 20 India species from xeno-canto (free API)
python data/download_data.py --all_india --max_per_species 150

# Or specific species
python data/download_data.py --species "House Sparrow" "Asian Koel" "Common Myna"
```

Saves:
- `data/audio/` — MP3 files
- `data/metadata.csv` — species, call_type, filepath, quality, country

---

## 🏋️ Step 2 — Train

```bash
python -c "from src.train import train; train()"
```

Key training features:
- **EfficientNet-B0** pretrained backbone (10× faster convergence vs random init)
- **Differential learning rates** — backbone at 0.1× LR, heads at 1× LR
- **Mixed precision (AMP)** — 2× speedup on CUDA
- **SpecAugment** — frequency/time masking for regularization
- **Mixup** — interpolation between spectrograms for robustness
- **Cosine annealing with warmup** — stable convergence
- **Class-balanced sampling** — handles rare species
- **Early stopping** — prevents overfitting

Expected training time:
| Hardware | Time |
|---|---|
| GPU V100/A100 | ~2 hours / 50 epochs |
| GPU RTX 3080 | ~4 hours |
| CPU only | ~24 hours *(not recommended)* |

---

## 📈 Step 3 — Evaluate

```bash
python -c "from src.evaluate import evaluate; evaluate()"
```

Generates in `logs/evaluation/`:
- Per-class accuracy & F1 bar charts
- Confusion matrices (normalized)
- t-SNE embedding space visualization
- `eval_summary.json` with key metrics

---

## 🎯 Step 4 — Inference

### Python API

```python
from src.inference import BirdLanguageInference

engine = BirdLanguageInference("checkpoints/best_model.pth")
results = engine.analyze("path/to/bird_recording.wav")

print(results["primary_species"])
# → {"species": "Asian Koel", "confidence": 0.87, "percentage": "87.0%"}

print(results["primary_call_type"])
# → {"call_type": "mating_song", "label": "Mating Song", "confidence": 0.92}

print(results["communication_meaning"])
# → "The Asian Koel is advertising its fitness to potential mates..."

# Timeline for long recordings
results = engine.analyze("long_recording.wav", return_chunk_details=True)
for event in results["timeline"]:
    print(f"{event['time_start']:.1f}s: {event['species']} — {event['call_type']}")
```

### REST API

```bash
# Start server
uvicorn app:app --host 0.0.0.0 --port 8000

# Analyze audio file
curl -X POST http://localhost:8000/analyze \
  -F "file=@bird_recording.wav"

# Analyze from URL (e.g. xeno-canto)
curl -X POST http://localhost:8000/analyze_url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://xeno-canto.org/sounds/uploaded/..."}'

# List supported species
curl http://localhost:8000/species
```

### Docker

```bash
docker build -t bird-language-ml .
docker run -p 8000:8000 -v $(pwd)/checkpoints:/app/checkpoints bird-language-ml
```

---

## 📂 Project Structure

```
bird_language_ml/
├── config.py              ← All hyperparameters in one place
├── app.py                 ← FastAPI REST server
├── requirements.txt
├── Dockerfile
├── data/
│   ├── download_data.py   ← xeno-canto downloader
│   ├── audio/             ← downloaded .mp3 files
│   └── metadata.csv       ← filepath, species, call_type
├── src/
│   ├── dataset.py         ← DataLoader, augmentations, mel-spectrograms
│   ├── model.py           ← EfficientNet + dual-head architecture
│   ├── train.py           ← Full training loop
│   ├── evaluate.py        ← Metrics, confusion matrix, t-SNE
│   └── inference.py       ← Production inference engine
├── checkpoints/           ← Saved model weights
└── logs/                  ← Training curves, evaluation plots
```

---

## 🏆 Using BirdCLEF Data (Competition Dataset)

For production-grade models, use BirdCLEF from Kaggle:

```bash
kaggle competitions download -c birdclef-2024
```

Update `config.py` → `TrainingConfig.metadata_csv` to point to BirdCLEF metadata.

---

## 📄 License

MIT License. Audio data from xeno-canto is Creative Commons licensed — cite accordingly.

---

<div align="center">
Made with ❤️ by <a href="https://github.com/sciencebanda09">sciencebanda09</a>
</div>
