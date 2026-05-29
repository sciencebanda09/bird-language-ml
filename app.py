"""
Bird Language ML - REST API Server
FastAPI server exposing the trained model as an HTTP endpoint.

Endpoints:
  POST /analyze          - Analyze uploaded audio file
  GET  /health           - Health check
  GET  /species          - List supported species
  GET  /call_types       - List call type categories
  POST /analyze_url      - Analyze audio from URL

Run:
  uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

Docker:
  docker build -t bird-language-ml .
  docker run -p 8000:8000 bird-language-ml
"""

import os
import io
import json
import tempfile
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import soundfile as sf
import urllib.request

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import SPECIES_LIST, CALL_TYPES, CALL_TYPE_DESCRIPTIONS, infer_cfg
from src.inference import BirdLanguageInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bird-language-api")

# ──────────────────────────────────────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Bird Language ML API",
    description="Classify bird vocalizations and decode their communication meaning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine (loaded once at startup)
engine: Optional[BirdLanguageInference] = None
MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/best_model.pth")


@app.on_event("startup")
async def load_model():
    global engine
    if Path(MODEL_PATH).exists():
        logger.info(f"Loading model from {MODEL_PATH}")
        engine = BirdLanguageInference(MODEL_PATH)
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}. Train first with: python -c \"from src.train import train; train()\"")


# ──────────────────────────────────────────────────────────────────────────────
# Response Models
# ──────────────────────────────────────────────────────────────────────────────

class SpeciesResult(BaseModel):
    species: str
    confidence: float
    percentage: str


class CallTypeResult(BaseModel):
    call_type: str
    label: str
    confidence: float
    percentage: str
    description: str


class AnalysisResponse(BaseModel):
    filepath: str
    duration_seconds: float
    primary_species: dict
    primary_call_type: dict
    top_species: List[dict]
    top_call_types: List[dict]
    communication_meaning: str
    model_confidence: dict


class URLRequest(BaseModel):
    url: str
    overlap: float = 0.5
    aggregate_method: str = "mean"
    return_chunk_details: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": engine is not None,
        "model_path": MODEL_PATH,
        "supported_species": len(SPECIES_LIST),
        "supported_call_types": len(CALL_TYPES),
    }


@app.get("/species")
async def list_species():
    return {"species": SPECIES_LIST, "count": len(SPECIES_LIST)}


@app.get("/call_types")
async def list_call_types():
    return {
        "call_types": [
            {"id": ct, "label": ct.replace("_", " ").title(), "description": CALL_TYPE_DESCRIPTIONS[ct]}
            for ct in CALL_TYPES
        ]
    }


@app.post("/analyze", response_model=dict)
async def analyze_audio(
    file: UploadFile = File(...),
    overlap: float = Query(0.5, ge=0.0, le=0.9, description="Sliding window overlap"),
    aggregate_method: str = Query("mean", description="Aggregation: mean|max|geometric_mean"),
    return_timeline: bool = Query(False, description="Return per-chunk timeline"),
):
    """
    Analyze an uploaded audio file (WAV, MP3, FLAC, OGG).
    Returns species identification and communication type classification.
    """
    if engine is None:
        raise HTTPException(503, detail="Model not loaded. Train first: python -c \"from src.train import train; train()\"")

    # Validate file type
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        raise HTTPException(400, detail=f"Unsupported format: {suffix}. Use WAV, MP3, FLAC, or OGG.")

    # Save to temp file and analyze
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        results = engine.analyze(
            tmp_path,
            overlap=overlap,
            aggregate_method=aggregate_method,
            return_chunk_details=return_timeline,
        )
        results["filepath"] = file.filename  # Replace temp path with original name
        return results
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Analysis error: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/analyze_url", response_model=dict)
async def analyze_from_url(request: URLRequest):
    """Analyze audio from a public URL (e.g., xeno-canto recording link)."""
    if engine is None:
        raise HTTPException(503, detail="Model not loaded.")

    url = request.url
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, detail="Invalid URL. Must start with http:// or https://")

    try:
        suffix = Path(url.split("?")[0]).suffix or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(400, detail=f"Failed to download audio: {e}")

    try:
        results = engine.analyze(
            tmp_path,
            overlap=request.overlap,
            aggregate_method=request.aggregate_method,
            return_chunk_details=request.return_chunk_details,
        )
        results["source_url"] = url
        return results
    except Exception as e:
        raise HTTPException(500, detail=f"Analysis error: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.get("/")
async def root():
    return {
        "name": "Bird Language ML API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/analyze", "/analyze_url", "/species", "/call_types"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, workers=1)
