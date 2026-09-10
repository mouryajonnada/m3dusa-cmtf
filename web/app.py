"""
Interactive Web Application Server
==================================
FastAPI backend for real-time fake news detection comparing:
- Baseline M3DUSA (Late Fusion)
- Novel Cross-Modal Transformer Fusion (CMTF)

Usage:
    python web/app.py [--port 8000]
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows UTF-8 stdout fix
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torch_geometric.data import Batch

from src.data.graph_builder import GraphBuilder, extract_hashtags
from src.data.preprocess import Tokenizer, clean_text
from src.models.model import FakeNewsDetector
from src.utils.config import load_config, merge_configs
from src.utils.logging import get_logger

log = get_logger("web_app")

app = FastAPI(
    title="M3DUSA vs. CMTF Fake News Detector",
    description="Interactive evaluation interface for Multimodal Fake News Detection",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models container
MODELS: dict[str, FakeNewsDetector] = {}
TOKENIZER: Optional[Tokenizer] = None
GRAPH_BUILDER: Optional[GraphBuilder] = None


def init_models():
    """Load and cache tokenizer, graph builder, and both trained models."""
    global MODELS, TOKENIZER, GRAPH_BUILDER

    if TOKENIZER is None:
        log.info("Loading Tokenizer and GraphBuilder...")
        TOKENIZER = Tokenizer(model_name="roberta-base", max_length=512)
        GRAPH_BUILDER = GraphBuilder()

    metadata = GRAPH_BUILDER.metadata()

    # 1. Baseline M3DUSA
    if "baseline" not in MODELS:
        ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / "baseline_m3dusa" / "best_model.pt"
        if ckpt_path.exists():
            log.info(f"Loading Baseline M3DUSA from {ckpt_path}...")
            cfg = load_config(str(PROJECT_ROOT / "experiments" / "baseline_m3dusa.yaml"))
            cfg = merge_configs(cfg, {"model": {"freeze_text_encoder": True}})
            model = FakeNewsDetector(cfg, metadata)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
            model.load_state_dict(state)
            model.eval()
            MODELS["baseline"] = model
            log.info("Baseline M3DUSA loaded successfully.")
        else:
            log.warning(f"Baseline checkpoint not found at {ckpt_path}")

    # 2. Novel CMTF
    if "cmtf" not in MODELS:
        ckpt_path = PROJECT_ROOT / "results" / "checkpoints" / "cross_modal_fusion" / "best_model.pt"
        if ckpt_path.exists():
            log.info(f"Loading CMTF model from {ckpt_path}...")
            cfg = load_config(str(PROJECT_ROOT / "experiments" / "cross_modal_fusion.yaml"))
            cfg = merge_configs(cfg, {"model": {"freeze_text_encoder": True}})
            model = FakeNewsDetector(cfg, metadata)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
            model.load_state_dict(state)
            model.eval()
            MODELS["cmtf"] = model
            log.info("CMTF model loaded successfully.")
        else:
            log.warning(f"CMTF checkpoint not found at {ckpt_path}")


@app.on_event("startup")
def startup_event():
    init_models()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str

class ModelPrediction(BaseModel):
    model_name: str
    prediction: str          # "REAL" or "FAKE"
    prob_fake: float
    prob_real: float
    confidence: float

class PredictResponse(BaseModel):
    input_text: str
    cleaned_text: str
    hashtags: list[str]
    graph_stats: dict
    baseline: Optional[ModelPrediction]
    cmtf: Optional[ModelPrediction]
    agreement: bool


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    init_models()
    cleaned = clean_text(req.text)
    if not cleaned:
        cleaned = req.text.strip()

    hashtags = extract_hashtags(cleaned)
    tokens = TOKENIZER.tokenize([cleaned])
    graph = GRAPH_BUILDER.build(cleaned)
    graph_batch = Batch.from_data_list([graph])

    batch = {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "graph_batch": graph_batch,
    }

    graph_stats = {
        "num_news_nodes": 1,
        "num_hashtag_nodes": int(graph["hashtag"].x.shape[0]),
        "num_edges": int(graph["news", "has_hashtag", "hashtag"].edge_index.shape[1]),
    }

    results = {}
    with torch.no_grad():
        for name, model in MODELS.items():
            logits = model(batch)
            probs = F.softmax(logits, dim=-1)[0].tolist()  # [fake_prob, real_prob]
            prob_fake, prob_real = float(probs[0]), float(probs[1])
            pred_label = "REAL" if prob_real >= prob_fake else "FAKE"
            confidence = max(prob_real, prob_fake)

            results[name] = ModelPrediction(
                model_name="Baseline M3DUSA" if name == "baseline" else "Cross-Modal Transformer Fusion (CMTF)",
                prediction=pred_label,
                prob_fake=round(prob_fake, 4),
                prob_real=round(prob_real, 4),
                confidence=round(confidence * 100, 2),
            )

    base_pred = results.get("baseline")
    cmtf_pred = results.get("cmtf")
    agreement = (base_pred.prediction == cmtf_pred.prediction) if (base_pred and cmtf_pred) else True

    return PredictResponse(
        input_text=req.text,
        cleaned_text=cleaned,
        hashtags=hashtags,
        graph_stats=graph_stats,
        baseline=base_pred,
        cmtf=cmtf_pred,
        agreement=agreement,
    )


@app.get("/api/samples")
def get_samples():
    """Curated real and fake political claims from PolitiFact for testing."""
    return [
        {
            "id": 1,
            "title": "Coronavirus was created in a military lab as a bioweapon.",
            "ground_truth": "FAKE",
            "category": "Disinformation / Conspiracy",
            "source": "PolitiFact debunked claim",
        },
        {
            "id": 2,
            "title": "NASA rover discovers evidence of ancient lake on surface of Mars.",
            "ground_truth": "REAL",
            "category": "Science / News",
            "source": "Fact-checked news report",
        },
        {
            "id": 3,
            "title": "Pope Francis shocks world, endorses Donald Trump for President.",
            "ground_truth": "FAKE",
            "category": "Electoral Fake News",
            "source": "Viral fake news story (2016/2020)",
        },
        {
            "id": 4,
            "title": "Congressional Budget Office releases nonpartisan estimate of federal deficit.",
            "ground_truth": "REAL",
            "category": "Government & Politics",
            "source": "Official government release",
        },
        {
            "id": 5,
            "title": "Secret audio proves White House ordered nationwide power grid shutdown.",
            "ground_truth": "FAKE",
            "category": "Clickbait / Sensationalism",
            "source": "Fabricated social media rumor",
        },
        {
            "id": 6,
            "title": "Department of Labor reports unemployment claims dropped to 50-year low.",
            "ground_truth": "REAL",
            "category": "Economics / Employment",
            "source": "Bureau of Labor Statistics",
        },
    ]


@app.get("/api/benchmark")
def get_benchmark():
    """Return metrics comparison between Baseline and CMTF."""
    csv_path = PROJECT_ROOT / "results" / "metrics" / "comparison.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Comparison table not found.")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


# Mount static files and results plots
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if PLOTS_DIR.exists():
    app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"message": "Frontend index.html under construction."})
    return FileResponse(index_file)


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the Fake News Detection Web App.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    args = parser.parse_args()

    import uvicorn
    print(f"\n========================================================")
    print(f"  M3DUSA vs. CMTF Web Interface")
    print(f"  Server running at: http://{args.host}:{args.port}")
    print(f"========================================================\n")
    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
